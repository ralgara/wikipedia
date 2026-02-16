"""Deterministic (rule-based) judge for agent outputs.

Runs fast, free, zero-LLM checks: JSON validity, required fields,
numeric accuracy, keyword presence, table detection.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.evals.lib.schemas import Check, JudgmentResult


def judge_agent(
    agent: str,
    raw_output: str,
    expected: dict[str, Any],
    ground_truth: dict[str, Any] | None = None,
) -> JudgmentResult:
    """Run all deterministic checks for an agent output.

    Args:
        agent: Agent name (planner, retrieval, graph, synthesis).
        raw_output: The agent's raw text output.
        expected: The expected section from the eval case for this agent.
        ground_truth: The ground_truth section from the eval case.

    Returns:
        JudgmentResult with per-check details.
    """
    checks: list[Check] = []

    # --- JSON validity ---
    if expected.get("valid_json"):
        parsed = _try_parse_json(raw_output)
        checks.append(Check(
            name="valid_json",
            passed=parsed is not None,
            detail="Valid JSON" if parsed else "Failed to parse as JSON",
        ))
    else:
        parsed = _try_parse_json(raw_output)

    # --- Agent-specific checks ---
    if agent == "planner":
        checks.extend(_check_planner(raw_output, parsed, expected))
    elif agent == "retrieval":
        checks.extend(_check_retrieval(raw_output, parsed, expected, ground_truth))
    elif agent == "graph":
        checks.extend(_check_graph(raw_output, parsed, expected))
    elif agent == "synthesis":
        checks.extend(_check_synthesis(raw_output, expected))

    passed_count = sum(1 for c in checks if c.passed)
    total = len(checks)
    score = passed_count / total if total > 0 else 0.0

    return JudgmentResult(
        agent=agent,
        judge_type="deterministic",
        passed=all(c.passed for c in checks),
        score=score,
        checks=checks,
        reasoning=f"{passed_count}/{total} checks passed",
    )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _try_parse_json(text: str) -> dict | list | None:
    """Try to parse JSON from text, handling markdown code fences."""
    # Strip markdown code fences if present
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        # Remove first and last lines (fences)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON within the text
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find matching end by counting braces
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    break
    return None


# ---------------------------------------------------------------------------
# Planner checks
# ---------------------------------------------------------------------------


def _check_planner(
    raw: str, parsed: dict | list | None, expected: dict[str, Any]
) -> list[Check]:
    checks = []

    # Required articles
    if "required_articles" in expected and isinstance(parsed, dict):
        articles_field = parsed.get("articles", [])
        if isinstance(articles_field, list):
            found = {a.lower().replace(" ", "_") for a in articles_field}
            required = {a.lower().replace(" ", "_") for a in expected["required_articles"]}
            missing = required - found
            checks.append(Check(
                name="required_articles",
                passed=len(missing) == 0,
                detail=f"Missing: {missing}" if missing else f"Found all: {required}",
            ))
        else:
            checks.append(Check(
                name="required_articles",
                passed=False,
                detail="'articles' field is not a list",
            ))

    # Time range
    if "time_range_includes" in expected and isinstance(parsed, dict):
        time_range = parsed.get("time_range", {})
        raw_str = json.dumps(time_range) if time_range else raw
        required_years = expected["time_range_includes"]
        found_all = all(str(y) in raw_str for y in required_years)
        checks.append(Check(
            name="time_range_coverage",
            passed=found_all,
            detail=f"Required years {required_years} in range: {time_range}",
        ))

    # Min tasks
    if "min_tasks" in expected and isinstance(parsed, dict):
        tasks = parsed.get("tasks", [])
        n = len(tasks) if isinstance(tasks, list) else 0
        checks.append(Check(
            name="min_tasks",
            passed=n >= expected["min_tasks"],
            detail=f"Found {n} tasks, need >= {expected['min_tasks']}",
        ))

    return checks


# ---------------------------------------------------------------------------
# Retrieval checks
# ---------------------------------------------------------------------------


def _check_retrieval(
    raw: str,
    parsed: dict | list | None,
    expected: dict[str, Any],
    ground_truth: dict[str, Any] | None,
) -> list[Check]:
    checks = []

    # Must identify higher article (comparison cases)
    if "must_identify_higher" in expected:
        higher = expected["must_identify_higher"]
        # Check if the output mentions this article as having more views
        # Look in both parsed JSON and raw text
        text_lower = raw.lower()
        higher_lower = higher.lower().replace("_", " ")
        # Heuristic: higher article should appear near words like "more", "higher", "greater"
        checks.append(Check(
            name="higher_article_correct",
            passed=higher_lower in text_lower or higher.lower() in text_lower,
            detail=f"Expected higher: {higher}",
        ))

    # Must contain specific values
    if "must_contain_values" in expected:
        for val in expected["must_contain_values"]:
            found = str(val) in raw
            checks.append(Check(
                name=f"contains_value_{val}",
                passed=found,
                detail=f"Value {val} {'found' if found else 'not found'} in output",
            ))

    # Must identify spike month (spike cases)
    if "must_identify_spike_month" in expected:
        month = expected["must_identify_spike_month"]
        found = month in raw
        checks.append(Check(
            name="identifies_spike_month",
            passed=found,
            detail=f"Spike month {month} {'found' if found else 'not found'}",
        ))

    # Must include top 1 article (ranking cases)
    if "must_include_top_1" in expected:
        top1 = expected["must_include_top_1"]
        found = top1 in raw or top1.replace("_", " ") in raw
        checks.append(Check(
            name="includes_top_1",
            passed=found,
            detail=f"Top article {top1} {'found' if found else 'not found'}",
        ))

    # Must identify trend (trend cases)
    if "must_identify_trend" in expected:
        direction = expected["must_identify_trend"].lower()
        text_lower = raw.lower()
        # Look for trend-related words
        trend_words = {
            "increasing": ["increasing", "growing", "rising", "upward", "growth", "increased", "more"],
            "decreasing": ["decreasing", "declining", "falling", "downward", "decline", "decreased", "fewer"],
            "stable": ["stable", "steady", "consistent", "flat", "unchanged", "similar"],
        }
        words = trend_words.get(direction, [direction])
        found = any(w in text_lower for w in words)
        checks.append(Check(
            name="identifies_trend",
            passed=found,
            detail=f"Expected trend: {direction}",
        ))

    return checks


# ---------------------------------------------------------------------------
# Graph checks
# ---------------------------------------------------------------------------


def _check_graph(
    raw: str, parsed: dict | list | None, expected: dict[str, Any]
) -> list[Check]:
    checks = []

    if "must_mention" in expected:
        text_lower = raw.lower()
        for keyword in expected["must_mention"]:
            found = keyword.lower() in text_lower
            checks.append(Check(
                name=f"mentions_{keyword.lower()}",
                passed=found,
                detail=f"'{keyword}' {'found' if found else 'not found'}",
            ))

    if expected.get("must_reference_events"):
        # Look for any event-like terms
        event_keywords = [
            "acquisition", "sovereignty", "nato", "accession",
            "climate", "mineral", "rare earth", "event", "spike",
        ]
        text_lower = raw.lower()
        found_any = any(kw in text_lower for kw in event_keywords)
        checks.append(Check(
            name="references_events",
            passed=found_any,
            detail="Found event references" if found_any else "No event keywords found",
        ))

    return checks


# ---------------------------------------------------------------------------
# Synthesis checks
# ---------------------------------------------------------------------------


def _check_synthesis(raw: str, expected: dict[str, Any]) -> list[Check]:
    checks = []

    if expected.get("must_contain_table"):
        # Look for markdown table pattern: | ... | ... |
        has_table = bool(re.search(r"\|.*\|.*\|", raw))
        checks.append(Check(
            name="table_present",
            passed=has_table,
            detail="Markdown table found" if has_table else "No table detected",
        ))

    if "must_name_articles" in expected:
        text_lower = raw.lower()
        for article in expected["must_name_articles"]:
            name = article.lower().replace("_", " ")
            found = name in text_lower or article.lower() in text_lower
            checks.append(Check(
                name=f"names_{article.lower()}",
                passed=found,
                detail=f"'{article}' {'found' if found else 'not found'}",
            ))

    if "must_contain_values" in expected:
        for val in expected["must_contain_values"]:
            found = str(val) in raw
            checks.append(Check(
                name=f"contains_{val}",
                passed=found,
                detail=f"Value {val} {'found' if found else 'not found'}",
            ))

    if "min_word_count" in expected:
        word_count = len(raw.split())
        threshold = expected["min_word_count"]
        checks.append(Check(
            name="min_word_count",
            passed=word_count >= threshold,
            detail=f"{word_count} words (need >= {threshold})",
        ))

    return checks
