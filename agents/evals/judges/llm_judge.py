"""LLM-as-judge — evaluates agent outputs using claude -p.

Always uses the flagship model (claude-opus-4-6) for consistent evaluation
regardless of which model the agent under test used.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from agents.evals.lib.schemas import Check, JudgmentResult

RUBRICS_DIR = Path(__file__).resolve().parent / "rubrics"
JUDGE_MODEL = "claude-opus-4-6"
JUDGE_TIMEOUT = 120


def judge_agent(
    agent: str,
    raw_output: str,
    expected: dict[str, Any],
    question: str,
    mock_data: dict[str, Any],
    plan_json: str | None = None,
) -> JudgmentResult:
    """Evaluate an agent's output using LLM-as-judge.

    Args:
        agent: Agent name (planner, retrieval, graph, synthesis).
        raw_output: The agent's raw text output.
        expected: Expected section with quality_notes.
        question: Original user question.
        mock_data: Mock data used by the agent.
        plan_json: Planner output (context for retrieval/graph judges).

    Returns:
        JudgmentResult with LLM-assigned scores.
    """
    rubric_path = RUBRICS_DIR / f"{agent}.txt"
    if not rubric_path.exists():
        return JudgmentResult(
            agent=agent,
            judge_type="llm",
            passed=False,
            score=0.0,
            reasoning=f"Rubric not found: {rubric_path}",
        )

    rubric_template = rubric_path.read_text()
    quality_notes = expected.get("quality_notes", "Evaluate overall quality.")

    prompt = rubric_template.format(
        question=question,
        mock_data=json.dumps(mock_data, indent=2, default=str)[:3000],
        plan_json=plan_json or "(not available)",
        agent_output=raw_output[:4000],
        quality_notes=quality_notes,
    )

    judge_output = _call_judge(prompt)
    return _parse_judge_output(agent, judge_output)


def judge_end_to_end(
    raw_output: str,
    expected: dict[str, Any],
    question: str,
) -> JudgmentResult:
    """Evaluate the final synthesized report end-to-end."""
    rubric_path = RUBRICS_DIR / "end_to_end.txt"
    if not rubric_path.exists():
        return JudgmentResult(
            agent="end_to_end",
            judge_type="llm",
            passed=False,
            score=0.0,
            reasoning=f"Rubric not found: {rubric_path}",
        )

    rubric_template = rubric_path.read_text()
    quality_notes = expected.get("quality_notes", "Evaluate overall report quality.")

    prompt = rubric_template.format(
        question=question,
        agent_output=raw_output[:5000],
        quality_notes=quality_notes,
    )

    judge_output = _call_judge(prompt)
    return _parse_judge_output("end_to_end", judge_output)


def _call_judge(prompt: str) -> str:
    """Call claude -p with the judge prompt."""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", JUDGE_MODEL, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=JUDGE_TIMEOUT,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return '{"error": "judge timeout"}'
    except FileNotFoundError:
        return '{"error": "claude CLI not found"}'


def _parse_judge_output(agent: str, raw: str) -> JudgmentResult:
    """Parse the judge's JSON response into a JudgmentResult."""
    # Try to extract JSON from the output
    parsed = _extract_json(raw)
    if parsed is None:
        return JudgmentResult(
            agent=agent,
            judge_type="llm",
            passed=False,
            score=0.0,
            reasoning="Failed to parse judge output as JSON",
            raw_judge_output=raw[:1000],
        )

    scores = parsed.get("scores", {})
    overall = parsed.get("overall", 3)
    reasoning = parsed.get("reasoning", "")

    # Normalize 1-5 to 0.0-1.0
    normalized = (overall - 1) / 4 if isinstance(overall, (int, float)) else 0.5

    checks = [
        Check(name=k, passed=v >= 3, detail=f"Score: {v}/5")
        for k, v in scores.items()
        if isinstance(v, (int, float))
    ]

    return JudgmentResult(
        agent=agent,
        judge_type="llm",
        passed=normalized >= 0.5,
        score=round(normalized, 3),
        checks=checks,
        reasoning=reasoning,
        raw_judge_output=raw[:2000],
    )


def _extract_json(text: str) -> dict | None:
    """Extract JSON from text, handling code fences."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    # Find JSON object in text
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        if depth == 0:
            try:
                return json.loads(text[start : i + 1])
            except (json.JSONDecodeError, ValueError):
                return None
    return None
