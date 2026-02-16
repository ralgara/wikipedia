"""LLM judge: evaluates pipeline outputs using Claude as an evaluator.

Supports model swapping to compare different judge models.
Falls back to heuristic scoring when no API key is available.
"""

import json
import os
import re
import time
from pathlib import Path

from evals.framework import (
    Dimension, DimensionScore, EvalResult, TestCase, RUBRICS, BatchResult,
)


# Default models for judge evaluation
MODELS = {
    "haiku": "claude-3-5-haiku-20241022",
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-0-20250514",
}

DEFAULT_MODEL = "haiku"


def _build_judge_prompt(dimension: Dimension, report_excerpt: str, data_summary: str) -> str:
    """Build the evaluation prompt for a specific dimension."""
    rubric = RUBRICS[dimension]

    criteria_text = "\n".join(f"  - {c}" for c in rubric["criteria"])

    return f"""You are an expert evaluator assessing a Wikipedia pageviews analytics report.

## Dimension: {dimension.value.upper()}
{rubric['description']}

## Criteria
{criteria_text}

## Data Summary
{data_summary}

## Report Excerpt
{report_excerpt}

## Instructions
Score this report on the "{dimension.value}" dimension from 0.0 to 1.0.
For each criterion, provide a score (0.0-1.0) and brief justification.

Respond in JSON format:
{{
  "dimension": "{dimension.value}",
  "overall_score": <float 0.0-1.0>,
  "criteria_scores": {{
    "<criterion_description>": <float 0.0-1.0>,
    ...
  }},
  "details": ["<finding 1>", "<finding 2>", ...],
  "reasoning": "<brief overall assessment>"
}}"""


def _extract_report_text(report_html: str, max_chars: int = 8000) -> str:
    """Extract readable text from HTML report for LLM evaluation."""
    # Remove base64 images (they're huge)
    text = re.sub(r'data:image/png;base64,[A-Za-z0-9+/=]+', '[CHART_IMAGE]', report_html)
    # Remove CSS
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
    # Remove tags but keep content
    text = re.sub(r'<[^>]+>', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_chars]


def _summarize_data(data_dir: Path) -> str:
    """Create a brief summary of fixture data for the judge."""
    import pandas as pd

    records = []
    for f in sorted(data_dir.glob("pageviews_*.json")):
        with open(f) as fh:
            records.extend(json.load(fh))

    if not records:
        return "No data available"

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    lines = [
        f"Date range: {df['date'].min().date()} to {df['date'].max().date()}",
        f"Total records: {len(df):,}",
        f"Unique articles: {df['article'].nunique():,}",
        f"Total views: {df['views'].sum():,}",
        f"Days: {df['date'].nunique()}",
    ]

    # Check manifest for known properties
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        props = manifest.get("known_properties", {})
        if props.get("spike_articles"):
            lines.append(f"Known spike articles: {list(props['spike_articles'].keys())}")
        if props.get("non_content_articles"):
            lines.append(f"Non-content articles in data: {props['non_content_articles']}")

    return "\n".join(lines)


def _call_llm(prompt: str, model_key: str = DEFAULT_MODEL, api_key: str = None) -> dict:
    """Call Claude API and parse JSON response."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package required: pip install anthropic")

    model_id = MODELS.get(model_key, model_key)
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key

    client = anthropic.Anthropic(**client_kwargs)

    response = client.messages.create(
        model=model_id,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text

    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    # Try parsing full response as JSON
    # Find the outermost { ... }
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start >= 0 and brace_end > brace_start:
        return json.loads(text[brace_start:brace_end + 1])

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")


def evaluate_dimension(
    dimension: Dimension,
    report_html: str,
    data_dir: Path,
    model_key: str = DEFAULT_MODEL,
    api_key: str = None,
) -> DimensionScore:
    """Evaluate a single dimension using the LLM judge."""
    report_text = _extract_report_text(report_html)
    data_summary = _summarize_data(data_dir)
    prompt = _build_judge_prompt(dimension, report_text, data_summary)

    result = _call_llm(prompt, model_key=model_key, api_key=api_key)

    return DimensionScore(
        dimension=dimension,
        score=float(result.get("overall_score", 0.0)),
        details=result.get("details", []),
        criteria_scores={k: float(v) for k, v in result.get("criteria_scores", {}).items()},
    )


def evaluate(
    case: TestCase,
    report_html: str,
    model_key: str = DEFAULT_MODEL,
    api_key: str = None,
) -> EvalResult:
    """Run full LLM judge evaluation on a report."""
    data_dir = Path(case.data_file)

    scores = []
    for dim in Dimension:
        try:
            score = evaluate_dimension(
                dim, report_html, data_dir,
                model_key=model_key, api_key=api_key,
            )
            scores.append(score)
            # Rate limiting
            time.sleep(1)
        except Exception as e:
            # On failure, record a zero score with error detail
            scores.append(DimensionScore(
                dimension=dim,
                score=0.0,
                details=[f"LLM judge error: {e}"],
            ))

    result = EvalResult(
        case_id=case.case_id,
        dimension_scores=scores,
        judge_type="llm",
        model=MODELS.get(model_key, model_key),
    )
    result.overall_score = result.compute_overall()
    return result


def evaluate_batch(
    cases: list[tuple[TestCase, str]],  # (case, report_html) pairs
    model_key: str = DEFAULT_MODEL,
    api_key: str = None,
) -> BatchResult:
    """Run LLM judge evaluation on a batch of test cases."""
    results = []
    for i, (case, report_html) in enumerate(cases):
        print(f"  [{i+1}/{len(cases)}] Evaluating {case.case_id}...")
        result = evaluate(case, report_html, model_key=model_key, api_key=api_key)
        results.append(result)

    return BatchResult(
        results=results,
        judge_type="llm",
        model=MODELS.get(model_key, model_key),
    )
