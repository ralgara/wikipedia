"""Heuristic scorer: evaluates pipeline outputs using structural analysis.

No LLM required. Checks structural properties of generated reports
and computed statistics against known fixture properties.
"""

import json
import re
from pathlib import Path

import pandas as pd
import numpy as np

from evals.framework import (
    Dimension, DimensionScore, EvalResult, TestCase, RUBRICS,
)


def _load_fixture_data(data_dir: Path) -> pd.DataFrame:
    """Load all JSON files from a fixture directory into a DataFrame."""
    records = []
    for f in sorted(data_dir.glob("pageviews_*.json")):
        with open(f) as fh:
            records.extend(json.load(fh))
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_manifest(data_dir: Path) -> dict:
    """Load fixture manifest with known properties."""
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def score_accuracy(df: pd.DataFrame, report_html: str, manifest: dict) -> DimensionScore:
    """Score statistical accuracy of the report."""
    details = []
    criteria_scores = {}

    # Check total views is mentioned
    total_views = df["views"].sum()
    # Look for formatted numbers in report (e.g. "123.4M" or "1.2B")
    has_total = bool(re.search(r"Total Views", report_html))
    criteria_scores["total_views_present"] = 1.0 if has_total else 0.0
    if has_total:
        details.append("Total views metric is present")
    else:
        details.append("MISSING: Total views metric not found")

    # Check date range
    min_date = df["date"].min().strftime("%Y-%m-%d")
    max_date = df["date"].max().strftime("%Y-%m-%d")
    has_dates = min_date in report_html and max_date in report_html
    criteria_scores["date_range_accurate"] = 1.0 if has_dates else 0.0
    if has_dates:
        details.append(f"Date range {min_date} to {max_date} is accurate")
    else:
        details.append(f"Date range not verified in report")

    # Check that spike articles appear in spike section if manifest has them
    spike_info = manifest.get("known_properties", {}).get("spike_articles", {})
    if spike_info:
        spike_found = sum(1 for a in spike_info if a.replace("_", " ") in report_html or a in report_html)
        spike_ratio = spike_found / len(spike_info) if spike_info else 0
        criteria_scores["spike_detection"] = spike_ratio
        details.append(f"Spike detection: {spike_found}/{len(spike_info)} known spikes found")
    else:
        criteria_scores["spike_detection"] = 0.5  # neutral if no manifest
        details.append("No spike manifest to validate against")

    # Check number of days
    num_days = df["date"].nunique()
    has_days = str(num_days) in report_html
    criteria_scores["days_count"] = 1.0 if has_days else 0.0
    if has_days:
        details.append(f"Days count ({num_days}) matches")
    else:
        details.append(f"Days count ({num_days}) not found in report")

    # Check unique articles count is reasonable
    criteria_scores["article_count"] = 0.5  # partial - hard to verify exactly
    details.append("Article count: partial verification")

    score = sum(criteria_scores.values()) / len(criteria_scores)
    return DimensionScore(
        dimension=Dimension.ACCURACY,
        score=score,
        details=details,
        criteria_scores=criteria_scores,
    )


def score_completeness(report_html: str) -> DimensionScore:
    """Score completeness of report sections."""
    details = []
    criteria_scores = {}

    sections = {
        "overview_metrics": [
            ("Total Views", "total views"),
            ("Days Analyzed", "days analyzed"),
            ("Unique Articles", "unique articles"),
            ("Avg Daily Views", "avg daily views"),
        ],
        "top_articles_table": [("Top Articles", "top articles section")],
        "day_of_week": [("Day of Week", "day of week analysis")],
        "spike_detection": [("Spike", "spike detection section")],
        "consistency": [("Consistent", "consistency analysis")],
        "daily_traffic_chart": [("Daily.*Traffic", "daily traffic chart")],
    }

    for section_key, checks in sections.items():
        found = 0
        for pattern, name in checks:
            if re.search(pattern, report_html, re.IGNORECASE):
                found += 1
        ratio = found / len(checks)
        criteria_scores[section_key] = ratio
        if ratio >= 1.0:
            details.append(f"PASS: {section_key} fully present")
        elif ratio > 0:
            details.append(f"PARTIAL: {section_key} ({found}/{len(checks)} checks)")
        else:
            details.append(f"MISSING: {section_key}")

    # Check for embedded images (base64 charts)
    chart_count = len(re.findall(r'data:image/png;base64,', report_html))
    expected_charts = 3  # daily_traffic, top_articles, day_of_week
    chart_ratio = min(chart_count / expected_charts, 1.0)
    criteria_scores["charts_embedded"] = chart_ratio
    details.append(f"Charts: {chart_count}/{expected_charts} expected charts embedded")

    score = sum(criteria_scores.values()) / len(criteria_scores)
    return DimensionScore(
        dimension=Dimension.COMPLETENESS,
        score=score,
        details=details,
        criteria_scores=criteria_scores,
    )


def score_synthesis(report_html: str) -> DimensionScore:
    """Score how well the report synthesizes data into insights.

    This is the hardest dimension to score heuristically.
    We look for narrative indicators: explanatory text, cross-references,
    contextual language beyond bare data presentation.
    """
    details = []
    criteria_scores = {}

    # Check for narrative/explanatory text (not just tables and charts)
    text_blocks = re.findall(r'<p[^>]*>([^<]{20,})</p>', report_html)
    narrative_count = len(text_blocks)
    # Expect at least some explanatory paragraphs
    criteria_scores["narrative_present"] = min(narrative_count / 5, 1.0)
    details.append(f"Narrative paragraphs: {narrative_count} found")

    # Check for contextual language (words that indicate explanation)
    insight_words = [
        "pattern", "trend", "typical", "unusual", "spike", "increase",
        "decrease", "average", "compared", "higher", "lower", "weekend",
        "weekday", "consistent", "notable", "significant",
    ]
    text_lower = report_html.lower()
    insight_count = sum(1 for w in insight_words if w in text_lower)
    criteria_scores["contextual_language"] = min(insight_count / 8, 1.0)
    details.append(f"Contextual/insight language: {insight_count}/{len(insight_words)} terms found")

    # Check for cross-referencing between sections
    # Look for mentions that connect data (e.g. "weekend traffic" near day-of-week)
    cross_ref_patterns = [
        r"weekend.*traffic|traffic.*weekend",
        r"spike.*date|date.*spike",
        r"consistently.*appear|appear.*consistently",
        r"standard deviation|deviation.*above",
    ]
    cross_refs = sum(1 for p in cross_ref_patterns if re.search(p, text_lower))
    criteria_scores["cross_references"] = min(cross_refs / 3, 1.0)
    details.append(f"Cross-references: {cross_refs}/{len(cross_ref_patterns)} patterns found")

    # Check for "why" explanations (the weakest area of template reports)
    why_patterns = [
        r"because|due to|driven by|caused by|result of",
        r"this (suggests|indicates|shows|means)",
        r"likely|probably|may be|could be",
    ]
    why_count = sum(1 for p in why_patterns if re.search(p, text_lower))
    criteria_scores["causal_explanations"] = min(why_count / 2, 1.0)
    details.append(f"Causal/explanatory language: {why_count}/{len(why_patterns)} patterns")

    # Check for summary/conclusion
    has_summary = bool(re.search(r"summary|conclusion|key finding|takeaway|overview", text_lower))
    criteria_scores["summary_present"] = 1.0 if has_summary else 0.0
    details.append(f"Summary/conclusion section: {'present' if has_summary else 'MISSING'}")

    score = sum(criteria_scores.values()) / len(criteria_scores)
    return DimensionScore(
        dimension=Dimension.SYNTHESIS,
        score=score,
        details=details,
        criteria_scores=criteria_scores,
    )


def score_filtering(df: pd.DataFrame, report_html: str, manifest: dict) -> DimensionScore:
    """Score content filtering correctness."""
    details = []
    criteria_scores = {}

    non_content = manifest.get("known_properties", {}).get("non_content_articles", [])
    flagged = manifest.get("known_properties", {}).get("flagged_articles", [])

    # Check Main_Page is NOT in top articles table
    # (it should be filtered out)
    main_page_in_rankings = bool(re.search(r'<td>.*Main.Page.*</td>', report_html))
    criteria_scores["main_page_filtered"] = 0.0 if main_page_in_rankings else 1.0
    details.append(f"Main_Page filtered: {'YES' if not main_page_in_rankings else 'NO - appears in rankings'}")

    # Check Special: pages filtered
    special_in_report = bool(re.search(r'Special:', report_html))
    criteria_scores["special_filtered"] = 0.0 if special_in_report else 1.0
    details.append(f"Special: pages filtered: {'YES' if not special_in_report else 'NO'}")

    # Check other non-content namespaces
    namespaces = ["User:", "Template:", "Category:", "Portal:", "Help:"]
    ns_found = sum(1 for ns in namespaces if ns in report_html)
    criteria_scores["namespaces_filtered"] = 1.0 if ns_found == 0 else max(0, 1.0 - ns_found / len(namespaces))
    details.append(f"Non-content namespaces in report: {ns_found}/{len(namespaces)}")

    # Check talk pages filtered
    talk_in_report = bool(re.search(r'_talk:', report_html))
    criteria_scores["talk_filtered"] = 0.0 if talk_in_report else 1.0
    details.append(f"Talk pages filtered: {'YES' if not talk_in_report else 'NO'}")

    # Check flagged articles
    if flagged:
        flagged_found = sum(1 for a in flagged if a in report_html)
        criteria_scores["flagged_handled"] = 1.0 if flagged_found == 0 else 0.0
        details.append(f"Flagged articles in report: {flagged_found}/{len(flagged)}")
    else:
        criteria_scores["flagged_handled"] = 1.0
        details.append("No flagged articles in fixture")

    score = sum(criteria_scores.values()) / len(criteria_scores)
    return DimensionScore(
        dimension=Dimension.FILTERING,
        score=score,
        details=details,
        criteria_scores=criteria_scores,
    )


def score_visualization(report_html: str) -> DimensionScore:
    """Score visualization quality."""
    details = []
    criteria_scores = {}

    # Check for chart images
    charts = re.findall(r'data:image/png;base64,', report_html)
    chart_count = len(charts)
    criteria_scores["charts_present"] = min(chart_count / 3, 1.0)
    details.append(f"Charts rendered: {chart_count}")

    # Check for chart alt text (accessibility)
    alt_texts = re.findall(r'alt="([^"]+)"', report_html)
    chart_alts = [a for a in alt_texts if a not in ["", " "]]
    criteria_scores["alt_text"] = min(len(chart_alts) / 3, 1.0)
    details.append(f"Alt text on images: {len(chart_alts)}")

    # Check that charts are within section containers
    chart_in_section = len(re.findall(r'class="chart".*?data:image/png', report_html, re.DOTALL))
    criteria_scores["chart_containers"] = min(chart_in_section / 3, 1.0)
    details.append(f"Charts in containers: {chart_in_section}")

    # Check for proper HTML structure
    has_responsive = "max-width: 100%" in report_html or "max-width:100%" in report_html
    criteria_scores["responsive"] = 1.0 if has_responsive else 0.0
    details.append(f"Responsive images: {'YES' if has_responsive else 'NO'}")

    # Check color scheme consistency
    has_colors = all(c in report_html for c in ["#1a1a2e", "#e94560", "#00adb5"])
    criteria_scores["color_scheme"] = 1.0 if has_colors else 0.5
    details.append(f"Consistent color scheme: {'YES' if has_colors else 'partial'}")

    score = sum(criteria_scores.values()) / len(criteria_scores)
    return DimensionScore(
        dimension=Dimension.VISUALIZATION,
        score=score,
        details=details,
        criteria_scores=criteria_scores,
    )


def evaluate(case: TestCase, report_html: str) -> EvalResult:
    """Run full heuristic evaluation on a report."""
    data_dir = Path(case.data_file)
    df = _load_fixture_data(data_dir)
    manifest = _load_manifest(data_dir)

    scores = [
        score_accuracy(df, report_html, manifest),
        score_completeness(report_html),
        score_synthesis(report_html),
        score_filtering(df, report_html, manifest),
        score_visualization(report_html),
    ]

    result = EvalResult(
        case_id=case.case_id,
        dimension_scores=scores,
        judge_type="heuristic",
        model="n/a",
    )
    result.overall_score = result.compute_overall()
    return result
