#!/usr/bin/env python3
"""Evaluation runner: orchestrates fixture generation, report generation, and scoring.

Usage:
    # Run heuristic eval (no API key needed)
    python -m evals.run_eval --judge heuristic

    # Run LLM judge eval
    python -m evals.run_eval --judge llm --model haiku

    # Run both for comparison
    python -m evals.run_eval --judge both --model haiku

    # Test model swapping
    python -m evals.run_eval --judge llm --model haiku --compare sonnet

    # Broader batch (all fixture sizes)
    python -m evals.run_eval --judge heuristic --batch all

    # Investigate specific dimension
    python -m evals.run_eval --judge heuristic --dimension synthesis
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.framework import (
    Dimension, TestCase, EvalResult, BatchResult, RUBRICS,
)
from evals import heuristic


FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_DIR = Path(__file__).parent / "results"


def ensure_fixtures():
    """Generate fixtures if they don't exist."""
    if not (FIXTURES_DIR / "small_7d" / "manifest.json").exists():
        print("Generating fixtures...")
        subprocess.run(
            [sys.executable, str(FIXTURES_DIR / "generate_fixtures.py")],
            check=True,
        )
    else:
        print("Fixtures already exist.")


def generate_report(data_dir: Path) -> str:
    """Run generate-report.py against fixture data and return HTML."""
    script = PROJECT_ROOT / "scripts" / "generate-report.py"

    # We need to temporarily point the script's DATA_DIR at our fixture
    # The simplest way: symlink fixture files into data/ temporarily,
    # or just run the analysis directly
    # Instead, let's import and call the functions directly

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

    # Monkeypatch DATA_DIR in generate-report module
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_report", script)
    mod = importlib.util.module_from_spec(spec)

    # Override DATA_DIR before executing
    import types
    original_data_dir = None

    spec.loader.exec_module(mod)
    # Now patch and run
    mod.DATA_DIR = data_dir

    mod.setup_plot_style()

    df = mod.load_data()
    filtered_df = mod.filter_content(df)
    stats = mod.calculate_stats(df, filtered_df)

    plots = {
        "daily_traffic": mod.plot_daily_traffic(filtered_df),
        "top_articles": mod.plot_top_articles(filtered_df),
        "day_of_week": mod.plot_day_of_week(filtered_df),
    }

    spike_df = mod.detect_spikes(filtered_df)
    if len(spike_df) > 0:
        plots["spikes"] = mod.plot_spike_examples(filtered_df, spike_df)

    top_articles = filtered_df.groupby("article").agg({
        "views": "sum",
        "rank": "min"
    }).reset_index().sort_values("views", ascending=False)
    top_articles["rank"] = range(1, len(top_articles) + 1)

    consistency = filtered_df.groupby("article").agg({
        "date": "nunique",
        "views": "sum"
    }).reset_index()
    consistency.columns = ["article", "days_appeared", "total_views"]
    consistency = consistency.sort_values("days_appeared", ascending=False)

    try:
        dow_stats = mod.compute_day_of_week_stats(filtered_df)
    except (ValueError, ZeroDivisionError, AttributeError):
        dow_stats = {}
    narrative = mod.generate_narrative(stats, spike_df, dow_stats)

    html = mod.generate_html(stats, plots, top_articles, spike_df, consistency, narrative)
    return html


def get_test_cases(batch_mode: str = "medium") -> list[TestCase]:
    """Get test cases based on batch mode."""
    cases = []

    if batch_mode in ("small", "all"):
        cases.append(TestCase(
            case_id="small_7d",
            description="7-day small fixture with known spikes",
            data_file=str(FIXTURES_DIR / "small_7d"),
            script="generate-report.py",
            expected={"has_spikes": True, "min_articles": 10},
        ))

    if batch_mode in ("medium", "all", "default"):
        cases.append(TestCase(
            case_id="medium_30d",
            description="30-day medium fixture with full patterns",
            data_file=str(FIXTURES_DIR / "medium_30d"),
            script="generate-report.py",
            expected={"has_spikes": True, "min_articles": 20},
        ))

    if batch_mode in ("large", "all"):
        cases.append(TestCase(
            case_id="large_90d",
            description="90-day large fixture for stress testing",
            data_file=str(FIXTURES_DIR / "large_90d"),
            script="generate-report.py",
            expected={"has_spikes": True, "min_articles": 20},
        ))

    return cases


def run_heuristic_eval(cases: list[TestCase]) -> BatchResult:
    """Run heuristic evaluation on all test cases."""
    results = []
    for case in cases:
        print(f"\n  Evaluating {case.case_id}...")
        print(f"    Generating report from {case.data_file}...")
        report_html = generate_report(Path(case.data_file))
        print(f"    Report generated ({len(report_html):,} chars)")

        print(f"    Running heuristic scorer...")
        result = heuristic.evaluate(case, report_html)
        results.append(result)

        print(f"    Overall score: {result.overall_score:.3f}")
        for ds in result.dimension_scores:
            print(f"      {ds.dimension.value:15s}: {ds.score:.3f} (weighted: {ds.weighted_score:.3f})")

    return BatchResult(results=results, judge_type="heuristic", model="n/a")


def run_llm_eval(cases: list[TestCase], model_key: str, api_key: str = None) -> BatchResult:
    """Run LLM judge evaluation on all test cases."""
    from evals import judge

    pairs = []
    for case in cases:
        print(f"\n  Generating report for {case.case_id}...")
        report_html = generate_report(Path(case.data_file))
        pairs.append((case, report_html))

    print(f"\n  Running LLM judge (model: {model_key})...")
    return judge.evaluate_batch(pairs, model_key=model_key, api_key=api_key)


def print_batch_summary(batch: BatchResult, label: str = ""):
    """Pretty-print batch results."""
    header = f"\n{'='*60}"
    if label:
        header += f"\n  {label}"
    header += f"\n  Judge: {batch.judge_type} | Model: {batch.model}"
    header += f"\n  Cases: {len(batch.results)}"
    header += f"\n{'='*60}"
    print(header)

    dims = batch.mean_by_dimension()
    print(f"\n  Mean Overall Score: {batch.mean_overall:.3f}")
    print(f"\n  Scores by Dimension:")
    for dim_name, score in sorted(dims.items()):
        bar = "#" * int(score * 30)
        dot = "." * (30 - len(bar))
        weight = RUBRICS[Dimension(dim_name)]["weight"]
        print(f"    {dim_name:15s}: {score:.3f} [{bar}{dot}] (weight: {weight})")

    print(f"\n  Per-Case Results:")
    for r in batch.results:
        print(f"    {r.case_id:20s}: {r.overall_score:.3f}")

    print()


def investigate_synthesis(batch: BatchResult):
    """Deep dive into synthesis scores."""
    print("\n" + "=" * 60)
    print("  SYNTHESIS SCORE INVESTIGATION")
    print("=" * 60)

    for r in batch.results:
        synth = next(
            (ds for ds in r.dimension_scores if ds.dimension == Dimension.SYNTHESIS),
            None,
        )
        if not synth:
            continue

        print(f"\n  Case: {r.case_id} | Synthesis Score: {synth.score:.3f}")
        print(f"  {'─' * 50}")

        if synth.criteria_scores:
            print(f"  Criteria breakdown:")
            for criterion, score in synth.criteria_scores.items():
                status = "PASS" if score >= 0.8 else "WEAK" if score >= 0.4 else "FAIL"
                print(f"    [{status:4s}] {criterion}: {score:.3f}")

        if synth.details:
            print(f"\n  Details:")
            for d in synth.details:
                print(f"    - {d}")

    # Diagnosis
    print(f"\n  {'─' * 50}")
    print(f"  DIAGNOSIS:")
    print(f"  The generate-report.py script includes data-driven narrative")
    print(f"  paragraphs with causal explanations for traffic patterns,")
    print(f"  spike events, and day-of-week trends.")
    print(f"  If synthesis score is below 1.0, check:")
    print(f"    1. Causal language patterns (driven by, this suggests, likely)")
    print(f"    2. Narrative paragraph count (need 5+ with 20+ chars)")
    print(f"    3. Edge cases with empty spike data or small datasets")
    print()


def save_results(batch: BatchResult, filename: str):
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / filename
    with open(output_path, "w") as f:
        json.dump(batch.summary(), f, indent=2)
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Wikipedia analytics pipeline evaluation")
    parser.add_argument("--judge", choices=["heuristic", "llm", "both"],
                        default="heuristic", help="Judge type (default: heuristic)")
    parser.add_argument("--model", default="haiku",
                        help="LLM model for judge (default: haiku)")
    parser.add_argument("--compare", default=None,
                        help="Second model for comparison (model swap test)")
    parser.add_argument("--batch", default="medium",
                        choices=["small", "medium", "large", "all"],
                        help="Batch size (default: medium)")
    parser.add_argument("--dimension", default=None,
                        help="Focus on specific dimension (e.g. synthesis)")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to evals/results/")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    print("Wikipedia Analytics Pipeline Evaluation")
    print(f"Judge: {args.judge} | Batch: {args.batch} | Model: {args.model}")
    print()

    # Ensure fixtures exist
    ensure_fixtures()

    # Get test cases
    cases = get_test_cases(args.batch)
    print(f"Test cases: {len(cases)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batches = []

    # Run heuristic eval
    if args.judge in ("heuristic", "both"):
        print("\n--- Heuristic Evaluation ---")
        heur_batch = run_heuristic_eval(cases)
        print_batch_summary(heur_batch, "Heuristic Evaluation")
        batches.append(("heuristic", heur_batch))

        if args.save:
            save_results(heur_batch, f"heuristic_{args.batch}_{timestamp}.json")

        # Investigate synthesis if requested or if score is low
        synth_scores = [
            ds.score
            for r in heur_batch.results
            for ds in r.dimension_scores
            if ds.dimension == Dimension.SYNTHESIS
        ]
        if args.dimension == "synthesis" or (synth_scores and max(synth_scores) < 0.75):
            investigate_synthesis(heur_batch)

    # Run LLM eval
    if args.judge in ("llm", "both"):
        if not api_key:
            print("\nWARNING: No API key found. Set ANTHROPIC_API_KEY or use --api-key.")
            print("Skipping LLM judge. Use --judge heuristic for API-free evaluation.")
        else:
            print(f"\n--- LLM Judge Evaluation (model: {args.model}) ---")
            llm_batch = run_llm_eval(cases, args.model, api_key)
            print_batch_summary(llm_batch, f"LLM Judge ({args.model})")
            batches.append((args.model, llm_batch))

            if args.save:
                save_results(llm_batch, f"llm_{args.model}_{args.batch}_{timestamp}.json")

            # Model comparison
            if args.compare:
                print(f"\n--- LLM Judge Evaluation (model: {args.compare}) ---")
                cmp_batch = run_llm_eval(cases, args.compare, api_key)
                print_batch_summary(cmp_batch, f"LLM Judge ({args.compare})")
                batches.append((args.compare, cmp_batch))

                if args.save:
                    save_results(cmp_batch, f"llm_{args.compare}_{args.batch}_{timestamp}.json")

                # Print comparison
                print("\n" + "=" * 60)
                print(f"  MODEL COMPARISON: {args.model} vs {args.compare}")
                print("=" * 60)
                dims1 = llm_batch.mean_by_dimension()
                dims2 = cmp_batch.mean_by_dimension()
                print(f"\n  {'Dimension':15s} | {args.model:>8s} | {args.compare:>8s} | {'Delta':>8s}")
                print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
                for dim in sorted(set(dims1) | set(dims2)):
                    s1 = dims1.get(dim, 0)
                    s2 = dims2.get(dim, 0)
                    delta = s2 - s1
                    sign = "+" if delta > 0 else ""
                    print(f"  {dim:15s} | {s1:8.3f} | {s2:8.3f} | {sign}{delta:7.3f}")
                print(f"  {'OVERALL':15s} | {llm_batch.mean_overall:8.3f} | {cmp_batch.mean_overall:8.3f} | {'+' if cmp_batch.mean_overall > llm_batch.mean_overall else ''}{cmp_batch.mean_overall - llm_batch.mean_overall:7.3f}")
                print()

    # Summary
    if len(batches) > 1:
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        for label, batch in batches:
            print(f"  {label:20s}: overall={batch.mean_overall:.3f}")
        print()


if __name__ == "__main__":
    main()
