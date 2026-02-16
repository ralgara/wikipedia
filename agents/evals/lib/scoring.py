"""Scoring, aggregation, and terminal output for eval results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.evals.lib.schemas import CaseResult, EvalRun, JudgmentResult

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def aggregate_case(case_result: CaseResult) -> dict[str, float]:
    """Compute aggregate scores for a case."""
    agent_scores: dict[str, float] = {}
    for agent_name, judgments in case_result.judgments.items():
        if judgments:
            agent_scores[agent_name] = sum(j.score for j in judgments) / len(judgments)

    e2e_scores = [j.score for j in case_result.end_to_end]
    if e2e_scores:
        agent_scores["end_to_end"] = sum(e2e_scores) / len(e2e_scores)

    all_scores = list(agent_scores.values())
    agent_scores["overall"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return agent_scores


def create_run(
    config: dict[str, Any],
    case_results: list[CaseResult],
) -> EvalRun:
    """Create an EvalRun with a timestamped ID."""
    ts = datetime.now(timezone.utc)
    run_id = f"run_{ts.strftime('%Y%m%d_%H%M%S')}"
    return EvalRun(
        run_id=run_id,
        timestamp=ts.isoformat(),
        config=config,
        case_results=case_results,
    )


def save_run(run: EvalRun) -> Path:
    """Save eval run to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / f"{run.run_id}.json"
    filepath.write_text(json.dumps(_run_to_dict(run), indent=2))
    return filepath


def load_run(filepath: Path) -> dict[str, Any]:
    """Load eval run from JSON."""
    return json.loads(filepath.read_text())


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------


def print_results(run: EvalRun, threshold: float = 0.6) -> None:
    """Print results table to terminal."""
    try:
        _print_rich(run, threshold)
    except ImportError:
        _print_plain(run, threshold)


def _print_rich(run: EvalRun, threshold: float) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(f"\n[bold]Eval Results: {run.run_id}[/bold]")
    console.print(f"Config: {json.dumps(run.config, default=str)}\n")

    for cr in run.case_results:
        scores = aggregate_case(cr)
        table = Table(title=f"Case: {cr.case_id} ({cr.case_type})")
        table.add_column("Agent", style="cyan")
        table.add_column("Det.", justify="right")
        table.add_column("LLM", justify="right")
        table.add_column("Combined", justify="right")
        table.add_column("Status")

        for agent_name, judgments in cr.judgments.items():
            det_scores = [j.score for j in judgments if j.judge_type == "deterministic"]
            llm_scores = [j.score for j in judgments if j.judge_type == "llm"]
            det = sum(det_scores) / len(det_scores) if det_scores else None
            llm = sum(llm_scores) / len(llm_scores) if llm_scores else None
            combined = scores.get(agent_name, 0)
            status = "[green]PASS[/green]" if combined >= threshold else "[red]FAIL[/red]"
            table.add_row(
                agent_name,
                f"{det:.2f}" if det is not None else "—",
                f"{llm:.2f}" if llm is not None else "—",
                f"{combined:.2f}",
                status,
            )

        # End-to-end
        if cr.end_to_end:
            e2e = scores.get("end_to_end", 0)
            e2e_status = "[green]PASS[/green]" if e2e >= threshold else "[red]FAIL[/red]"
            table.add_row("end_to_end", "—", f"{e2e:.2f}", f"{e2e:.2f}", e2e_status)

        # Overall
        overall = scores.get("overall", 0)
        overall_status = "[bold green]PASS[/bold green]" if overall >= threshold else "[bold red]FAIL[/bold red]"
        table.add_section()
        table.add_row("[bold]OVERALL[/bold]", "", "", f"[bold]{overall:.2f}[/bold]", overall_status)

        console.print(table)
        console.print()

    # Summary
    total = len(run.case_results)
    passed = sum(1 for cr in run.case_results if aggregate_case(cr).get("overall", 0) >= threshold)
    console.print(f"[bold]Summary: {passed}/{total} cases passed (threshold: {threshold})[/bold]\n")


def _print_plain(run: EvalRun, threshold: float) -> None:
    """Fallback plain text output."""
    print(f"\n{'='*50}")
    print(f"Eval Results: {run.run_id}")
    print(f"{'='*50}")

    for cr in run.case_results:
        scores = aggregate_case(cr)
        print(f"\nCase: {cr.case_id} ({cr.case_type})")
        print(f"{'─'*40}")
        for agent_name in cr.judgments:
            combined = scores.get(agent_name, 0)
            status = "PASS" if combined >= threshold else "FAIL"
            print(f"  {agent_name:<15} {combined:.2f}  {status}")
        overall = scores.get("overall", 0)
        status = "PASS" if overall >= threshold else "FAIL"
        print(f"  {'OVERALL':<15} {overall:.2f}  {status}")

    total = len(run.case_results)
    passed = sum(1 for cr in run.case_results if aggregate_case(cr).get("overall", 0) >= threshold)
    print(f"\nSummary: {passed}/{total} cases passed (threshold: {threshold})\n")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def print_comparison(run1_path: Path, run2_path: Path) -> None:
    """Compare two eval runs and print deltas."""
    r1 = load_run(run1_path)
    r2 = load_run(run2_path)

    print(f"\nComparing: {r1['run_id']} vs {r2['run_id']}")
    print(f"{'='*60}")

    # Build score maps
    scores1 = _extract_scores(r1)
    scores2 = _extract_scores(r2)

    all_cases = sorted(set(scores1.keys()) | set(scores2.keys()))
    for case_id in all_cases:
        s1 = scores1.get(case_id, {})
        s2 = scores2.get(case_id, {})
        all_agents = sorted(set(s1.keys()) | set(s2.keys()))
        print(f"\nCase: {case_id}")
        print(f"  {'Agent':<15} {'Run 1':>8} {'Run 2':>8} {'Delta':>8}")
        print(f"  {'─'*42}")
        for agent in all_agents:
            v1 = s1.get(agent)
            v2 = s2.get(agent)
            v1_str = f"{v1:.2f}" if v1 is not None else "—"
            v2_str = f"{v2:.2f}" if v2 is not None else "—"
            if v1 is not None and v2 is not None:
                delta = v2 - v1
                arrow = "▲" if delta > 0.02 else ("▼" if delta < -0.02 else " ")
                delta_str = f"{delta:+.2f} {arrow}"
            else:
                delta_str = "—"
            print(f"  {agent:<15} {v1_str:>8} {v2_str:>8} {delta_str:>8}")

    print()


def _extract_scores(run_dict: dict) -> dict[str, dict[str, float]]:
    """Extract per-case, per-agent scores from a run dict."""
    scores = {}
    for cr in run_dict.get("case_results", []):
        case_id = cr["case_id"]
        case_scores = {}
        for agent_name, judgments in cr.get("judgments", {}).items():
            if judgments:
                case_scores[agent_name] = sum(j["score"] for j in judgments) / len(judgments)
        scores[case_id] = case_scores
    return scores


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _run_to_dict(run: EvalRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "timestamp": run.timestamp,
        "config": run.config,
        "case_results": [_case_result_to_dict(cr) for cr in run.case_results],
    }


def _case_result_to_dict(cr: CaseResult) -> dict[str, Any]:
    return {
        "case_id": cr.case_id,
        "case_type": cr.case_type,
        "agent_results": {
            name: {
                "agent": ar.agent,
                "model": ar.model,
                "exit_code": ar.exit_code,
                "duration_seconds": ar.duration_seconds,
                "error": ar.error,
                # raw_output omitted for size — available in temp dir
            }
            for name, ar in cr.agent_results.items()
        },
        "judgments": {
            name: [_judgment_to_dict(j) for j in judgments]
            for name, judgments in cr.judgments.items()
        },
        "end_to_end": [_judgment_to_dict(j) for j in cr.end_to_end],
        "aggregate": aggregate_case(cr),
    }


def _judgment_to_dict(j: JudgmentResult) -> dict[str, Any]:
    return {
        "agent": j.agent,
        "judge_type": j.judge_type,
        "passed": j.passed,
        "score": j.score,
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in j.checks],
        "reasoning": j.reasoning,
    }
