#!/usr/bin/env python3
"""Run evaluation suite for the multi-agent orchestrator.

Usage:
    ./agents/evals/run-evals.py                              # Run all cases
    ./agents/evals/run-evals.py --case greenland_vs_sweden    # Single case
    ./agents/evals/run-evals.py --type comparison             # Single type
    ./agents/evals/run-evals.py --agent planner               # Single agent
    ./agents/evals/run-evals.py --model retrieval:haiku       # Model swap
    ./agents/evals/run-evals.py --judge deterministic         # Fast, free
    ./agents/evals/run-evals.py --limit 5                     # Quick iteration
    ./agents/evals/run-evals.py --compare results/r1.json results/r2.json
"""

import argparse
import os
import sys
from pathlib import Path

# Allow imports from project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    print("pyyaml required: uv pip install pyyaml", file=sys.stderr)
    sys.exit(1)

from agents.evals.lib.schemas import AgentResult, CaseResult, EvalCase
from agents.evals.lib.runner import run_agent, run_pipeline
from agents.evals.lib.scoring import (
    create_run,
    print_comparison,
    print_results,
    save_run,
)
from agents.evals.judges.deterministic import judge_agent as deterministic_judge

CASES_DIR = SCRIPT_DIR / "cases"
GENERATED_DIR = CASES_DIR / "_generated"


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------


def load_cases(
    case_filter: str | None = None,
    type_filter: str | None = None,
    limit: int | None = None,
) -> list[EvalCase]:
    """Load eval cases from YAML files."""
    cases = []

    # Load hand-crafted cases
    for f in sorted(CASES_DIR.glob("*.yaml")):
        cases.append(_load_yaml_case(f))

    # Load generated cases
    if GENERATED_DIR.exists():
        for f in sorted(GENERATED_DIR.glob("*.yaml")):
            cases.append(_load_yaml_case(f))

    # Apply filters
    if case_filter:
        cases = [c for c in cases if c.id == case_filter]
    if type_filter:
        cases = [c for c in cases if c.type == type_filter]
    if limit:
        cases = cases[:limit]

    return cases


def _load_yaml_case(filepath: Path) -> EvalCase:
    with open(filepath) as f:
        data = yaml.safe_load(f)
    return EvalCase.from_dict(data)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_case(
    case: EvalCase,
    models: dict[str, str],
    agents_filter: list[str] | None,
    judge_types: list[str],
    timeout: int,
) -> CaseResult:
    """Run agents and judges for a single eval case."""
    agents_to_run = agents_filter or ["planner", "retrieval", "graph", "synthesis"]

    # Only run agents that have expectations defined
    agents_to_run = [a for a in agents_to_run if a in case.expected]

    # Run the pipeline
    agent_results = run_pipeline(
        question=case.question,
        mock_data=case.mock_data_inline,
        models=models,
        agents=agents_to_run,
        timeout=timeout,
    )

    # Judge each agent
    judgments: dict[str, list] = {}
    for agent_name, result in agent_results.items():
        agent_judgments = []
        expected = case.expected.get(agent_name, {})

        if "deterministic" in judge_types:
            j = deterministic_judge(
                agent=agent_name,
                raw_output=result.raw_output,
                expected=expected,
                ground_truth=case.ground_truth,
            )
            agent_judgments.append(j)

        if "llm" in judge_types:
            try:
                from agents.evals.judges.llm_judge import judge_agent as llm_judge_fn
                j = llm_judge_fn(
                    agent=agent_name,
                    raw_output=result.raw_output,
                    expected=expected,
                    question=case.question,
                    mock_data=case.mock_data_inline,
                )
                agent_judgments.append(j)
            except ImportError:
                pass  # LLM judge not yet implemented

        judgments[agent_name] = agent_judgments

    # End-to-end LLM judge
    e2e_judgments = []
    if "llm" in judge_types and "end_to_end" in case.expected:
        synthesis_result = agent_results.get("synthesis")
        if synthesis_result:
            try:
                from agents.evals.judges.llm_judge import judge_end_to_end
                j = judge_end_to_end(
                    raw_output=synthesis_result.raw_output,
                    expected=case.expected["end_to_end"],
                    question=case.question,
                )
                e2e_judgments.append(j)
            except ImportError:
                pass

    return CaseResult(
        case_id=case.id,
        case_type=case.type,
        agent_results=agent_results,
        judgments=judgments,
        end_to_end=e2e_judgments,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_model_args(model_args: list[str] | None) -> dict[str, str]:
    """Parse --model agent:model pairs."""
    if not model_args:
        return {}
    models = {}
    for m in model_args:
        if ":" not in m:
            print(f"Invalid --model format: {m} (expected agent:model)", file=sys.stderr)
            sys.exit(1)
        agent, model = m.split(":", 1)
        models[agent] = model
    return models


def main():
    parser = argparse.ArgumentParser(
        description="Run eval suite for multi-agent orchestrator"
    )
    parser.add_argument("--case", help="Run only this case ID")
    parser.add_argument("--type", help="Run only cases of this type")
    parser.add_argument("--agent", action="append", help="Run only these agents (repeatable)")
    parser.add_argument("--model", action="append", help="Model override as agent:model (repeatable)")
    parser.add_argument("--judge", choices=["deterministic", "llm", "all"], default="all",
                        help="Which judge(s) to use")
    parser.add_argument("--limit", type=int, help="Max number of cases to run")
    parser.add_argument("--timeout", type=int, default=120, help="Per-agent timeout in seconds")
    parser.add_argument("--threshold", type=float, default=0.6, help="Pass/fail threshold")
    parser.add_argument("--compare", nargs=2, metavar="FILE", help="Compare two result files")
    parser.add_argument("--list", action="store_true", help="List available cases and exit")
    args = parser.parse_args()

    # Compare mode
    if args.compare:
        print_comparison(Path(args.compare[0]), Path(args.compare[1]))
        return

    # Load cases
    cases = load_cases(case_filter=args.case, type_filter=args.type, limit=args.limit)

    # List mode
    if args.list:
        print(f"Available cases: {len(cases)}")
        for c in cases:
            print(f"  {c.id:<45} {c.type:<15} {c.description[:50]}")
        return

    if not cases:
        print("No eval cases found. Run generate-cases.py first.", file=sys.stderr)
        sys.exit(1)

    # Parse config
    models = parse_model_args(args.model)
    judge_types = ["deterministic", "llm"] if args.judge == "all" else [args.judge]
    agents_filter = args.agent  # None means all

    print(f"Running {len(cases)} case(s)")
    print(f"Judges: {judge_types}")
    if models:
        print(f"Model overrides: {models}")
    if agents_filter:
        print(f"Agents: {agents_filter}")
    print()

    # Run evaluations
    case_results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.id} ({case.type})...")
        cr = evaluate_case(
            case=case,
            models=models,
            agents_filter=agents_filter,
            judge_types=judge_types,
            timeout=args.timeout,
        )
        case_results.append(cr)

    # Build run and save
    config = {
        "models": models or "default",
        "judges": judge_types,
        "case_filter": args.case,
        "type_filter": args.type,
        "agents_filter": agents_filter,
    }
    run = create_run(config, case_results)
    filepath = save_run(run)

    # Print results
    print_results(run, threshold=args.threshold)
    print(f"Results saved: {filepath}")

    # Exit code: 0 if all pass, 1 if any fail
    from agents.evals.lib.scoring import aggregate_case
    all_pass = all(
        aggregate_case(cr).get("overall", 0) >= args.threshold
        for cr in case_results
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
