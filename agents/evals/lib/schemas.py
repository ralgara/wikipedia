"""Data models for the evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    """A single evaluation case with question, mock data, and expected outputs."""

    id: str
    type: str  # comparison, trend, spike, ranking, edge_case
    question: str
    mock_data_inline: dict[str, Any]  # pageviews + optional ontology
    ground_truth: dict[str, Any]  # computed answers for deterministic checks
    expected: dict[str, dict[str, Any]]  # per-agent expectations
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalCase:
        return cls(
            id=d["id"],
            type=d["type"],
            question=d["question"],
            mock_data_inline=d.get("mock_data_inline", {}),
            ground_truth=d.get("ground_truth", {}),
            expected=d.get("expected", {}),
            description=d.get("description", ""),
        )


@dataclass
class AgentResult:
    """Output from running a single agent."""

    agent: str  # planner, retrieval, graph, synthesis
    model: str  # e.g. "claude-opus-4-6", "claude-haiku-4-5-20251001"
    raw_output: str
    exit_code: int
    duration_seconds: float
    error: str | None = None


@dataclass
class Check:
    """A single deterministic check result."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class JudgmentResult:
    """Result from a judge evaluating an agent output."""

    agent: str
    judge_type: str  # "deterministic" or "llm"
    passed: bool
    score: float  # 0.0 to 1.0
    checks: list[Check] = field(default_factory=list)
    reasoning: str = ""
    raw_judge_output: str = ""


@dataclass
class CaseResult:
    """Full evaluation result for a single case."""

    case_id: str
    case_type: str
    agent_results: dict[str, AgentResult]  # keyed by agent name
    judgments: dict[str, list[JudgmentResult]]  # keyed by agent name
    end_to_end: list[JudgmentResult] = field(default_factory=list)


@dataclass
class EvalRun:
    """A complete evaluation run across multiple cases."""

    run_id: str
    timestamp: str
    config: dict[str, Any]  # models, judges, filters
    case_results: list[CaseResult] = field(default_factory=list)
