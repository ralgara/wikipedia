"""Core evaluation framework: dimensions, rubrics, and scoring."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Dimension(str, Enum):
    """Scoring dimensions for report evaluation."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    SYNTHESIS = "synthesis"
    FILTERING = "filtering"
    VISUALIZATION = "visualization"


RUBRICS = {
    Dimension.ACCURACY: {
        "description": "Statistical correctness of computed metrics",
        "criteria": [
            "Total views sum matches raw data",
            "Top articles ranking is correct",
            "Spike detection z-scores are valid",
            "Date ranges are accurate",
            "Averages and aggregations are correct",
        ],
        "weight": 0.25,
    },
    Dimension.COMPLETENESS: {
        "description": "Coverage of all expected report sections and data",
        "criteria": [
            "Overview metrics present (total views, days, unique articles, avg daily)",
            "Top articles table with ranks and views",
            "Day-of-week traffic pattern analysis",
            "Spike detection with multipliers",
            "Consistency analysis (articles appearing most days)",
            "Daily traffic time series chart",
        ],
        "weight": 0.20,
    },
    Dimension.SYNTHESIS: {
        "description": "How well the report connects data into coherent insights",
        "criteria": [
            "Narrative connects spikes to patterns (not just listing data)",
            "Cross-references between dimensions (e.g. spikes + consistency)",
            "Contextual explanations for trends (weekday/weekend differences)",
            "Identifies meaningful patterns beyond raw numbers",
            "Summary ties individual findings into overall story",
        ],
        "weight": 0.25,
    },
    Dimension.FILTERING: {
        "description": "Proper exclusion of non-content pages",
        "criteria": [
            "Main_Page excluded from rankings",
            "Special: pages excluded",
            "Talk pages excluded",
            "Non-content namespaces excluded",
            "Flagged articles handled correctly",
        ],
        "weight": 0.15,
    },
    Dimension.VISUALIZATION: {
        "description": "Quality and appropriateness of charts",
        "criteria": [
            "Charts have readable labels and titles",
            "Axes are properly formatted (e.g. M/K suffixes)",
            "Color scheme is consistent",
            "Chart types match data (bar for ranking, line for time series)",
            "Charts render without errors",
        ],
        "weight": 0.15,
    },
}


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    dimension: Dimension
    score: float  # 0.0 to 1.0
    details: list[str] = field(default_factory=list)
    criteria_scores: dict[str, float] = field(default_factory=dict)

    @property
    def weight(self) -> float:
        return RUBRICS[self.dimension]["weight"]

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class EvalResult:
    """Complete evaluation result for one test case."""
    case_id: str
    dimension_scores: list[DimensionScore]
    overall_score: float = 0.0
    judge_type: str = "unknown"
    model: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.dimension_scores and self.overall_score == 0.0:
            self.overall_score = self.compute_overall()

    def compute_overall(self) -> float:
        total_weight = sum(ds.weight for ds in self.dimension_scores)
        if total_weight == 0:
            return 0.0
        return sum(ds.weighted_score for ds in self.dimension_scores) / total_weight

    def summary(self) -> dict:
        return {
            "case_id": self.case_id,
            "overall_score": round(self.overall_score, 3),
            "judge_type": self.judge_type,
            "model": self.model,
            "dimensions": {
                ds.dimension.value: {
                    "score": round(ds.score, 3),
                    "weighted": round(ds.weighted_score, 3),
                    "details": ds.details,
                }
                for ds in self.dimension_scores
            },
        }


@dataclass
class TestCase:
    """A single evaluation test case."""
    case_id: str
    description: str
    data_file: str  # path to fixture data
    script: str  # which script to evaluate
    expected: dict = field(default_factory=dict)  # expected properties


@dataclass
class BatchResult:
    """Results from a batch evaluation run."""
    results: list[EvalResult]
    judge_type: str
    model: str

    @property
    def mean_overall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.overall_score for r in self.results) / len(self.results)

    def mean_by_dimension(self) -> dict[str, float]:
        dims: dict[str, list[float]] = {}
        for r in self.results:
            for ds in r.dimension_scores:
                dims.setdefault(ds.dimension.value, []).append(ds.score)
        return {d: sum(scores) / len(scores) for d, scores in dims.items()}

    def summary(self) -> dict:
        return {
            "num_cases": len(self.results),
            "mean_overall": round(self.mean_overall, 3),
            "judge_type": self.judge_type,
            "model": self.model,
            "by_dimension": {
                k: round(v, 3) for k, v in self.mean_by_dimension().items()
            },
            "cases": [r.summary() for r in self.results],
        }
