#!/usr/bin/env python3
"""Tests for the evaluation framework.

Tests framework logic, heuristic scoring, and LLM judge code paths
(with mocked API calls for the judge).

Usage:
    python -m evals.test_eval
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.framework import (
    Dimension, DimensionScore, EvalResult, TestCase, BatchResult, RUBRICS,
)
from evals import heuristic
from evals import judge


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_framework_basics():
    """Test framework data structures."""
    print("test_framework_basics...")

    # DimensionScore
    ds = DimensionScore(
        dimension=Dimension.ACCURACY,
        score=0.85,
        details=["test detail"],
        criteria_scores={"c1": 0.9, "c2": 0.8},
    )
    assert ds.weight == 0.25
    assert abs(ds.weighted_score - 0.2125) < 0.001

    # EvalResult
    scores = [
        DimensionScore(dimension=Dimension.ACCURACY, score=0.9),
        DimensionScore(dimension=Dimension.COMPLETENESS, score=1.0),
        DimensionScore(dimension=Dimension.SYNTHESIS, score=0.67),
        DimensionScore(dimension=Dimension.FILTERING, score=1.0),
        DimensionScore(dimension=Dimension.VISUALIZATION, score=0.95),
    ]
    result = EvalResult(case_id="test", dimension_scores=scores, judge_type="test")
    assert result.overall_score > 0
    summary = result.summary()
    assert "overall_score" in summary
    assert "dimensions" in summary
    assert "synthesis" in summary["dimensions"]

    # BatchResult
    batch = BatchResult(results=[result, result], judge_type="test", model="test-model")
    assert batch.mean_overall == result.overall_score
    dims = batch.mean_by_dimension()
    assert "accuracy" in dims
    batch_summary = batch.summary()
    assert batch_summary["num_cases"] == 2

    print("  PASSED")


def test_rubrics_complete():
    """Test that rubrics cover all dimensions."""
    print("test_rubrics_complete...")

    for dim in Dimension:
        assert dim in RUBRICS, f"Missing rubric for {dim}"
        rubric = RUBRICS[dim]
        assert "description" in rubric
        assert "criteria" in rubric
        assert "weight" in rubric
        assert len(rubric["criteria"]) >= 3

    # Weights should sum to 1.0
    total_weight = sum(r["weight"] for r in RUBRICS.values())
    assert abs(total_weight - 1.0) < 0.001, f"Weights sum to {total_weight}, expected 1.0"

    print("  PASSED")


def test_heuristic_scoring():
    """Test heuristic scorer against medium fixture."""
    print("test_heuristic_scoring...")

    fixture_dir = FIXTURES_DIR / "medium_30d"
    if not fixture_dir.exists():
        print("  SKIP (fixtures not generated)")
        return

    # Generate a report first
    from evals.run_eval import generate_report
    report_html = generate_report(fixture_dir)
    assert len(report_html) > 1000

    case = TestCase(
        case_id="test_medium",
        description="test",
        data_file=str(fixture_dir),
        script="generate-report.py",
    )

    result = heuristic.evaluate(case, report_html)

    # Basic checks
    assert result.overall_score > 0.5, f"Overall score too low: {result.overall_score}"
    assert len(result.dimension_scores) == 5

    # Check each dimension scored
    for ds in result.dimension_scores:
        assert 0.0 <= ds.score <= 1.0, f"{ds.dimension}: score {ds.score} out of range"
        assert len(ds.details) > 0, f"{ds.dimension}: no details"

    # Check synthesis dimension has improved with narrative
    synth = next(ds for ds in result.dimension_scores if ds.dimension == Dimension.SYNTHESIS)

    # Check specific synthesis criteria
    assert "causal_explanations" in synth.criteria_scores
    assert synth.criteria_scores["causal_explanations"] >= 0.5, \
        f"Report should have causal language, got {synth.criteria_scores['causal_explanations']}"

    print(f"  PASSED (overall={result.overall_score:.3f}, synthesis={synth.score:.3f})")


def test_model_swapping():
    """Test model swapping in the LLM judge (mocked API calls)."""
    print("test_model_swapping...")

    # Mock responses for different models
    def make_mock_response(model_id, score_offset=0.0):
        """Create a mock LLM response."""
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = json.dumps({
            "dimension": "accuracy",
            "overall_score": 0.85 + score_offset,
            "criteria_scores": {"c1": 0.9, "c2": 0.8},
            "details": [f"Evaluated with {model_id}"],
            "reasoning": f"Assessment from {model_id}",
        })
        mock_response.content = [mock_content]
        return mock_response

    # Test with haiku model
    with patch("anthropic.Anthropic") as MockClient:
        client_instance = MockClient.return_value
        client_instance.messages.create.return_value = make_mock_response("haiku")

        score = judge.evaluate_dimension(
            Dimension.ACCURACY,
            "<html><body>test report</body></html>",
            FIXTURES_DIR / "medium_30d",
            model_key="haiku",
            api_key="test-key",
        )
        assert score.dimension == Dimension.ACCURACY
        assert abs(score.score - 0.85) < 0.001
        assert "haiku" in score.details[0]

        # Verify the correct model was passed
        call_kwargs = client_instance.messages.create.call_args
        assert call_kwargs.kwargs["model"] == judge.MODELS["haiku"]

    # Test with sonnet model
    with patch("anthropic.Anthropic") as MockClient:
        client_instance = MockClient.return_value
        client_instance.messages.create.return_value = make_mock_response("sonnet", 0.05)

        score = judge.evaluate_dimension(
            Dimension.ACCURACY,
            "<html><body>test report</body></html>",
            FIXTURES_DIR / "medium_30d",
            model_key="sonnet",
            api_key="test-key",
        )
        assert abs(score.score - 0.90) < 0.001
        assert "sonnet" in score.details[0]

        call_kwargs = client_instance.messages.create.call_args
        assert call_kwargs.kwargs["model"] == judge.MODELS["sonnet"]

    # Test with opus model
    with patch("anthropic.Anthropic") as MockClient:
        client_instance = MockClient.return_value
        client_instance.messages.create.return_value = make_mock_response("opus", 0.10)

        score = judge.evaluate_dimension(
            Dimension.ACCURACY,
            "<html><body>test report</body></html>",
            FIXTURES_DIR / "medium_30d",
            model_key="opus",
            api_key="test-key",
        )
        assert abs(score.score - 0.95) < 0.001
        assert "opus" in score.details[0]

        call_kwargs = client_instance.messages.create.call_args
        assert call_kwargs.kwargs["model"] == judge.MODELS["opus"]

    print("  PASSED (haiku, sonnet, opus model IDs verified)")


def test_llm_judge_full_eval_mocked():
    """Test full LLM judge evaluation with mocked API."""
    print("test_llm_judge_full_eval_mocked...")

    fixture_dir = FIXTURES_DIR / "medium_30d"
    if not fixture_dir.exists():
        print("  SKIP (fixtures not generated)")
        return

    dim_scores = {
        "accuracy": 0.88,
        "completeness": 0.95,
        "synthesis": 0.67,
        "filtering": 0.92,
        "visualization": 0.90,
    }

    call_count = [0]

    def mock_create(**kwargs):
        dim_name = list(Dimension)[call_count[0] % len(Dimension)].value
        call_count[0] += 1
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = json.dumps({
            "dimension": dim_name,
            "overall_score": dim_scores.get(dim_name, 0.8),
            "criteria_scores": {"c1": 0.9},
            "details": [f"LLM assessed {dim_name}"],
            "reasoning": "test",
        })
        mock_response.content = [mock_content]
        return mock_response

    case = TestCase(
        case_id="test_llm",
        description="test",
        data_file=str(fixture_dir),
        script="generate-report.py",
    )

    with patch("anthropic.Anthropic") as MockClient:
        client_instance = MockClient.return_value
        client_instance.messages.create.side_effect = mock_create

        with patch("evals.judge.time.sleep"):  # skip delays
            result = judge.evaluate(case, "<html>test</html>", model_key="haiku", api_key="test-key")

    assert result.judge_type == "llm"
    assert result.model == judge.MODELS["haiku"]
    assert len(result.dimension_scores) == 5
    assert result.overall_score > 0

    # Verify synthesis score came through
    synth = next(ds for ds in result.dimension_scores if ds.dimension == Dimension.SYNTHESIS)
    assert abs(synth.score - 0.67) < 0.001, f"Expected synthesis=0.67, got {synth.score}"

    print(f"  PASSED (overall={result.overall_score:.3f}, synthesis={synth.score:.3f})")


def test_batch_eval_mocked():
    """Test batch evaluation with model comparison."""
    print("test_batch_eval_mocked...")

    fixture_dir = FIXTURES_DIR / "medium_30d"
    if not fixture_dir.exists():
        print("  SKIP (fixtures not generated)")
        return

    call_count = [0]

    def mock_create(**kwargs):
        model_id = kwargs.get("model", "unknown")
        dim_idx = call_count[0] % len(Dimension)
        call_count[0] += 1
        # Sonnet scores slightly higher than haiku
        bonus = 0.05 if "sonnet" in model_id else 0.0
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = json.dumps({
            "dimension": list(Dimension)[dim_idx].value,
            "overall_score": 0.80 + bonus,
            "criteria_scores": {"c1": 0.85 + bonus},
            "details": [f"Scored by {model_id}"],
            "reasoning": "test",
        })
        mock_response.content = [mock_content]
        return mock_response

    cases = [
        TestCase(case_id="batch_1", description="t1", data_file=str(fixture_dir), script="generate-report.py"),
        TestCase(case_id="batch_2", description="t2", data_file=str(fixture_dir), script="generate-report.py"),
    ]

    with patch("anthropic.Anthropic") as MockClient:
        client_instance = MockClient.return_value
        client_instance.messages.create.side_effect = mock_create

        with patch("evals.judge.time.sleep"):
            # Evaluate with haiku
            haiku_batch = judge.evaluate_batch(
                [(c, "<html>test</html>") for c in cases],
                model_key="haiku", api_key="test-key",
            )

            # Reset counter for sonnet
            call_count[0] = 0
            sonnet_batch = judge.evaluate_batch(
                [(c, "<html>test</html>") for c in cases],
                model_key="sonnet", api_key="test-key",
            )

    assert haiku_batch.judge_type == "llm"
    assert haiku_batch.model == judge.MODELS["haiku"]
    assert len(haiku_batch.results) == 2
    assert sonnet_batch.model == judge.MODELS["sonnet"]

    # Sonnet should score slightly higher
    assert sonnet_batch.mean_overall >= haiku_batch.mean_overall

    print(f"  PASSED (haiku={haiku_batch.mean_overall:.3f}, sonnet={sonnet_batch.mean_overall:.3f})")


def test_judge_prompt_construction():
    """Test that judge prompts are well-formed for each dimension."""
    print("test_judge_prompt_construction...")

    for dim in Dimension:
        prompt = judge._build_judge_prompt(dim, "sample report text", "sample data summary")
        assert dim.value.upper() in prompt
        assert "sample report text" in prompt
        assert "sample data summary" in prompt
        assert "JSON" in prompt
        # Check all criteria are in the prompt
        for criterion in RUBRICS[dim]["criteria"]:
            assert criterion in prompt, f"Missing criterion in {dim}: {criterion}"

    print("  PASSED")


def main():
    print("=" * 60)
    print("  Eval Framework Tests")
    print("=" * 60)

    tests = [
        test_framework_basics,
        test_rubrics_complete,
        test_heuristic_scoring,
        test_model_swapping,
        test_llm_judge_full_eval_mocked,
        test_batch_eval_mocked,
        test_judge_prompt_construction,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
