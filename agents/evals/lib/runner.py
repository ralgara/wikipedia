"""Agent runner — invokes agents via claude -p subprocess.

Replicates the prompts from orchestrator.sh as Python templates,
giving control over model selection, mock data injection, and timeouts.

NOTE: Prompts here are extracted from agents/orchestrator.sh.
If you change prompts in orchestrator.sh, update them here too.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any

from agents.evals.lib.schemas import AgentResult

DEFAULT_TIMEOUT = 120  # seconds

# ---------------------------------------------------------------------------
# Prompt templates (extracted from orchestrator.sh)
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """You are a query planner for a Wikipedia analytics system.

Decompose this question into structured sub-tasks. Output ONLY valid JSON.

Question: {question}

Output format:
{{
  "articles": ["Article1", "Article2"],
  "time_range": {{"start": "YYYY", "end": "YYYY"}},
  "tasks": [
    {{"agent": "retrieval", "action": "description"}},
    {{"agent": "graph", "action": "description"}},
    {{"agent": "quant", "action": "description"}}
  ],
  "requires_speculation": true/false
}}"""

RETRIEVAL_PROMPT = """You are a data retrieval agent for Wikipedia pageview analytics.
You have access to this pageview database:

{pageviews_data}

Task: Extract and structure the pageview data relevant to this plan:

{plan_json}

Output a clean JSON summary with:
- Yearly data for each article
- Computed ratio (article1 / article2) per year
- Year-over-year trends
- Notable spikes or anomalies

Output ONLY valid JSON."""

GRAPH_PROMPT = """You are an ontology and taxonomy agent for a knowledge system.
You have access to this knowledge graph:

{ontology_data}

Task: Analyze the relationships between entities in this plan:

{plan_json}

Output a JSON analysis with:
- How the entities relate (geographic, political, cultural)
- What contextual factors could drive differential interest
- Any event-driven tags that explain attention spikes

Output ONLY valid JSON."""

SYNTHESIS_PROMPT = """You are a synthesis agent that produces insightful analytical reports.

You have three inputs:

ORIGINAL QUESTION:
{question}

PAGEVIEW DATA:
{retrieval_result}

ONTOLOGY & CONTEXT:
{graph_result}

Produce a markdown report with:
1. A summary table: Year | Article1 views | Article2 views | Ratio | Notes
2. A narrative explaining the trend
3. Factors driving the divergence (use the ontology data)
4. Speculative inference about what is driving recent trends

Be specific with numbers. Use the actual data provided.
Format as clean markdown suitable for a terminal or document."""


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------


def run_agent(
    agent_name: str,
    question: str,
    mock_data: dict[str, Any],
    plan_json: str | None = None,
    retrieval_result: str | None = None,
    graph_result: str | None = None,
    model: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> AgentResult:
    """Run a single agent via claude -p and capture output.

    Args:
        agent_name: One of planner, retrieval, graph, synthesis.
        question: The user's original question.
        mock_data: Dict with 'pageviews' and optionally 'ontology' keys.
        plan_json: Planner output (required for retrieval/graph/synthesis).
        retrieval_result: Retrieval output (required for synthesis).
        graph_result: Graph output (required for synthesis).
        model: Optional model override (e.g. 'haiku', 'sonnet', 'opus').
        timeout: Subprocess timeout in seconds.

    Returns:
        AgentResult with raw output and metadata.
    """
    prompt = _build_prompt(
        agent_name, question, mock_data, plan_json, retrieval_result, graph_result
    )

    cmd = ["claude", "-p", prompt, "--output-format", "text"]
    if model:
        cmd.extend(["--model", model])

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start
        return AgentResult(
            agent=agent_name,
            model=model or "default",
            raw_output=result.stdout,
            exit_code=result.returncode,
            duration_seconds=round(elapsed, 2),
            error=result.stderr if result.returncode != 0 else None,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return AgentResult(
            agent=agent_name,
            model=model or "default",
            raw_output="",
            exit_code=-1,
            duration_seconds=round(elapsed, 2),
            error=f"Timeout after {timeout}s",
        )
    except FileNotFoundError:
        return AgentResult(
            agent=agent_name,
            model=model or "default",
            raw_output="",
            exit_code=-1,
            duration_seconds=0,
            error="claude CLI not found. Is Claude Code installed?",
        )


def run_pipeline(
    question: str,
    mock_data: dict[str, Any],
    models: dict[str, str] | None = None,
    agents: list[str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, AgentResult]:
    """Run the full pipeline (or a subset) sequentially.

    Args:
        question: User question.
        mock_data: Mock data with pageviews and ontology.
        models: Optional per-agent model overrides {agent_name: model}.
        agents: Optional list of agents to run. Default: all four.
        timeout: Per-agent timeout.

    Returns:
        Dict of agent_name -> AgentResult.
    """
    if models is None:
        models = {}
    if agents is None:
        agents = ["planner", "retrieval", "graph", "synthesis"]

    results: dict[str, AgentResult] = {}

    # Planner
    if "planner" in agents:
        results["planner"] = run_agent(
            "planner", question, mock_data,
            model=models.get("planner"), timeout=timeout,
        )

    plan_json = results.get("planner", AgentResult("planner", "", "{}", 0, 0)).raw_output

    # Retrieval
    if "retrieval" in agents:
        results["retrieval"] = run_agent(
            "retrieval", question, mock_data,
            plan_json=plan_json,
            model=models.get("retrieval"), timeout=timeout,
        )

    # Graph
    if "graph" in agents:
        results["graph"] = run_agent(
            "graph", question, mock_data,
            plan_json=plan_json,
            model=models.get("graph"), timeout=timeout,
        )

    # Synthesis
    if "synthesis" in agents:
        retrieval_out = results.get("retrieval", AgentResult("retrieval", "", "{}", 0, 0)).raw_output
        graph_out = results.get("graph", AgentResult("graph", "", "{}", 0, 0)).raw_output
        results["synthesis"] = run_agent(
            "synthesis", question, mock_data,
            plan_json=plan_json,
            retrieval_result=retrieval_out,
            graph_result=graph_out,
            model=models.get("synthesis"), timeout=timeout,
        )

    return results


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_prompt(
    agent_name: str,
    question: str,
    mock_data: dict[str, Any],
    plan_json: str | None,
    retrieval_result: str | None,
    graph_result: str | None,
) -> str:
    pageviews = mock_data.get("pageviews", {})
    ontology = mock_data.get("ontology")

    if agent_name == "planner":
        return PLANNER_PROMPT.format(question=question)

    elif agent_name == "retrieval":
        return RETRIEVAL_PROMPT.format(
            pageviews_data=json.dumps(pageviews, indent=2),
            plan_json=plan_json or "{}",
        )

    elif agent_name == "graph":
        ontology_str = json.dumps(ontology, indent=2) if ontology else "{}"
        return GRAPH_PROMPT.format(
            ontology_data=ontology_str,
            plan_json=plan_json or "{}",
        )

    elif agent_name == "synthesis":
        return SYNTHESIS_PROMPT.format(
            question=question,
            retrieval_result=retrieval_result or "{}",
            graph_result=graph_result or "{}",
        )

    else:
        raise ValueError(f"Unknown agent: {agent_name}")
