#!/usr/bin/env bash
# ==============================================================================
# Wiki Insights — Multi-agent orchestrator prototype
# Pattern: Orchestrator-Worker with parallel fan-out using claude -p
#
# Flow:
#   1. PLANNER    — decomposes natural language question into sub-tasks
#   2. FAN-OUT    — retrieval + graph agents run in parallel
#   3. SYNTHESIS  — merges data + ontology into a final report
#
# Usage:
#   ./orchestrator.sh "How much more hits did Greenland vs Sweden get..."
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/wiki-insights.XXXXXX")
MOCK_DIR="$SCRIPT_DIR/mock_data"

QUESTION="${1:?Usage: $0 \"your question here\"}"

echo "═══ Wiki Insights Orchestrator ═══"
echo "Question: $QUESTION"
echo "Workspace: $WORK_DIR"
echo ""

# ------------------------------------------------------------------------------
# STEP 1: PLANNER — Decompose the question
# ------------------------------------------------------------------------------
echo "▸ Step 1: Decomposing question..."

claude -p "You are a query planner for a Wikipedia analytics system.

Decompose this question into structured sub-tasks. Output ONLY valid JSON.

Question: $QUESTION

Output format:
{
  \"articles\": [\"Article1\", \"Article2\"],
  \"time_range\": {\"start\": \"YYYY\", \"end\": \"YYYY\"},
  \"tasks\": [
    {\"agent\": \"retrieval\", \"action\": \"description\"},
    {\"agent\": \"graph\", \"action\": \"description\"},
    {\"agent\": \"quant\", \"action\": \"description\"}
  ],
  \"requires_speculation\": true/false
}" > "$WORK_DIR/plan.json"

echo "  Plan saved. Articles identified:"
cat "$WORK_DIR/plan.json"
echo ""

# ------------------------------------------------------------------------------
# STEP 2: FAN-OUT — Parallel agent execution
# ------------------------------------------------------------------------------
echo "▸ Step 2: Fan-out — launching retrieval + graph agents in parallel..."

# --- RETRIEVAL AGENT ---
# In production: this wraps your pageviews database via MCP
# For now: reads mock data and interprets it
(
claude -p "You are a data retrieval agent for Wikipedia pageview analytics.
You have access to this pageview database:

$(cat "$MOCK_DIR/pageviews.json")

Task: Extract and structure the pageview data relevant to this plan:

$(cat "$WORK_DIR/plan.json")

Output a clean JSON summary with:
- Yearly data for each article
- Computed ratio (article1 / article2) per year
- Year-over-year trends
- Notable spikes or anomalies

Output ONLY valid JSON." > "$WORK_DIR/retrieval_result.json"

echo "  ✓ Retrieval agent done"
) &
PID_RETRIEVAL=$!

# --- GRAPH / ONTOLOGY AGENT ---
# In production: queries your knowledge graph via MCP
# For now: reads mock ontology and reasons about it
(
claude -p "You are an ontology and taxonomy agent for a knowledge system.
You have access to this knowledge graph:

$(cat "$MOCK_DIR/ontology.json")

Task: Analyze the relationships between entities in this plan:

$(cat "$WORK_DIR/plan.json")

Output a JSON analysis with:
- How the entities relate (geographic, political, cultural)
- What contextual factors could drive differential interest
- Any event-driven tags that explain attention spikes

Output ONLY valid JSON." > "$WORK_DIR/graph_result.json"

echo "  ✓ Graph agent done"
) &
PID_GRAPH=$!

# Wait for both agents
wait $PID_RETRIEVAL $PID_GRAPH
echo ""

# ------------------------------------------------------------------------------
# STEP 3: SYNTHESIS — Merge everything into a report
# ------------------------------------------------------------------------------
echo "▸ Step 3: Synthesizing final report..."

claude -p "You are a synthesis agent that produces insightful analytical reports.

You have three inputs:

ORIGINAL QUESTION:
$QUESTION

PAGEVIEW DATA:
$(cat "$WORK_DIR/retrieval_result.json")

ONTOLOGY & CONTEXT:
$(cat "$WORK_DIR/graph_result.json")

Produce a markdown report with:
1. A summary table: Year | Article1 views | Article2 views | Ratio | Notes
2. A narrative explaining the trend
3. Factors driving the divergence (use the ontology data)
4. Speculative inference about what is driving recent trends

Be specific with numbers. Use the actual data provided.
Format as clean markdown suitable for a terminal or document." > "$WORK_DIR/report.md"

echo ""
echo "═══════════════════════════════════════════"
echo "           FINAL REPORT"
echo "═══════════════════════════════════════════"
echo ""
cat "$WORK_DIR/report.md"

echo ""
echo "═══════════════════════════════════════════"
echo "Artifacts saved in: $WORK_DIR"
echo "  plan.json            — decomposed query"
echo "  retrieval_result.json — pageview data"
echo "  graph_result.json    — ontology analysis"
echo "  report.md            — final report"
echo "═══════════════════════════════════════════"
