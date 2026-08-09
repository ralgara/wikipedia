# Wiki Insights — Agent Orchestration Context

## Goal

Build a multi-agent system that answers quantitative, ontology-aware questions about Wikipedia article performance. Example query:

> "How much more hits did article 'Greenland' vs 'Sweden' get in wikipedia last year and how has that relationship changed since 2007? What factors could be driving this trend?"

The system should normalize and decompose natural language questions into executable elements: hit count lookups by article, taxonomy/ontology tagging for relationships between subjects, quantitative aggregation with time awareness, and speculative inference. The output is a formatted markdown report with data tables, trend narratives, and contextual analysis.

## Project Intent

This project is primarily **didactic**. The Wikipedia analytics use case is a vehicle for learning multi-agent orchestration — not a production system. The goal is to introduce concepts and complexity gradually: start with a shell script anyone can read, then layer in MCP, then the Agent SDK, then distribution patterns. Each tier should be fully understandable before moving to the next. We favor clarity over optimization and explicit plumbing over magical abstractions.

## Design Rationale

### Why Multi-Agent?

A single monolithic prompt can answer this question (and does — Claude already produces good reports end-to-end). But we want to decouple concerns for three reasons:

1. **Data sovereignty** — the retrieval layer should wrap a real pageviews database we control, not rely on the LLM's training data or web search. Same for the ontology graph.
2. **Composability** — the retrieval agent, graph agent, and synthesis agent are reusable across different question types. A question about "Greenland vs Sweden" and "Bitcoin vs Ethereum" should use the same agents with different data.
3. **Verifiability** — intermediate artifacts (the plan, the raw data, the ontology analysis) are inspectable. We can debug each agent independently.

### Four Orchestration Patterns Evaluated

We considered four archetypal patterns for multi-agent coordination:

1. **Orchestrator-Worker (hub-and-spoke)** — one conductor decomposes tasks and delegates to specialist workers. The conductor owns the plan, collects results, synthesizes. Most common, easiest to reason about.
2. **Pipeline (assembly line)** — agents chained sequentially, output of one feeds the next. Clean and debuggable but too rigid for our case — Wikipedia lookup and ontology tagging should run in parallel, not sequentially.
3. **Blackboard (shared state)** — classic AI pattern from the 80s. All agents read/write to a shared workspace. Very flexible but coordination gets complex. We borrow elements of this for intermediate file storage.
4. **Event-driven / Pub-Sub** — agents emit and subscribe to events. Maximally decoupled. Great for scaling, overkill for a local prototype.

**Decision:** Orchestrator-Worker with parallel fan-out, borrowing from the blackboard pattern for intermediate state (temp directory with JSON artifacts).

### Hybrid LLM / Deterministic Design

A key design question: should the orchestrator (planner) be an LLM or deterministic code?

If question decomposition follows predictable patterns (detect entities → look up stats → compare → infer), a Python function can do the planning and just call LLM agents for the fuzzy parts. This is cheaper, faster, and more reliable.

If the decomposition itself requires judgment (understanding what "relationship" means, deciding which data sources are relevant), then the planner should be an LLM agent.

**Decision for prototype:** LLM planner (simplest to build). Target: migrate to deterministic planner for structured parts, LLM only for ontology reasoning and speculative inference.

## Concurrency Model — From CLI Constraints to True Parallelism

### The Problem

The interactive `claude` CLI is single-threaded by design — one conversational turn at a time. When a task is running, the prompt accepts input but queues it. This is because the interactive session is optimized for human conversation, not machine orchestration.

### Concurrency Lives at Different Levels

**Level 0: Inside a single agent turn** — Claude can launch multiple tool calls in parallel within one response. The parallelism is real but invisible at the prompt. You submit one message, Claude fans out internally, you get one response. This is how subagents (the Task tool) work inside Claude Code.

**Level 1: `claude -p` (non-interactive mode)** — The key unlock for scripting. `claude -p "prompt"` runs a single turn, prints to stdout, and exits. No interactive session. Multiple instances run truly concurrently via `&` + `wait`. Each can have its own `--system-prompt`, `--mcp-config`, and tools.

**Level 2: Agent SDK (programmatic)** — Same as Level 1 but with proper control flow. Python or TypeScript script instantiates multiple agents, runs them with `asyncio.gather` or `Promise.all`, collects structured results. The SDK handles API calls, retries, and context management.

**Level 3: Multiple interactive sessions + shared state** — N terminals, each running `claude` with a different system prompt via `--system-prompt-file`. Coordination through a shared directory (blackboard) or MCP servers acting as message buses. Most visible for demos and debugging.

**The concurrency boundary is the process, not the session.** Each `claude` invocation is an independent agent.

## Incremental Development Path (Tiers)

The prototype is designed to evolve through tiers of increasing sophistication. Each tier is self-contained and runnable.

### Tier 1: Shell Script + `claude -p` (CURRENT)

The simplest viable multi-agent system. A bash script orchestrates 4 `claude -p` calls with shell-level parallelism. Mock data piped via `$(cat ...)`. Results collected via stdout redirection. Artifacts saved to temp directory.

**Pros:** Zero dependencies, transparent, easy to debug, runnable anywhere with Claude Code installed.
**Cons:** No structured I/O, no error handling, prompts inline in bash, no retry logic, shell quoting fragility.

### Tier 2: MCP Servers Replace Mocks

Each data source becomes an MCP server. The pageviews database exposes a `lookup_pageviews(article, start_year, end_year)` tool. The knowledge graph exposes `get_relationships(entity1, entity2)` and `get_taxonomy(entity)` tools. Each `claude -p` agent connects to its MCP server via `--mcp-config`. The `$(cat mock.json)` calls become real tool invocations.

**What changes:** Mock JSON files → running MCP servers. Each agent's prompt drops the inline data and instead uses tool calls. The orchestrator script structure stays the same.

### Tier 3: Agent SDK Migration

Move from bash to Python/TypeScript. The orchestrator becomes a proper program with structured I/O (JSON in/out instead of stdout parsing), error handling and retries, typed agent configurations, `asyncio.gather` for fan-out, and logging/observability.

**What changes:** `orchestrator.sh` → `orchestrator.py`. The conceptual flow is identical. The agents are still `claude -p` under the hood (or direct API calls via the SDK).

### Tier 4: Skills, A2A, and Distribution

Package reusable capabilities (ontology tagger, quant reasoner) as Skills — bundles of prompts + tools + best practices. Introduce A2A (Agent-to-Agent protocol) for inter-agent discovery and communication if scaling to multi-vendor or distributed deployments. This tier is about ecosystem integration, not local orchestration.

## Architecture

```
User Question
     │
  Planner Agent (decompose + normalize)
     │
     ├── Retrieval Agent (pageviews DB — mocked, will wrap real DB via MCP)
     ├── Graph Agent (ontology/taxonomy — mocked, will wrap real graph via MCP)
     │
  Synthesis Agent (merge data + ontology → markdown report)
     │
  Final Report
```

The Planner and Synthesis could be the same agent (common "bookend" pattern — the orchestrator opens and closes the workflow). Currently separate for clarity.

## Current Prototype

### File Structure
```
wiki-insights/
├── orchestrator.sh           # Main script — 4 claude -p calls
├── mock_data/
│   ├── pageviews.json        # Mock pageview database
│   └── ontology.json         # Mock knowledge graph
└── CONTEXT.md                # This file
```

### orchestrator.sh Flow
1. **Planner** — `claude -p` decomposes question into structured JSON (articles, time range, sub-tasks)
2. **Fan-out** — two parallel `claude -p` calls:
   - Retrieval agent: reads pageviews mock, computes ratios and trends
   - Graph agent: reads ontology mock, analyzes entity relationships and event tags
3. **Synthesis** — `claude -p` merges all results into a markdown report

Intermediate artifacts saved to a temp dir for inspection.

### Running
```bash
./orchestrator.sh "How much more hits did Greenland vs Sweden get in wikipedia last year?"
```

## Component Roadmap

| Component | Current (Tier 1) | Tier 2 | Tier 3+ |
|-----------|------------------|--------|---------|
| Pageviews DB | `mock_data/pageviews.json` | MCP server wrapping real DB | Same, with caching |
| Knowledge Graph | `mock_data/ontology.json` | MCP server wrapping real graph | Same, with inference |
| Planner | `claude -p` with inline prompt | Same | Deterministic Python for structured parts |
| Retrieval Agent | `claude -p` + `$(cat mock)` | `claude -p` + `--mcp-config` | SDK agent with typed tools |
| Graph Agent | `claude -p` + `$(cat mock)` | `claude -p` + `--mcp-config` | SDK agent with typed tools |
| Synthesis Agent | `claude -p` with inline prompt | Same, with few-shot examples | SDK agent with report templates |
| Orchestrator | Bash script | Bash script (same) | Python/TS with Agent SDK |
| Coordination | Filesystem (temp dir) | Filesystem + MCP | Structured SDK messaging |

## Design Decisions Still Open

- **Planner: LLM vs deterministic** — Prototype uses LLM. Target: deterministic for structured decomposition, LLM only for ambiguous queries.
- **Bookend pattern** — Should planner and synthesis be the same agent? Saves context passing overhead but makes the agent's role less focused.
- **A2A protocol** — Not needed for local prototype. Relevant when scaling to distributed or multi-vendor agents.
- **Skills packaging** — The ontology tagger and quant reasoner could be packaged as reusable Skills. Depends on how many different orchestration flows we build.
- **Report quality** — The synthesis agent may need few-shot examples or a report template to consistently match the desired output format (the Greenland/Sweden table example).
- **Error handling** — What happens when a fan-out agent fails? Retry? Degrade gracefully? Not addressed in Tier 1.
