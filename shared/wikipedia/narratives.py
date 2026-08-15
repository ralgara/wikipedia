"""LLM-generated spike narratives with local JSON cache.

Requires: anthropic  (pip install anthropic)
Cache:    data/narratives_cache.json  (keyed "article::YYYY-MM-DD")

The narrator has the web_search tool. That is the whole point of it: a spike is a
real-world event, and the events worth explaining are usually more recent than any
model's training data. Before search was wired in, every 2026 spike and 90% of 2025
came back as some variant of "I have training data only through April 2024" — which
the report then published verbatim on a public page.

Cost note: search runs only for spikes that are not already cached, and the cache is
durable (published to the bucket under wikipedia/derived/). A steady-state daily run
looks up a handful of new spikes, not the whole archive.
"""

import json
from pathlib import Path

# Resolve cache path relative to this file:
# shared/wikipedia/narratives.py -> shared/wikipedia -> shared -> repo_root -> data/
_REPO_ROOT = Path(__file__).parent.parent.parent
CACHE_FILE = _REPO_ROOT / 'data' / 'narratives_cache.json'

# web_search_20260209 requires Opus 4.6+ / Sonnet 4.6+; it is not available on Haiku,
# which is what this used to run on. The model and the tool are one decision, not two.
MODEL = 'claude-opus-5'
_SEARCH_TOOL = {'type': 'web_search_20260209', 'name': 'web_search', 'max_uses': 4}

# The exact string the model is told to emit when it genuinely cannot attribute a
# spike. Kept as a constant so callers can recognise it rather than pattern-matching
# whatever prose the model improvised — the old failure mode.
REFUSAL = 'Not enough data for narrative.'

# Server-side tool loops can stop with stop_reason="pause_turn" when they hit the
# server's iteration limit. Re-sending resumes; this bounds how many times we will.
_MAX_CONTINUATIONS = 3

# After this many consecutive infrastructure failures, stop calling for the rest of the
# batch. The failure that motivated it — an exhausted credit balance — fails every call
# identically, so continuing just burns wall-clock to produce the same error N more times.
_MAX_CONSECUTIVE_FAILURES = 3


class NarrativeUnavailable(Exception):
    """The call failed; the model did not decline.

    These are different outcomes and conflating them corrupts the cache. A refusal is a
    durable fact about a spike ("nothing findable explains this") and is worth persisting.
    A 429, a 500, an expired key or an exhausted credit balance says nothing about the
    spike at all — persisting it writes a permanent wrong answer that no later run will
    retry, because the key is now present in the cache.

    This is not hypothetical: it happened on 2026-08-14. The credit balance ran out midway
    through a refresh, every subsequent call raised, and the original `except Exception:
    return REFUSAL` wrote 16 refusals that looked exactly like genuine ones.
    """

_SYSTEM = (
    "You analyze Wikipedia traffic data. "
    "Given an article name and the date of an unusual traffic spike, write 1–2 sentences "
    "explaining the most likely real-world cause (a news event, death, film/game release, "
    "sports result, anniversary, etc.).\n\n"
    "Search the web for what happened involving that subject on or just before the spike "
    "date. The spike dates you are given are frequently more recent than your training "
    "data, so do not answer from memory alone — a spike you cannot place is almost always "
    "one you have not looked up yet. Prefer contemporaneous reporting from around the "
    "spike date over later retrospectives.\n\n"
    "Write for a reader looking at a traffic chart: name the event and why it drove "
    "people to that article. No preamble, no hedging about your sources, no meta-commentary "
    "about searching.\n\n"
    "Length is a hard constraint, not a guideline: at most two sentences and about 35 "
    "words. This is rendered as a caption inside a table cell, so a paragraph breaks the "
    "layout. Name the event, the date if it differs from the spike, and the connection — "
    "drop supporting detail (venue, ratings, guest appearances, runtimes, box office) "
    "unless it is the actual reason for the spike.\n\n"
    f"If searching does not turn up a plausible cause, respond with exactly: {REFUSAL}"
)


def is_refusal(narrative: str) -> bool:
    """True if a narrative is a non-answer rather than an explanation.

    Covers the sanctioned REFUSAL token and the improvised prose hedges the old
    Haiku-without-search path produced ("I have training data only through...",
    "I cannot determine..."). Used to select cache entries worth regenerating;
    callers rendering a report can also use it to omit the box entirely.
    """
    if not narrative:
        return True
    low = narrative.strip().lower()
    if low.startswith(REFUSAL.lower().rstrip('.')):
        return True
    return any(marker in low for marker in (
        'not enough data',
        'narrative unavailable',
        'training data',
        'knowledge cutoff',
        'training cutoff',
        'i cannot identify',
        'i cannot determine',
        'i cannot reliably',
        "i don't have information",
        'i do not have information',
    ))


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def _text_of(resp) -> str:
    """Concatenate the text blocks of a response.

    Not resp.content[0].text: with a server-side tool in play the first block is
    routinely a thinking or server_tool_use block, and indexing position 0 either
    raises or returns the wrong thing.
    """
    return ''.join(
        block.text for block in resp.content if getattr(block, 'type', None) == 'text'
    ).strip()


def _call_api(client, article: str, date_str: str, multiplier: float, views: int,
              model: str = MODEL, use_search: bool = True) -> str:
    msg = (
        f"Article: {article.replace('_', ' ')}\n"
        f"Spike date: {date_str}\n"
        f"Views on spike day: {views:,}\n"
        f"Multiplier above average: {multiplier:.1f}x\n\n"
        "What caused this Wikipedia traffic spike?"
    )
    messages = [{'role': 'user', 'content': msg}]

    kwargs = {
        'model': model,
        # Headroom, not a target. Thinking is on by default on Opus 5 and max_tokens
        # caps thinking + visible text together, so the old 120 would truncate before
        # the model said anything. The output itself is two sentences.
        'max_tokens': 4096,
        'system': _SYSTEM,
        # A short factual lookup. Low effort is both cheaper and better calibrated
        # here than the default; it also keeps thinking from crowding the budget.
        'output_config': {'effort': 'low'},
    }
    if use_search:
        kwargs['tools'] = [_SEARCH_TOOL]

    try:
        for _ in range(_MAX_CONTINUATIONS + 1):
            resp = client.messages.create(messages=messages, **kwargs)

            if resp.stop_reason == 'refusal':
                return REFUSAL

            # The server-side search loop hit its iteration cap. Re-send with the
            # assistant turn appended and it picks up where it left off; do NOT add
            # a "continue" user message, the API resumes on its own.
            if resp.stop_reason == 'pause_turn':
                messages = messages[:1] + [{'role': 'assistant', 'content': resp.content}]
                continue

            text = _text_of(resp)
            return text or REFUSAL

        # Still paused after _MAX_CONTINUATIONS — treat as unattributable rather
        # than looping the daily pipeline.
        return REFUSAL
    except Exception as exc:
        # Deliberately NOT returning REFUSAL: see NarrativeUnavailable. The caller
        # declines to cache this, so the spike is retried on the next run.
        raise NarrativeUnavailable(f'{type(exc).__name__}: {exc}') from exc


def batch_generate(spikes, top_n: int = 20, verbose: bool = True,
                   model: str = MODEL, use_search: bool = True,
                   refresh_degraded: bool = False) -> dict:
    """Generate narratives for up to top_n spikes.

    Args:
        spikes: iterable of dicts with keys: article, spike_date, spike_views, multiplier.
                spike_date may be a datetime or ISO string.
        top_n:  max number of spikes to process.
        verbose: print progress.
        model:  override the narrator model.
        use_search: give the narrator the web_search tool. Off answers from training
                data alone, which is what produced the refusals this replaced.
        refresh_degraded: re-fetch cached entries that are refusals rather than
                explanations. Off by default — the cache is expensive and
                non-reproducible, so overwriting it is always an explicit act.

    Returns:
        dict keyed by "article::YYYY-MM-DD" → narrative string.
    """
    cache = _load_cache()
    results = {}
    to_fetch = []

    for spike in list(spikes)[:top_n]:
        sd = spike['spike_date']
        date_str = sd.strftime('%Y-%m-%d') if hasattr(sd, 'strftime') else str(sd)[:10]
        key = f"{spike['article']}::{date_str}"
        cached = cache.get(key)
        if cached is not None and not (refresh_degraded and is_refusal(cached)):
            results[key] = cached
        else:
            to_fetch.append((key, spike['article'], date_str,
                             float(spike['multiplier']), int(spike['spike_views'])))

    if not to_fetch:
        return results

    if verbose:
        how = 'with web search' if use_search else 'from training knowledge only'
        print(f"  Fetching {len(to_fetch)} spike narrative(s) via {model} ({how})...")

    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        for key, *_ in to_fetch:
            results[key] = "Narrative unavailable (pip install anthropic)"
        return results

    consecutive_failures = 0
    failed = 0

    for key, article, date_str, multiplier, views in to_fetch:
        if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            failed += 1
            continue
        try:
            narrative = _call_api(client, article, date_str, multiplier, views,
                                  model=model, use_search=use_search)
        except NarrativeUnavailable as exc:
            # Not cached, so the next run retries this spike rather than inheriting
            # a refusal that was really an outage.
            consecutive_failures += 1
            failed += 1
            if verbose:
                print(f"    {article} ({date_str}): CALL FAILED — {exc}")
            continue

        consecutive_failures = 0
        cache[key] = narrative
        results[key] = narrative
        if verbose:
            short = narrative[:70] + ('…' if len(narrative) > 70 else '')
            print(f"    {article} ({date_str}): {short}")

    if failed:
        # Loud, because the quiet version of this is a report that silently lost its
        # narratives and a caller who thinks the spikes were simply unattributable.
        print(f"  WARNING: {failed} of {len(to_fetch)} narrative call(s) failed and were "
              f"NOT cached; they will be retried on the next run.")
        if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            print(f"  Stopped calling after {_MAX_CONSECUTIVE_FAILURES} consecutive "
                  f"failures — check credentials, credit balance, and rate limits.")

    _save_cache(cache)
    return results
