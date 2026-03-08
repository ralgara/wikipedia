"""LLM-generated spike narratives with local JSON cache.

Requires: anthropic  (pip install anthropic)
Cache:    data/narratives_cache.json  (keyed "article::YYYY-MM-DD")
"""

import json
from pathlib import Path

# Resolve cache path relative to this file:
# shared/wikipedia/narratives.py -> shared/wikipedia -> shared -> repo_root -> data/
_REPO_ROOT = Path(__file__).parent.parent.parent
CACHE_FILE = _REPO_ROOT / 'data' / 'narratives_cache.json'

_SYSTEM = (
    "You analyze Wikipedia traffic data. "
    "Given an article name and the date of an unusual traffic spike, write 1–2 sentences "
    "explaining the most likely real-world cause (a news event, death, film/game release, "
    "sports result, anniversary, etc.). "
    "Draw only on your training knowledge — do not speculate beyond what you know. "
    "If you cannot identify a plausible cause with reasonable confidence, "
    "respond with exactly: Not enough data for narrative."
)


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def _call_api(client, article: str, date_str: str, multiplier: float, views: int) -> str:
    msg = (
        f"Article: {article.replace('_', ' ')}\n"
        f"Spike date: {date_str}\n"
        f"Views on spike day: {views:,}\n"
        f"Multiplier above average: {multiplier:.1f}x\n\n"
        "What caused this Wikipedia traffic spike?"
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            system=_SYSTEM,
            messages=[{"role": "user", "content": msg}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return "Not enough data for narrative"


def batch_generate(spikes, top_n: int = 20, verbose: bool = True) -> dict:
    """Generate narratives for up to top_n spikes.

    Args:
        spikes: iterable of dicts with keys: article, spike_date, spike_views, multiplier.
                spike_date may be a datetime or ISO string.
        top_n:  max number of spikes to process.
        verbose: print progress.

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
        if key in cache:
            results[key] = cache[key]
        else:
            to_fetch.append((key, spike['article'], date_str,
                             float(spike['multiplier']), int(spike['spike_views'])))

    if not to_fetch:
        return results

    if verbose:
        print(f"  Fetching {len(to_fetch)} spike narrative(s) from Claude Haiku...")

    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        for key, *_ in to_fetch:
            results[key] = "Narrative unavailable (pip install anthropic)"
        return results

    for key, article, date_str, multiplier, views in to_fetch:
        narrative = _call_api(client, article, date_str, multiplier, views)
        cache[key] = narrative
        results[key] = narrative
        if verbose:
            short = narrative[:70] + ('…' if len(narrative) > 70 else '')
            print(f"    {article} ({date_str}): {short}")

    _save_cache(cache)
    return results
