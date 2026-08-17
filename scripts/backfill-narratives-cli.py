#!/usr/bin/env python3
"""Backfill degraded spike narratives through the Claude Code CLI, on a subscription.

    ./scripts/backfill-narratives-cli.py                 # dry run: list what it would do
    ./scripts/backfill-narratives-cli.py --execute       # regenerate, cap of 20
    ./scripts/backfill-narratives-cli.py --execute --limit 60

WHY THIS EXISTS, AND WHY IT IS NOT A PIPELINE STAGE.

The daily pipeline talks to the Messages API with a key from Secret Manager. When that
balance runs dry, narrative generation stops. This script is the manual way out: it shells
out to `claude -p`, which authenticates with the operator's own interactive login and
therefore draws on a Claude subscription rather than the API balance. Run it from a host
where `claude` is installed and logged in — warp-host is, the pipeline container is not.

It is deliberately an operator tool rather than stage 8:

  * `claude -p` without --bare loads the whole Claude Code harness — memory, skills,
    plugins, CLAUDE.md. Measured 2026-08-16 on this workload that is 20k-75k input tokens
    of overhead per 35-word answer, reported as $0.08-$0.23 of equivalent cost per
    narrative. The direct API path costs a small fraction of that for the same output.
  * --bare strips the overhead but, per the headless docs, "never reads OAuth credentials"
    and so drops subscription auth entirely. Cheap and subscription are mutually
    exclusive; there is no flag combination that gives both.
  * The container has neither the CLI, nor Node, nor a login. Adding all three to run a
    two-sentence completion would be a large regression in image size and in the
    credential story (the deployment deliberately holds no key material on disk).

So: the API path stays the unattended default because it is cheap and headless. This
exists so a drained balance never blocks a backfill, and so the one-off cost of catching
up lands on a subscription that is already paid for.

The cache keys carry only article and date, so the view count and multiplier the API path
includes are absent here. They were always flavour rather than evidence — the model
attributes the spike from the subject and the date.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.wikipedia.narratives import (  # noqa: E402
    CACHE_FILE, REFUSAL, _wants_search, is_refusal,
)

# Sonnet by default: this is a lookup, not a reasoning task, and the harness overhead
# already dominates. Override for a stubborn batch.
DEFAULT_MODEL = 'claude-sonnet-5'

_PROMPT_SYSTEM = (
    "You explain Wikipedia traffic spikes. Reply with the explanation and nothing else: "
    "at most two sentences, about 35 words, naming the event and why it sent people to "
    "that article. No preamble, no bullet points, and no trailing Sources or citations "
    "section — the text is rendered as a caption in a table cell. "
    f"If you cannot identify a cause with reasonable confidence, reply with exactly: {REFUSAL}"
)


def _clean(text: str) -> str:
    """Strip the citation block Claude Code appends by default.

    The CLI is helpful in a way that breaks a caption: it likes to end with a 'Sources:'
    list of markdown links. Asking it not to helps but is not reliable, so cut it here
    too — the prompt is the request, this is the guarantee.
    """
    for marker in ('\nSources:', '\n\nSources:', '\nSource:', '\n\nSource:'):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def generate(article: str, date_str: str, model: str, timeout: int,
             context: str = '') -> tuple[str, float]:
    """Return (narrative, cost_usd). Raises RuntimeError if the call fails."""
    subject = article.replace('_', ' ')
    ask = f"Article: {subject}. Wikipedia traffic spike date: {date_str}. What caused it?"
    if context:
        ask += ("\n\nOther articles that were also unusual that day:\n" + context +
                "\n\nThese may share a cause or may be unrelated; check rather than assume, "
                "and only mention one if it is genuinely part of the explanation.")
    if _wants_search(date_str):
        ask += (" Search the web for what happened involving this subject on or just "
                "before that date; do not answer from memory alone.")

    cmd = [
        'claude', '-p', ask,
        '--model', model,
        '--system-prompt', _PROMPT_SYSTEM,
        '--allowedTools', 'WebSearch',
        '--output-format', 'json',
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'timed out after {timeout}s') from exc

    if proc.returncode != 0:
        raise RuntimeError(f'claude exited {proc.returncode}: {proc.stderr.strip()[:200]}')

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'unparseable output: {proc.stdout[:200]}') from exc

    if payload.get('is_error'):
        raise RuntimeError(f"claude reported an error: {payload.get('result', '')[:200]}")

    return _clean(payload.get('result', '')), float(payload.get('total_cost_usd') or 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--execute', action='store_true',
                    help='Actually call Claude and write the cache. Default is a dry run.')
    ap.add_argument('--limit', type=int, default=20,
                    help='Maximum narratives to regenerate (default 20).')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--timeout', type=int, default=300, help='Per-call timeout, seconds.')
    ap.add_argument('--cache', type=Path, default=CACHE_FILE)
    ap.add_argument('--context-file', type=Path,
                    help='JSON from export-cospike-context.py: same-day co-spike blurbs. '
                         'Without it each spike is explained in isolation, which is how a '
                         'cause sitting at rank 4 the same morning gets missed.')
    args = ap.parse_args()

    if not args.cache.exists():
        print(f'No cache at {args.cache}', file=sys.stderr)
        return 1

    cache = json.loads(args.cache.read_text())
    context_map: dict[str, str] = {}
    if args.context_file and args.context_file.exists():
        context_map = json.loads(args.context_file.read_text())
        print(f'context: {len(context_map)} block(s) from {args.context_file}')
    elif args.context_file:
        print(f'context file {args.context_file} not found — continuing without it')
    degraded = sorted(k for k, v in cache.items() if is_refusal(v))

    print(f'{args.cache}: {len(cache)} entries, {len(degraded)} degraded')
    if not degraded:
        print('Nothing to do.')
        return 0

    targets = degraded[:args.limit]
    if not args.execute:
        print(f'\nDRY RUN — would regenerate {len(targets)} of {len(degraded)}:')
        for key in targets:
            article, date_str = key.rsplit('::', 1)
            mode = 'search' if _wants_search(date_str) else 'memory'
            ctx = 'ctx' if key in context_map else '   '
            print(f'  [{mode:6}] [{ctx}] {article}  ({date_str})')
        print('\nRe-run with --execute to apply.')
        return 0

    # Confirm the CLI is actually available before promising anything.
    try:
        subprocess.run(['claude', '--version'], capture_output=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f'claude CLI not usable here: {exc}', file=sys.stderr)
        print('Run this from a host with Claude Code installed and logged in.', file=sys.stderr)
        return 1

    fixed = failed = still = 0
    spend = 0.0

    for i, key in enumerate(targets, 1):
        article, date_str = key.rsplit('::', 1)
        try:
            narrative, cost = generate(article, date_str, args.model, args.timeout,
                                       context=context_map.get(key, ''))
        except RuntimeError as exc:
            # Same discipline as the API path: a failed call is not a refusal, so it is
            # not written. The entry stays degraded and is retried next time.
            failed += 1
            print(f'  [{i}/{len(targets)}] {article} ({date_str}): FAILED — {exc}')
            continue

        spend += cost
        if is_refusal(narrative):
            still += 1
            print(f'  [{i}/{len(targets)}] {article} ({date_str}): still unattributed')
            continue

        cache[key] = narrative
        fixed += 1
        short = narrative[:66] + ('…' if len(narrative) > 66 else '')
        print(f'  [{i}/{len(targets)}] {article} ({date_str}): {short}')

        # Write as we go. A long backfill that dies halfway should keep what it earned.
        args.cache.write_text(json.dumps(cache, indent=2, sort_keys=True))

    print(f'\n{fixed} fixed, {still} still unattributed, {failed} failed.')
    print(f'Reported equivalent cost: ${spend:.2f} '
          f'(subscription usage, not an API charge — client-side estimate).')
    remaining = sum(1 for v in cache.values() if is_refusal(v))
    print(f'{remaining} degraded entries remain in the cache.')
    if args.execute:
        print('\nThe cache is durable and published by the pipeline; the next daily run '
              'uploads it to gs://wikipedia-cortex-data/wikipedia/derived/.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
