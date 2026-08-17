#!/usr/bin/env python3
"""Export same-day co-spike context for cached narrative keys, as JSON.

    ./scripts/export-cospike-context.py --out data/cospike_context.json

WHY A SEPARATE STEP. The context needs pandas and the archive, which live in the pipeline
container. The CLI backfill (`backfill-narratives-cli.py`) needs the `claude` binary and an
interactive login, which live on the host. Neither side has the other's dependencies, so
the handoff is a file: run this in the container, then point the backfill at the output.

By default it covers exactly the keys that are still degraded, since those are the ones a
backfill will regenerate — computing context for all 243 would load years nobody needs.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.wikipedia.analysis import load_year_data, filter_content  # noqa: E402
from shared.wikipedia import baseline  # noqa: E402
from shared.wikipedia.narratives import CACHE_FILE, is_refusal  # noqa: E402

DATA_DIR = ROOT / 'data'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=DATA_DIR / 'cospike_context.json')
    ap.add_argument('--cache', type=Path, default=CACHE_FILE)
    ap.add_argument('--all-keys', action='store_true',
                    help='Cover every cached key, not just the degraded ones.')
    ap.add_argument('--top-n', type=int, default=6)
    args = ap.parse_args()

    cache = json.loads(args.cache.read_text())
    keys = sorted(cache if args.all_keys else [k for k, v in cache.items() if is_refusal(v)])
    if not keys:
        print('No keys to cover.')
        return 0

    # One key is "article::YYYY-MM-DD"; load only the years those dates fall in.
    targets = [(k.rsplit('::', 1)[0], k.rsplit('::', 1)[1]) for k in keys]
    years = sorted({int(d[:4]) for _, d in targets})
    print(f'{len(keys)} key(s) across years {years}')

    frames = []
    for year in years:
        try:
            frames.append(load_year_data(DATA_DIR, year))
        except FileNotFoundError:
            print(f'  no data for {year}, skipping')
    if not frames:
        print('No archive data for the requested years.', file=sys.stderr)
        return 1

    df = filter_content(pd.concat(frames, ignore_index=True))
    # seasonal only pays for itself with more than one year in hand; day-of-year factors
    # need the same calendar position observed in at least two.
    enriched = baseline.add_expectation(df, seasonal=len(years) > 1)

    spikes = pd.DataFrame({
        'article': [a for a, _ in targets],
        'spike_date': pd.to_datetime([d for _, d in targets]),
    })
    context = baseline.co_spike_context(enriched, spikes, top_n=args.top_n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(context, indent=2, sort_keys=True))
    print(f'Wrote {len(context)} context block(s) to {args.out}')
    missing = [k for k in keys if k not in context]
    if missing:
        # Not an error: a spike day where nothing else was unusual has no context to give.
        print(f'  {len(missing)} key(s) had no co-spiking companions that day')
    return 0


if __name__ == '__main__':
    sys.exit(main())
