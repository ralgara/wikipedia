#!/usr/bin/env python3
"""Co-spike correlation kernel: which articles spike together, and more than chance.

Builds an article x article co-occurrence graph from spike days within +/- k days, then
scores each edge by lift over what independence would predict. Thematic clusters fall out
with no taxonomy and no Wikidata — the roadmap's v0.3 "co-occurrence kernel".

    uv run python scripts/analyze-correlations.py
    uv run python scripts/analyze-correlations.py --score flat --window 3
    uv run python scripts/analyze-correlations.py --years 2024 2025 2026

TWO LENSES, AND YOU WANT BOTH

  --score residual (default)  Spikes measured against the expectation baseline. Surfaces
                              genuine news cascades: a death and the works it prompted, an
                              election and its winner, a film and its cast.
  --score flat                Spikes measured against a flat per-article mean, which is
                              what `analysis.detect_spikes` has always done. This is the
                              lens that shows PERIODIC structure — Independence Day,
                              Fireworks and Fourth of July co-spiking every July is a real
                              and stable cluster, and it is invisible under residual
                              scoring precisely because the model expects it.

Residual answers "what moved together unexpectedly". Flat answers "what moves together,
including on schedule". Neither subsumes the other, so the flag exists rather than a
default nobody can override.

Caveat on that claim, since it was written before it was tested: the residual lens still
surfaces a clear Christmas cluster (Die Hard, Elf, Home Alone 2, Bing Crosby, Donna Reed,
Charles Dickens). Either those articles fail the day-of-year fitting gates, or their
December spikes exceed even a seasonally-adjusted expectation. So residual suppresses
periodic structure less completely than the paragraph above implies, and the flat lens has
not yet been run for comparison.

WHY LIFT AND NOT RAW COUNTS

Raw co-occurrence rewards articles that spike constantly: a hyperactive article pairs with
everything, and the top of the list becomes a popularity ranking. Lift divides observed
co-occurrence by what independence predicts, so a pair that spikes together three times out
of three scores far above a pair that spikes together ten times out of four hundred.

    expected = n_a * n_b * (2*window + 1) / n_days
    lift     = observed / expected

MIN_SUPPORT then keeps a single coincidence from topping the chart on a divide-by-tiny.

Output:
  data/correlations/edges.json    — scored edges, and the clusters they induce
  reports/correlations.html       — visual report
"""

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from math import log2
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.wikipedia.analysis import (  # noqa: E402
    load_all_data, load_year_data, filter_content, detect_spikes,
)
from shared.wikipedia import baseline  # noqa: E402
from shared.wikipedia.report_utils import (  # noqa: E402
    COLORS, format_number, wiki_link, make_badge, html_page,
)

DATA_DIR = ROOT / 'data'
REPORTS_DIR = ROOT / 'reports'

# An edge needs this many co-spike days before it is reported. Two articles that spiked
# together once are a coincidence with an enormous lift; the floor is what stops the
# ranking being entirely made of them.
MIN_SUPPORT = 2

# Cap on the size of a day's NEIGHBOURHOOD (the day plus its window), above which the day
# is skipped as site-wide rather than thematic. Set to 60 on the first attempt, which threw
# away 412 of 590 days — with residual scoring a typical neighbourhood is already ~114
# articles, so the cap was rejecting the ordinary case and keeping almost nothing. It only
# needs to catch genuine outliers: an outage, a data artefact, a day where the whole top
# 1000 shifts at once.
MAX_SPIKES_PER_DAY = 250


def _episodes(days, window: int) -> int:
    """Collapse a set of days into distinct episodes, merging any within `window`.

    This is load-bearing, and getting it wrong produced the first run's nonsense. With a
    +/-1 day window, ONE co-spike is visible from three anchor days, so counting anchor
    days credited a single coincidence with support=3. Worse, per-article spike counts were
    counted in raw days while support was counted in anchor days, so the two sides of the
    lift ratio used different units — every one-off pair came out at exactly 590x (the day
    count), which is what a lift of "observed 3, expected 3/590" means.

    Counting episodes on both sides puts them back in the same unit: an event, not a day
    that happened to be near one.
    """
    days = sorted(days)
    if not days:
        return 0
    n = 1
    for prev, cur in zip(days, days[1:]):
        if (cur - prev).days > window:
            n += 1
    return n


def spike_days(df: pd.DataFrame, score: str, threshold: float) -> pd.DataFrame:
    """Every article-day above threshold. Columns: article, date."""
    if score == 'flat':
        # detect_spikes collapses to one row per article, which is the wrong shape here —
        # a kernel needs every spike day, not each article's single biggest. Recompute the
        # z-score directly with the same flat definition.
        stats = df.groupby('article')['views'].agg(['mean', 'std']).reset_index()
        stats = stats[stats['std'] > 0]
        merged = df.merge(stats, on='article')
        merged['z'] = (merged['views'] - merged['mean']) / merged['std']
        hits = merged[(merged['z'] > threshold) &
                      (merged['views'] >= baseline.MIN_VIEWS_FOR_SPIKE)]
        return hits[['article', 'date']]

    enriched = baseline.add_expectation(df, seasonal=True)
    hits = enriched[(enriched['residual'] > threshold) &
                    (enriched['views'] >= baseline.MIN_VIEWS_FOR_SPIKE)]
    return hits[['article', 'date']]


def build_edges(spikes: pd.DataFrame, window: int) -> tuple[list[dict], dict]:
    """Score co-spike pairs by lift. Returns (edges, diagnostics).

    Support and expectation are both counted in DAYS, with the window applied as a
    dilation of one side:

        support  = |days(a) INTERSECT dilate(days(b), w)|
        expected = |days(a)| * |dilate(days(b), w)| / n_days
        lift     = support / expected

    Two earlier formulations were wrong in ways worth keeping written down.

    Counting ANCHOR days inflated support: with w=1 a single co-spike is visible from three
    anchor days, so one coincidence scored support=3 against per-article counts measured in
    raw days. Mixed units, and every one-off pair landed on exactly n_days/span.

    Collapsing to EPISODES fixed the units and broke the semantics. Google Chrome carries a
    residual above 3 for 103 CONSECUTIVE days in 2025 — a level shift, not a spike, because
    its traffic stepped up and the median baseline sits below the new plateau. Episode
    counting called that one rare event, so co-occurring with it looked miraculous and it
    pulled unrelated articles into every cluster. A 103-day plateau should be easy to
    coincide with, not hard.

    Days-with-dilation gets both right: the plateau earns a large expectation and stops
    dominating, and the window no longer multiplies the observation.
    """
    by_article: dict[str, set] = defaultdict(set)
    for article, date in zip(spikes['article'], spikes['date']):
        by_article[article].add(date.normalize())

    by_day: dict[pd.Timestamp, list[str]] = defaultdict(list)
    for article, days in by_article.items():
        for d in days:
            by_day[d].append(article)

    all_days = sorted(by_day)
    n_days = len(all_days)

    one = pd.Timedelta(days=1)
    dilated = {
        a: {d + k * one for d in days for k in range(-window, window + 1)}
        for a, days in by_article.items()
    }

    # Candidate generation: only pairs that share at least one day are worth scoring, which
    # keeps this far below the O(N^2) all-pairs cost.
    dropped_days = 0
    candidates: set[tuple[str, str]] = set()
    for i, day in enumerate(all_days):
        neighbourhood: list[str] = []
        for j in range(max(0, i - window), min(n_days, i + window + 1)):
            if abs((all_days[j] - day).days) <= window:
                neighbourhood.extend(by_day[all_days[j]])
        neighbourhood = sorted(set(neighbourhood))
        if len(neighbourhood) > MAX_SPIKES_PER_DAY:
            dropped_days += 1
            continue
        candidates.update(combinations(neighbourhood, 2))

    edges = []
    for a, b in candidates:
        days_a = by_article[a]
        dil_b = dilated[b]
        support = len(days_a & dil_b)
        if support < MIN_SUPPORT:
            continue
        expected = len(days_a) * len(dil_b) / n_days
        if expected <= 0:
            continue
        lift = support / expected
        edges.append({
            'a': a,
            'b': b,
            'support': support,
            'days_a': len(days_a),
            'days_b': len(by_article[b]),
            'expected': round(expected, 3),
            'lift': round(lift, 2),
            'days': sorted(d.strftime('%Y-%m-%d') for d in (days_a & dil_b))[:5],
            # Lift alone has a ceiling: two rare articles overlapping perfectly hit the
            # maximum achievable ratio, so dozens of unrelated pairs tie at exactly
            # n_days/span and the table stops discriminating. Weighting by log support
            # breaks those ties toward the pairs with more evidence behind them. A
            # heuristic, deliberately — not a p-value, and not presented as one.
            'score': round(lift * log2(1 + support), 2),
        })

    edges.sort(key=lambda e: (e['score'], e['support']), reverse=True)
    diagnostics = {
        'n_days': n_days,
        'n_spikes': int(len(spikes)),
        'n_articles': int(spikes['article'].nunique()),
        'n_pairs_considered': len(candidates),
        'n_edges': len(edges),
        'days_dropped_as_sitewide': dropped_days,
    }
    return edges, diagnostics


def cluster(edges: list[dict], top_n: int, min_lift: float) -> list[list[str]]:
    """Connected components over the strongest edges — cheap, and enough to read.

    Deliberately not a community-detection algorithm. With a lift floor the graph is
    already sparse, and components answer the question being asked ("what groups with
    what") without a tuning parameter nobody would know how to set.
    """
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # Single linkage chains through hubs: one article that co-spikes with everything
    # merges every group it touches, which is how the first run produced a 95-member
    # 'cluster' of unrelated topics. Requiring meaningful support as well as lift keeps
    # incidental edges from acting as bridges.
    for e in edges[:top_n]:
        if e['lift'] >= min_lift and e['support'] >= MIN_SUPPORT + 1:
            union(e['a'], e['b'])

    groups: dict[str, list[str]] = defaultdict(list)
    for node in parent:
        groups[find(node)].append(node)
    out = [sorted(v) for v in groups.values() if len(v) > 1]
    out.sort(key=len, reverse=True)
    return out


def build_html(edges, clusters, diag, score, window) -> str:
    rows = ''
    for e in edges[:40]:
        rows += f'''
        <tr>
          <td>{wiki_link(e['a'])}</td>
          <td>{wiki_link(e['b'])}</td>
          <td>{make_badge(f"{e['lift']:.0f}x", COLORS['highlight'])}</td>
          <td>{e['support']}</td>
          <td style="color:{COLORS['muted']};font-size:.82rem">{', '.join(e['days'][:3])}</td>
        </tr>'''

    cluster_html = ''
    for i, group in enumerate(clusters[:15], 1):
        members = ' · '.join(wiki_link(a) for a in group[:12])
        more = f' <span style="color:{COLORS["muted"]}">+{len(group)-12} more</span>' if len(group) > 12 else ''
        cluster_html += (f'<div style="margin-bottom:.75rem">{make_badge(i, COLORS["accent"])} '
                         f'{members}{more}</div>')
    if not cluster_html:
        cluster_html = f'<p style="color:{COLORS["muted"]}">No clusters above the lift floor.</p>'

    lens = ('measured against the expectation baseline — unexpected co-movement'
            if score == 'residual' else
            'measured against a flat per-article mean — includes periodic co-movement')

    body = f'''
    <section class="section">
      <h2>Overview</h2>
      <p style="color:{COLORS['muted']}">Spikes {lens}. Co-occurrence counted within
      &plusmn;{window} days; edges scored by lift over independence.</p>
      <div class="metrics">
        <div class="metric"><div class="metric-value">{format_number(diag['n_spikes'])}</div><div class="metric-label">Spike Days</div></div>
        <div class="metric"><div class="metric-value">{format_number(diag['n_articles'])}</div><div class="metric-label">Spiking Articles</div></div>
        <div class="metric"><div class="metric-value">{format_number(diag['n_edges'])}</div><div class="metric-label">Edges (support &ge; {MIN_SUPPORT})</div></div>
        <div class="metric"><div class="metric-value">{len(clusters)}</div><div class="metric-label">Clusters</div></div>
      </div>
      <p style="color:{COLORS['muted']};font-size:.85rem">{diag['days_dropped_as_sitewide']} day(s)
      skipped as site-wide (more than {MAX_SPIKES_PER_DAY} articles spiking at once).</p>
    </section>

    <section class="section highlight">
      <h2>Clusters</h2>
      <p style="color:{COLORS['muted']};font-size:.88rem">Connected components over the
      strongest edges. No taxonomy involved — these are induced entirely by co-timing.</p>
      {cluster_html}
    </section>

    <section class="section">
      <h2>Strongest Pairs</h2>
      <table>
        <thead><tr><th>Article</th><th>Co-spikes with</th><th>Lift</th><th>Support</th><th>Example days</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    '''
    nav = '<a href="index.html">← Archive</a> <a href="longitudinal.html">Longitudinal</a>'
    return html_page(title='Co-Spike Correlations',
                     subtitle=f'{score} scoring &nbsp;·&nbsp; &plusmn;{window}-day window',
                     nav=nav, body=body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--score', choices=['residual', 'flat'], default='residual')
    ap.add_argument('--window', type=int, default=1,
                    help='Co-occurrence window in days each side (default 1).')
    ap.add_argument('--threshold', type=float, default=3.0)
    ap.add_argument('--years', type=int, nargs='*',
                    help='Restrict to these years. Default is the whole archive.')
    ap.add_argument('--min-lift', type=float, default=5.0,
                    help='Lift floor for cluster membership (default 5).')
    ap.add_argument('--top-edges', type=int, default=500,
                    help='How many of the strongest edges feed clustering (default 500).')
    ap.add_argument('--no-report', action='store_true')
    ap.add_argument('--data-dir', type=Path, default=DATA_DIR)
    ap.add_argument('--reports-dir', type=Path, default=REPORTS_DIR)
    args = ap.parse_args()

    if args.years:
        frames = []
        for y in args.years:
            try:
                frames.append(load_year_data(args.data_dir, y))
            except FileNotFoundError:
                print(f'  no data for {y}, skipping')
        if not frames:
            print('No data for the requested years.', file=sys.stderr)
            return 1
        raw = pd.concat(frames, ignore_index=True)
    else:
        print('Loading all data...')
        raw = load_all_data(args.data_dir)

    df = filter_content(raw)
    print(f"  {len(df):,} content rows, {df['date'].dt.date.nunique():,} days, "
          f"{df['article'].nunique():,} articles")

    print(f'Detecting spikes ({args.score} scoring, threshold {args.threshold})...')
    spikes = spike_days(df, args.score, args.threshold)
    print(f'  {len(spikes):,} spike days across {spikes["article"].nunique():,} articles')

    print(f'Building co-spike graph (±{args.window} day window)...')
    edges, diag = build_edges(spikes, args.window)
    print(f"  {diag['n_pairs_considered']:,} pairs seen, {diag['n_edges']:,} edges with "
          f"support ≥ {MIN_SUPPORT}")
    if diag['days_dropped_as_sitewide']:
        print(f"  {diag['days_dropped_as_sitewide']} day(s) skipped as site-wide "
              f"(> {MAX_SPIKES_PER_DAY} simultaneous spikes)")

    clusters = cluster(edges, args.top_edges, args.min_lift)
    print(f'  {len(clusters)} cluster(s) at lift ≥ {args.min_lift}')

    out_dir = args.data_dir / 'correlations'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'edges_{args.score}.json'
    out_file.write_text(json.dumps(
        {'diagnostics': diag, 'params': vars(args) | {'data_dir': str(args.data_dir),
                                                      'reports_dir': str(args.reports_dir)},
         'edges': edges[:2000], 'clusters': clusters}, indent=2, default=str))
    print(f'→ {out_file}')

    print('\n=== Strongest pairs ===')
    for e in edges[:12]:
        print(f"  {e['lift']:>8.1f}x  support={e['support']:<3} "
              f"{e['a'][:34]:<34} + {e['b'][:34]}")

    print('\n=== Clusters ===')
    for i, group in enumerate(clusters[:8], 1):
        print(f"  {i}. " + ' · '.join(a.replace('_', ' ')[:26] for a in group[:6])
              + (f'  (+{len(group)-6})' if len(group) > 6 else ''))

    if not args.no_report:
        args.reports_dir.mkdir(parents=True, exist_ok=True)
        report = args.reports_dir / 'correlations.html'
        report.write_text(build_html(edges, clusters, diag, args.score, args.window))
        print(f'\n→ {report}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
