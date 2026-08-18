#!/usr/bin/env python3
"""Tiered analysis: what the archive looks like below the headline articles.

    uv run python scripts/analyze-bands.py
    uv run python scripts/analyze-bands.py --years 2024 2025 2026
    uv run python scripts/analyze-bands.py --no-movers    # skip the expensive part

Answers "what does the world look like if you remove the top N" — without removing a
top N. Rank is relative to a truncated window, so dropping the top re-ranks inside the
archive rather than revealing anything beneath it, and band membership would shift
whenever the top moved. Absolute view bands do not move, so they can be compared across
days and across years.

Three views:

  BAND MIX          how many articles sit in each band each day. Structural, and it
                    changes over eleven years.
  ATTENTION SHARE   what fraction of all views the top band absorbs. The concentration
                    question: is attention pooling at the top over time?
  CENSORING LINE    views needed to make the daily top 1000. Every tiered result has to
                    be read against this — a band thinning out means nothing if the line
                    rose that day and pushed its members out of the archive entirely.

Plus the strongest residual movers WITHIN each band, which is where the interesting
middle actually becomes visible: it is invisible in a global ranking because the headline
articles occupy the whole top of it.

Output:
  data/bands/stats.json     — per-day series and per-band movers
  reports/bands.html        — visual report
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.wikipedia.analysis import (  # noqa: E402
    load_all_data, load_year_data, filter_content,
)
from shared.wikipedia import baseline  # noqa: E402
from shared.wikipedia.report_utils import (  # noqa: E402
    COLORS, setup_plot_style, fig_to_base64, format_number,
    wiki_link, make_badge, html_page,
)

DATA_DIR = ROOT / 'data'
REPORTS_DIR = ROOT / 'reports'

# Bands are ORDINAL (headline > major > middle > threshold), so this is a sequential ramp
# in one hue — light to dark — not the categorical PALETTE the other reports use. A
# categorical set here would imply the bands are unordered kinds, which they are not.
#
# Stepped against the dark surface with the palette validator rather than by eye. Adjacent
# steps clear the normal-vision floor (worst pair ΔE 15.9) and CVD separation, which
# matters because stacked segments touch. The validator also FAILs this palette on its
# lightness-band and chroma-floor checks: both are categorical-palette checks, and a
# sequential ramp necessarily spans a wide lightness range with desaturated ends — that
# span IS the encoding. Its darkest step lands under 3:1 contrast, so the legend and the
# table view below are required relief, not decoration.
BAND_COLORS = {
    'headline':  '#d9fbff',
    'major':     '#66d9e2',
    'middle':    '#0092a0',
    'threshold': '#00505c',
}
BAND_ORDER = ['headline', 'major', 'middle', 'threshold']
BAND_LABEL = {
    'headline':  'Headline (≥150k)',
    'major':     'Major (40k–150k)',
    'middle':    'Middle (15k–40k)',
    'threshold': 'Threshold (<15k)',
}


def compute(df: pd.DataFrame, with_movers: bool) -> dict:
    df = df.copy()
    df['band'] = baseline.view_band(df['views'])

    counts = (df.groupby([df['date'], 'band']).size()
              .unstack(fill_value=0).reindex(columns=BAND_ORDER, fill_value=0))
    views = (df.groupby([df['date'], 'band'])['views'].sum()
             .unstack(fill_value=0).reindex(columns=BAND_ORDER, fill_value=0))
    share = views.div(views.sum(axis=1), axis=0)

    cens = baseline.censoring_threshold(df).set_index('date')

    stats = {
        'dates': [d.strftime('%Y-%m-%d') for d in counts.index],
        'counts': {b: counts[b].astype(int).tolist() for b in BAND_ORDER},
        'share': {b: [round(v, 5) for v in share[b].tolist()] for b in BAND_ORDER},
        'censoring': {
            'threshold': cens['threshold_views'].astype(int).tolist(),
            'articles': cens['articles_present'].astype(int).tolist(),
        },
        'summary': {
            'days': int(len(counts)),
            'band_totals': {b: int(counts[b].sum()) for b in BAND_ORDER},
            'threshold_min': int(cens['threshold_views'].min()),
            'threshold_median': int(cens['threshold_views'].median()),
            'threshold_max': int(cens['threshold_views'].max()),
        },
        'movers': {},
    }

    if with_movers:
        print('  fitting expectation baseline for within-band movers...')
        enriched = baseline.add_expectation(df, seasonal=True)
        spikes = baseline.detect_spikes_residual(enriched=enriched)
        for band in BAND_ORDER:
            sub = spikes[spikes['band'] == band].head(12)
            stats['movers'][band] = [{
                'article': r['article'],
                'date': r['spike_date'].strftime('%Y-%m-%d'),
                'views': int(r['spike_views']),
                'residual': round(float(r['residual']), 1),
            } for _, r in sub.iterrows()]
    return stats


# ── Plots ─────────────────────────────────────────────────────────────────────

def _monthly(stats: dict, key: str) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Resample to month starts. Eleven years of daily points is denser than the pixels."""
    idx = pd.to_datetime(stats['dates'])
    frame = pd.DataFrame({b: stats[key][b] for b in BAND_ORDER}, index=idx)
    return frame.resample('MS').mean()


def plot_band_mix(stats: dict) -> str:
    m = _monthly(stats, 'counts')
    fig, ax = plt.subplots(figsize=(12, 4.5))
    # edgecolor at the surface colour is the 2px spacer between stacked segments: without
    # it adjacent bands share a hard edge and the boundary reads as a data feature.
    ax.stackplot(m.index, [m[b] for b in BAND_ORDER],
                 labels=[BAND_LABEL[b] for b in BAND_ORDER],
                 colors=[BAND_COLORS[b] for b in BAND_ORDER],
                 edgecolor=COLORS['bg'], linewidth=1.6)
    ax.set_ylabel('Articles per day')
    ax.set_title('Band mix — how the top 1000 divides by traffic', fontsize=13,
                 fontweight='bold', color=COLORS['text'], pad=26, loc='left')
    # Above the axes, not inside: this stack fills the full plot height, so any
    # in-axes placement covers data. Caught by rendering it and looking.
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=4, fontsize=8,
              frameon=False)
    ax.grid(True, axis='y', alpha=0.25)
    ax.margins(x=0)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_attention_share(stats: dict) -> str:
    m = _monthly(stats, 'share')
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(m.index, [m[b] * 100 for b in BAND_ORDER],
                 labels=[BAND_LABEL[b] for b in BAND_ORDER],
                 colors=[BAND_COLORS[b] for b in BAND_ORDER],
                 edgecolor=COLORS['bg'], linewidth=1.6)
    ax.set_ylabel('Share of daily views (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Attention share — how much of the traffic the top band absorbs',
                 fontsize=13, fontweight='bold', color=COLORS['text'], pad=26, loc='left')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=4, fontsize=8,
              frameon=False)
    ax.grid(True, axis='y', alpha=0.25)
    ax.margins(x=0)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_censoring(stats: dict) -> str:
    idx = pd.to_datetime(stats['dates'])
    s = pd.Series(stats['censoring']['threshold'], index=idx).resample('MS').mean()
    fig, ax = plt.subplots(figsize=(12, 3.6))
    # One series, so no legend box — the title names it.
    ax.plot(s.index, s.values, color=COLORS['warning'], linewidth=2)
    ax.set_ylabel('Views to make the top 1000')
    ax.set_title('Censoring line — what it took to chart',
                 fontsize=13, fontweight='bold', color=COLORS['text'])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
    ax.grid(True, alpha=0.25)
    ax.margins(x=0)
    plt.tight_layout()
    return fig_to_base64(fig)


# ── HTML ──────────────────────────────────────────────────────────────────────

def _movers_table(movers: list) -> str:
    if not movers:
        return f'<p style="color:{COLORS["muted"]}">No spikes in this band.</p>'
    rows = ''.join(
        f'<tr><td>{wiki_link(m["article"])}</td><td>{m["date"]}</td>'
        f'<td>{format_number(m["views"])}</td>'
        f'<td>{make_badge(f"{m["residual"]:.0f}x", COLORS["highlight"])}</td></tr>'
        for m in movers)
    return (f'<table><thead><tr><th>Article</th><th>Date</th><th>Peak Views</th>'
            f'<th>Residual</th></tr></thead><tbody>{rows}</tbody></table>')


def build_html(stats: dict, plots: dict) -> str:
    s = stats['summary']

    # The band-mix table is the required relief for the palette's darkest step, which sits
    # under 3:1 against the surface: anything unreadable in the chart is readable here.
    total = sum(s['band_totals'].values()) or 1
    mix_rows = ''.join(
        f'<tr><td><span style="display:inline-block;width:.7rem;height:.7rem;'
        f'background:{BAND_COLORS[b]};border-radius:2px;margin-right:.4rem"></span>'
        f'{BAND_LABEL[b]}</td><td>{format_number(s["band_totals"][b])}</td>'
        f'<td>{s["band_totals"][b] / total * 100:.1f}%</td></tr>'
        for b in BAND_ORDER)

    movers_html = ''.join(
        f'<section class="section"><h3>{BAND_LABEL[b]}</h3>'
        f'{_movers_table(stats["movers"].get(b, []))}</section>'
        for b in BAND_ORDER if stats['movers'].get(b))

    body = f'''
    <section class="section">
      <h2>Overview</h2>
      <p style="color:{COLORS['muted']}">Bands are absolute view thresholds, not rank
      slices. Rank is relative to a truncated window, so removing a top N re-ranks inside
      the archive rather than revealing anything below it — and band membership would move
      whenever the top moved.</p>
      <div class="metrics">
        <div class="metric"><div class="metric-value">{format_number(s['days'])}</div><div class="metric-label">Days</div></div>
        <div class="metric"><div class="metric-value">{format_number(s['threshold_median'])}</div><div class="metric-label">Median Censoring Line</div></div>
        <div class="metric"><div class="metric-value">{format_number(s['threshold_min'])}</div><div class="metric-label">Easiest Day to Chart</div></div>
        <div class="metric"><div class="metric-value">{format_number(s['threshold_max'])}</div><div class="metric-label">Hardest Day to Chart</div></div>
      </div>
    </section>

    <section class="section">
      <h2>Band Mix</h2>
      <div class="chart"><img src="data:image/png;base64,{plots['mix']}" alt="Band mix over time"></div>
      <table>
        <thead><tr><th>Band</th><th>Article-days</th><th>Share</th></tr></thead>
        <tbody>{mix_rows}</tbody>
      </table>
    </section>

    <section class="section warning">
      <h2>Attention Share</h2>
      <div class="chart"><img src="data:image/png;base64,{plots['share']}" alt="Attention share by band"></div>
    </section>

    <section class="section highlight">
      <h2>Censoring Line</h2>
      <p style="color:{COLORS['muted']};font-size:.88rem">The archive keeps the daily top
      1000, so absence is "below the line" rather than zero — and the line moves. A band
      thinning out means nothing on a day the line rose.</p>
      <div class="chart"><img src="data:image/png;base64,{plots['censoring']}" alt="Censoring threshold"></div>
    </section>

    <section class="section">
      <h2>Strongest Movers Within Band</h2>
      <p style="color:{COLORS['muted']};font-size:.88rem">Residual against the expectation
      baseline, ranked within band — a global ranking is all headline articles. The
      Threshold band is sparse by construction: the spike floor is 10k views and the band
      tops out at 15k, so only a sliver of it is eligible to register a spike at all.</p>
      {movers_html}
    </section>
    '''
    nav = ('<a href="index.html">← Archive</a> <a href="longitudinal.html">Longitudinal</a> '
           '<a href="correlations.html">Correlations</a>')
    return html_page(title='Tiered Analysis — View Bands',
                     subtitle=f"{s['days']} days &nbsp;·&nbsp; absolute view bands",
                     nav=nav, body=body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--years', type=int, nargs='*')
    ap.add_argument('--no-movers', action='store_true',
                    help='Skip the within-band movers (skips fitting the baseline).')
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
    print(f"  {len(df):,} content rows across {df['date'].dt.date.nunique():,} days")

    stats = compute(df, with_movers=not args.no_movers)

    out_dir = args.data_dir / 'bands'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'stats.json').write_text(json.dumps(stats, indent=2))
    print(f"→ {out_dir / 'stats.json'}")

    s = stats['summary']
    print('\n=== Band mix (article-days) ===')
    total = sum(s['band_totals'].values()) or 1
    for b in BAND_ORDER:
        n = s['band_totals'][b]
        print(f'  {BAND_LABEL[b]:<22} {n:>10,}  {n/total*100:5.1f}%')
    print(f"\nCensoring line: {s['threshold_min']:,} / {s['threshold_median']:,} / "
          f"{s['threshold_max']:,} views (min / median / max)")

    for b in BAND_ORDER:
        movers = stats['movers'].get(b, [])[:4]
        if movers:
            print(f'\n  top movers — {BAND_LABEL[b]}')
            for m in movers:
                print(f"    {m['residual']:>8.1f}x  {m['views']:>9,}  "
                      f"{m['article'][:40]:<40} {m['date']}")

    if not args.no_report:
        setup_plot_style()
        plots = {
            'mix': plot_band_mix(stats),
            'share': plot_attention_share(stats),
            'censoring': plot_censoring(stats),
        }
        args.reports_dir.mkdir(parents=True, exist_ok=True)
        report = args.reports_dir / 'bands.html'
        report.write_text(build_html(stats, plots))
        print(f'\n→ {report}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
