#!/usr/bin/env python3
"""Longitudinal appearance density and article-stability analysis.

Answers questions about how articles appear across the full archive:
  - Distribution of appearance frequency (power-law? bimodal?)
  - How many one-shot articles? At what rank do they appear?
  - Which articles have the most steady traffic (lowest CV)?
  - Which articles have the most steady rank (lowest rank variance)?
  - Which articles have the highest rank volatility?
  - What lives at rank 900–1000 — hoverers, rising newcomers, or fading perennials?
  - Seasonal detection via lag-365 autocorrelation (no taxonomy required).

Output:
  data/longitudinal/stats.json   — all computed metrics (cache for reports)
  reports/longitudinal.html      — visual report

Usage:
    uv run python scripts/analyze-longitudinal.py
    uv run python scripts/analyze-longitudinal.py --no-report   # stats + print only
    uv run python scripts/analyze-longitudinal.py --no-cache    # recompute stats even if cache exists
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.wikipedia.analysis import load_all_data, filter_content
from shared.wikipedia.report_utils import (
    COLORS, setup_plot_style, fig_to_base64, format_number,
    wiki_link, html_page,
)

DATA_DIR = ROOT / 'data'
CACHE_DIR = DATA_DIR / 'longitudinal'
REPORTS_DIR = ROOT / 'reports'
CACHE_FILE = CACHE_DIR / 'stats.json'

# Strata boundaries (days present out of total archive days)
STRATA = [
    ('Permanent',   0.90, 1.01,  '#4ecca3'),  # ≥90% of days
    ('Perennial',   0.25, 0.90,  '#00adb5'),  # 25–90%
    ('Occasional',  0.05, 0.25,  '#ffc93c'),  # 5–25%
    ('Rare',        0.01, 0.05,  '#e94560'),  # 1–5%
    ('One-shot',    0.00, 0.01,  '#a0a0a0'),  # <1% (includes exactly-once)
]


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> dict:
    """Compute all longitudinal metrics. df must be pre-filtered content."""
    print("Computing appearance frequencies...")
    total_days = df['date'].dt.date.nunique()

    # Days present per article
    presence = df.groupby('article').agg(
        days_present=('date', 'nunique'),
        total_views=('views', 'sum'),
        mean_views=('views', 'mean'),
        std_views=('views', 'std'),
        mean_rank=('rank', 'mean'),
        std_rank=('rank', 'std'),
        min_rank=('rank', 'min'),
        max_rank=('rank', 'max'),
        median_rank=('rank', 'median'),
    ).reset_index()

    presence['appearance_rate'] = presence['days_present'] / total_days
    presence['cv_views'] = presence['std_views'] / presence['mean_views']
    presence['rank_volatility'] = presence['max_rank'] - presence['min_rank']

    n_articles = len(presence)
    n_oneshot = int((presence['days_present'] == 1).sum())

    # Strata assignment
    def assign_stratum(rate):
        for name, lo, hi, _ in STRATA:
            if lo <= rate < hi:
                return name
        return 'One-shot'
    presence['stratum'] = presence['appearance_rate'].apply(assign_stratum)

    strata_counts = presence.groupby('stratum')['article'].count().to_dict()

    # One-shot rank distribution
    oneshot_df = presence[presence['days_present'] == 1].copy()
    # We need rank for their single appearance — fetch from original df
    # (presence['min_rank'] == max_rank for oneshots)
    oneshot_rank_hist = {
        '1-100':   int((oneshot_df['median_rank'] <= 100).sum()),
        '101-500': int(((oneshot_df['median_rank'] > 100) & (oneshot_df['median_rank'] <= 500)).sum()),
        '501-900': int(((oneshot_df['median_rank'] > 500) & (oneshot_df['median_rank'] <= 900)).sum()),
        '901-1000':int((oneshot_df['median_rank'] > 900).sum()),
    }

    # Steady traffic: lowest CV among articles present ≥500 days
    frequent = presence[presence['days_present'] >= 500].copy()
    steady_traffic = frequent.nsmallest(20, 'cv_views')[
        ['article', 'days_present', 'mean_views', 'cv_views']
    ].to_dict('records')

    # Steady rank: lowest rank std among frequent articles
    steady_rank = frequent.nsmallest(20, 'std_rank')[
        ['article', 'days_present', 'mean_rank', 'std_rank']
    ].to_dict('records')

    # Volatile rank: highest rank volatility among frequent articles
    volatile_rank = frequent.nlargest(20, 'rank_volatility')[
        ['article', 'days_present', 'min_rank', 'max_rank', 'rank_volatility', 'mean_rank']
    ].to_dict('records')

    # Rank 900–1000 stratum: median rank in that band, ≥50 appearances
    hoverers_mask = (
        (presence['median_rank'] >= 900) &
        (presence['days_present'] >= 50)
    )
    hoverers = presence[hoverers_mask].sort_values('days_present', ascending=False).head(30)[
        ['article', 'days_present', 'mean_rank', 'std_rank', 'total_views']
    ].to_dict('records')

    # Appearance frequency distribution — bucketed for JSON storage
    freq_hist, freq_edges = np.histogram(
        presence['days_present'],
        bins=np.logspace(0, np.log10(total_days), 60)
    )
    freq_distribution = {
        'bin_left': [float(x) for x in freq_edges[:-1]],
        'bin_right': [float(x) for x in freq_edges[1:]],
        'count': [int(x) for x in freq_hist],
    }

    print("Computing seasonal detection (lag-365 autocorrelation)...")
    # Only for articles with ≥700 appearances (enough data)
    seasonal_candidates = presence[presence['days_present'] >= 700]['article'].tolist()
    seasonal_results = compute_seasonal(df, seasonal_candidates)

    return {
        'total_days': total_days,
        'total_articles': n_articles,
        'n_oneshot': n_oneshot,
        'strata_counts': strata_counts,
        'oneshot_rank_hist': oneshot_rank_hist,
        'steady_traffic': steady_traffic,
        'steady_rank': steady_rank,
        'volatile_rank': volatile_rank,
        'hoverers': hoverers,
        'freq_distribution': freq_distribution,
        'seasonal': seasonal_results,
    }


def compute_seasonal(df: pd.DataFrame, articles: list[str]) -> list[dict]:
    """Compute lag-365 autocorrelation for a list of articles.

    Returns top 20 highest-autocorrelation articles (most seasonal).
    """
    if not articles:
        return []

    # Build a date index spanning the full archive
    all_dates = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
    subset = df[df['article'].isin(articles)].copy()

    results = []
    total = len(articles)
    for i, article in enumerate(articles):
        if i % 100 == 0:
            print(f"  Seasonal: {i}/{total}", end='\r')
        art_df = subset[subset['article'] == article].set_index('date')['views']
        art_series = art_df.reindex(all_dates, fill_value=0)

        if len(art_series) < 730:  # need at least 2 years
            continue

        # Autocorrelation at lag 365
        values = art_series.values.astype(float)
        mean = values.mean()
        if mean == 0:
            continue
        normed = values - mean
        auto = np.correlate(normed, normed, mode='full')
        auto = auto / auto[len(auto) // 2]  # normalise by lag-0
        lag365 = float(auto[len(auto) // 2 + 365]) if len(auto) // 2 + 365 < len(auto) else 0.0

        results.append({
            'article': article,
            'lag365_autocorr': round(lag365, 4),
        })

    print()
    results.sort(key=lambda x: x['lag365_autocorr'], reverse=True)
    return results[:20]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_freq_distribution(stats: dict) -> str:
    dist = stats['freq_distribution']
    left = np.array(dist['bin_left'])
    counts = np.array(dist['count'])
    widths = np.array(dist['bin_right']) - left

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(left, counts, width=widths, color=COLORS['info'], alpha=0.85, align='edge')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Days Present in Archive (log scale)')
    ax.set_ylabel('Number of Articles (log scale)')
    ax.set_title('Article Appearance Frequency Distribution', fontsize=13,
                 fontweight='bold', color=COLORS['text'])
    ax.grid(True, alpha=0.3, which='both')

    # Stratum boundaries
    total_days = stats['total_days']
    for _, lo, _, color in STRATA[:-1]:
        ax.axvline(lo * total_days, color=color, linestyle='--', alpha=0.6, linewidth=1)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_strata_bars(stats: dict) -> str:
    order = [s[0] for s in STRATA]
    colors = {s[0]: s[3] for s in STRATA}
    counts = stats['strata_counts']
    labels = [s for s in order if s in counts]
    values = [counts[s] for s in labels]
    bar_colors = [colors[s] for s in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=bar_colors, alpha=0.85)
    ax.set_ylabel('Number of Articles')
    ax.set_title('Articles by Appearance Stratum', fontsize=13,
                 fontweight='bold', color=COLORS['text'])
    ax.set_yscale('log')
    ax.grid(True, axis='y', alpha=0.3, which='both')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.1,
                format_number(val), ha='center', fontsize=9, color=COLORS['text'])
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_oneshot_ranks(stats: dict) -> str:
    hist = stats['oneshot_rank_hist']
    labels = list(hist.keys())
    values = list(hist.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=COLORS['muted'], alpha=0.85)
    ax.set_ylabel('One-shot Articles')
    ax.set_title('One-shot Articles by Rank at Appearance', fontsize=13,
                 fontweight='bold', color=COLORS['text'])
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(values) * 0.01,
                format_number(val), ha='center', fontsize=9, color=COLORS['text'])
    plt.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML sections
# ---------------------------------------------------------------------------

def _table(headers: list[str], rows: list[list]) -> str:
    th = ''.join(f'<th>{h}</th>' for h in headers)
    body = ''
    for row in rows:
        body += '<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>'
    return f'<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def build_html(stats: dict, plots: dict) -> str:
    total_days = stats['total_days']
    n_articles = stats['total_articles']
    n_oneshot = stats['n_oneshot']
    sc = stats['strata_counts']

    # --- Overview ---
    overview = f'''
    <section class="section">
      <h2>Archive Overview</h2>
      <div class="metrics">
        <div class="metric"><div class="metric-value">{format_number(total_days)}</div><div class="metric-label">Days in Archive</div></div>
        <div class="metric"><div class="metric-value">{format_number(n_articles)}</div><div class="metric-label">Distinct Articles</div></div>
        <div class="metric"><div class="metric-value">{format_number(n_oneshot)}</div><div class="metric-label">One-shot Articles</div></div>
        <div class="metric"><div class="metric-value">{round(n_oneshot/n_articles*100,1)}%</div><div class="metric-label">One-shot Fraction</div></div>
      </div>
      <p style="color:{COLORS['muted']};font-size:.85rem">
        Avg appearance: {round(total_days * 1000 / n_articles, 1)} article-days per distinct article
        &nbsp;·&nbsp; {format_number(sc.get('Permanent', 0))} permanent
        &nbsp;·&nbsp; {format_number(sc.get('Perennial', 0))} perennial
        &nbsp;·&nbsp; {format_number(sc.get('Occasional', 0))} occasional
      </p>
    </section>
    '''

    # --- Frequency distribution ---
    freq_section = f'''
    <section class="section">
      <h2>Appearance Frequency Distribution</h2>
      <p class="narrative">Each bar shows how many distinct articles appear in the archive
      for that many days. Log-log axes reveal the power-law tail. Dashed vertical lines mark
      stratum boundaries (Permanent / Perennial / Occasional / Rare / One-shot).</p>
      <div class="chart"><img src="data:image/png;base64,{plots['freq_dist']}" alt="Frequency distribution"></div>
      <div class="chart"><img src="data:image/png;base64,{plots['strata']}" alt="Strata breakdown"></div>
    </section>
    '''

    # --- One-shot ---
    oneshot_section = f'''
    <section class="section highlight">
      <h2>One-shot Articles ({format_number(n_oneshot)})</h2>
      <p class="narrative">Articles appearing exactly once across the entire archive.
      Most one-shots live at rank 900+, but a surprising number break into the top 100 —
      these are single-day viral events.</p>
      <div class="chart"><img src="data:image/png;base64,{plots['oneshot_ranks']}" alt="One-shot rank distribution"></div>
    </section>
    '''

    # --- Steady traffic ---
    st_rows = [
        [wiki_link(r['article']),
         format_number(int(r['days_present'])),
         format_number(int(r['mean_views'])),
         f"{r['cv_views']:.3f}"]
        for r in stats['steady_traffic']
    ]
    steady_traffic_section = f'''
    <section class="section success">
      <h2>Most Steady Traffic (lowest CV, ≥500 days)</h2>
      <p class="narrative">Coefficient of variation = σ/μ of daily views. Low CV means
      traffic is consistent rather than spike-driven. These articles represent durable,
      ambient human curiosity.</p>
      {_table(['Article', 'Days Present', 'Avg Views/Day', 'CV'], st_rows)}
    </section>
    '''

    # --- Steady rank ---
    sr_rows = [
        [wiki_link(r['article']),
         format_number(int(r['days_present'])),
         f"{r['mean_rank']:.0f}",
         f"{r['std_rank']:.1f}"]
        for r in stats['steady_rank']
    ]
    steady_rank_section = f'''
    <section class="section success">
      <h2>Most Steady Rank (lowest rank σ, ≥500 days)</h2>
      <p class="narrative">Rank stability differs from traffic stability — an article can have
      volatile absolute views but a stable rank if its peers are equally volatile.</p>
      {_table(['Article', 'Days Present', 'Mean Rank', 'Rank σ'], sr_rows)}
    </section>
    '''

    # --- Volatile rank ---
    vr_rows = [
        [wiki_link(r['article']),
         format_number(int(r['days_present'])),
         str(int(r['min_rank'])),
         str(int(r['max_rank'])),
         format_number(int(r['rank_volatility']))]
        for r in stats['volatile_rank']
    ]
    volatile_section = f'''
    <section class="section highlight">
      <h2>Highest Rank Volatility (≥500 days)</h2>
      <p class="narrative">Articles that swing across the widest rank range. These are
      "news-reactive" articles with a large permanent base — they go from rank 10 to rank 900+
      depending on whether they are in the news.</p>
      {_table(['Article', 'Days Present', 'Min Rank', 'Max Rank', 'Range'], vr_rows)}
    </section>
    '''

    # --- Hoverers 900–1000 ---
    hov_rows = [
        [wiki_link(r['article']),
         format_number(int(r['days_present'])),
         f"{r['mean_rank']:.0f}",
         f"{r['std_rank']:.1f}",
         format_number(int(r['total_views']))]
        for r in stats['hoverers']
    ]
    hoverers_section = f'''
    <section class="section warning">
      <h2>Rank 900–1000 Hoverers (≥50 appearances)</h2>
      <p class="narrative">Articles whose median rank sits near the bottom of the top-1000.
      Candidates: slowly-declining former perennials, niche-audience pages with steady small
      demand, or seasonal articles caught mid-cycle.</p>
      {_table(['Article', 'Days Present', 'Mean Rank', 'Rank σ', 'Total Views'], hov_rows)}
    </section>
    '''

    # --- Seasonal ---
    if stats['seasonal']:
        seas_rows = [
            [wiki_link(r['article']), f"{r['lag365_autocorr']:.3f}"]
            for r in stats['seasonal']
        ]
        seasonal_section = f'''
        <section class="section">
          <h2>Most Seasonal Articles (lag-365 autocorrelation, ≥700 days)</h2>
          <p class="narrative">Articles with strong annual periodicity in their traffic —
          detected without any taxonomy, purely from the time series. High autocorrelation
          at lag-365 indicates the article spikes at roughly the same calendar period each year.</p>
          {_table(['Article', 'Lag-365 Autocorrelation'], seas_rows)}
        </section>
        '''
    else:
        seasonal_section = ''

    body = (overview + freq_section + oneshot_section +
            '<div class="grid">' + steady_traffic_section + steady_rank_section + '</div>' +
            volatile_section + hoverers_section + seasonal_section)

    return html_page(
        title='Wikipedia Pageviews — Longitudinal Analysis',
        subtitle=f'Appearance density & article stability across {format_number(total_days)} days',
        nav='<a href="index.html">← Archive</a> <a href="all-time.html">All Time</a>',
        body=body,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Longitudinal appearance density analysis')
    parser.add_argument('--no-report', action='store_true', help='Print stats only; skip HTML report')
    parser.add_argument('--no-cache', action='store_true', help='Recompute stats even if cache exists')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR,
                        help='Override path to data/ directory (useful when running from a worktree)')
    parser.add_argument('--reports-dir', type=Path, default=REPORTS_DIR,
                        help='Override path to reports/ directory')
    args = parser.parse_args()

    data_dir = args.data_dir
    reports_dir = args.reports_dir
    cache_dir = data_dir / 'longitudinal'
    cache_file = cache_dir / 'stats.json'

    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not args.no_cache:
        print(f"Loading cached stats from {cache_file}")
        stats = json.loads(cache_file.read_text())
    else:
        print("Loading all data...")
        raw = load_all_data(data_dir)
        df = filter_content(raw)
        print(f"  {len(df):,} content rows across {df['date'].dt.date.nunique():,} days, "
              f"{df['article'].nunique():,} distinct articles")
        stats = compute_stats(df)
        cache_file.write_text(json.dumps(stats, indent=2))
        print(f"Stats cached → {cache_file}")

    # Print summary
    print(f"\n=== Appearance Density Summary ===")
    print(f"Total days       : {format_number(stats['total_days'])}")
    print(f"Distinct articles: {format_number(stats['total_articles'])}")
    print(f"One-shot articles: {format_number(stats['n_oneshot'])} "
          f"({stats['n_oneshot']/stats['total_articles']*100:.1f}%)")
    print(f"\nStrata:")
    for name, _, _, _ in STRATA:
        count = stats['strata_counts'].get(name, 0)
        print(f"  {name:<12}: {format_number(count)}")
    print(f"\nTop 5 steady traffic (lowest CV):")
    for r in stats['steady_traffic'][:5]:
        print(f"  {r['article']:<40} CV={r['cv_views']:.3f}  days={r['days_present']}")
    print(f"\nTop 5 volatile rank:")
    for r in stats['volatile_rank'][:5]:
        print(f"  {r['article']:<40} range={r['rank_volatility']}  ({r['min_rank']}–{r['max_rank']})")
    if stats['seasonal']:
        print(f"\nTop 5 seasonal (lag-365 autocorr):")
        for r in stats['seasonal'][:5]:
            print(f"  {r['article']:<40} r={r['lag365_autocorr']:.3f}")

    if args.no_report:
        return

    print("\nGenerating report...")
    setup_plot_style()
    plots = {
        'freq_dist':    plot_freq_distribution(stats),
        'strata':       plot_strata_bars(stats),
        'oneshot_ranks': plot_oneshot_ranks(stats),
    }

    html = build_html(stats, plots)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / 'longitudinal.html'
    out.write_text(html)
    print(f"Report → {out}")


if __name__ == '__main__':
    main()
