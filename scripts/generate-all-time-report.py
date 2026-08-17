#!/usr/bin/env python3
"""Generate all-time Wikipedia pageviews report and archive index.

Outputs:
    reports/all-time.html   — decade overview, longitudinal trends, biggest spikes
    reports/index.html      — navigation hub with coverage stats and year links

Usage:
    ./scripts/generate-all-time-report.py
    ./scripts/generate-all-time-report.py --no-narratives
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.wikipedia.analysis import (
    load_all_data, filter_content, detect_spikes, longitudinal_data,
)
from shared.wikipedia.narratives import batch_generate
from shared.wikipedia import baseline
from shared.wikipedia.report_utils import (
    COLORS, PALETTE, setup_plot_style, fig_to_base64, format_number,
    wiki_link, make_badge, plot_top_articles, plot_spike_examples, html_page,
)

DATA_DIR   = ROOT / 'data'
REPORTS_DIR = ROOT / 'reports'


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_annual_traffic(df: pd.DataFrame) -> str:
    df = df.copy()
    df['year'] = df['date'].dt.year
    annual = df.groupby('year')['views'].sum()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(annual.index, annual.values, alpha=0.25, color=COLORS['info'])
    ax.plot(annual.index, annual.values, color=COLORS['info'], linewidth=2, marker='o', markersize=5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Views (Top 1000)')
    ax.set_title('Annual Wikipedia Traffic — All Time', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.0f}M'))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_longitudinal(long_df: pd.DataFrame) -> str:
    """Multi-line chart: annual views for the top-10 all-time articles."""
    articles = long_df['article'].unique()
    years = sorted(long_df['year'].unique())

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, article in enumerate(articles):
        art = long_df[long_df['article'] == article].sort_values('year')
        label = article.replace('_', ' ')[:30]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(art['year'], art['total_views'], marker='o', markersize=4,
                linewidth=1.8, color=color, label=label, alpha=0.9)

    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Views')
    ax.set_title('Top 10 Articles — View Trajectories', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc='upper left', fontsize=8, framealpha=0.7,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_seasonal(df: pd.DataFrame) -> str:
    """Average total views by month-of-year across all years."""
    df = df.copy()
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    daily = df.groupby('date')['views'].sum().reset_index()
    daily['month'] = pd.to_datetime(daily['date']).dt.month
    monthly_avg = daily.groupby('month')['views'].mean()

    labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, [monthly_avg.get(m, 0) for m in range(1, 13)],
                  color=COLORS['success'], alpha=0.85)
    ax.set_ylabel('Avg Daily Views')
    ax.set_title('Seasonal Pattern — Avg Traffic by Month (all years)', fontsize=13,
                 fontweight='bold', color=COLORS['text'])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _spikes_table(spike_df: pd.DataFrame, narratives: dict) -> str:
    if spike_df.empty:
        return '<p style="color:#a0a0a0">No significant spikes detected.</p>'
    rows = ''
    for _, row in spike_df.head(20).iterrows():
        date_str = str(row['spike_date'])[:10]
        key = f"{row['article']}::{date_str}"
        narrative = narratives.get(key, '')
        narrative_html = (f'<div class="narrative">{narrative}</div>'
                          if narrative else '')
        rows += f'''
        <tr>
          <td>{wiki_link(row["article"])}{narrative_html}</td>
          <td>{date_str}</td>
          <td>{format_number(row["spike_views"])}</td>
          <td>{make_badge(f"{row['multiplier']:.0f}x", COLORS["highlight"])}</td>
        </tr>'''
    return f'''
    <table>
      <thead><tr><th>Article</th><th>Date</th><th>Peak Views</th><th>Spike</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>'''


def generate_all_time_html(df: pd.DataFrame, spike_df: pd.DataFrame,
                            narratives: dict, plots: dict) -> str:
    total_views = int(df['views'].sum())
    days = df['date'].dt.date.nunique()
    unique_articles = df['article'].nunique()
    date_min = df['date'].min().strftime('%Y-%m-%d')
    date_max = df['date'].max().strftime('%Y-%m-%d')

    # All-time top 20 table
    top = df.groupby('article')['views'].sum().nlargest(20).reset_index()
    top_rows = ''
    for i, row in top.iterrows():
        top_rows += (f'<tr><td>{make_badge(i+1, COLORS["accent"])}</td>'
                     f'<td>{wiki_link(row["article"])}</td>'
                     f'<td>{format_number(row["views"])}</td></tr>')

    spike_chart = (f'<div class="chart"><img src="data:image/png;base64,{plots["spikes"]}" alt="Spikes"></div>'
                   if plots.get('spikes') else '')

    body = f'''
    <section class="section">
      <h2>Overview</h2>
      <div class="metrics">
        <div class="metric"><div class="metric-value">{date_min[:4]}–{date_max[:4]}</div><div class="metric-label">Date Range</div></div>
        <div class="metric"><div class="metric-value">{format_number(days)}</div><div class="metric-label">Days in Archive</div></div>
        <div class="metric"><div class="metric-value">{format_number(total_views)}</div><div class="metric-label">Total Views</div></div>
        <div class="metric"><div class="metric-value">{format_number(unique_articles)}</div><div class="metric-label">Unique Articles</div></div>
      </div>
    </section>

    <section class="section">
      <h2>Annual Traffic Trend</h2>
      <div class="chart"><img src="data:image/png;base64,{plots['annual']}" alt="Annual Traffic"></div>
    </section>

    <div class="grid">
      <section class="section highlight">
        <h2>All-Time Top 20 Articles</h2>
        <table>
          <thead><tr><th>#</th><th>Article</th><th>Total Views</th></tr></thead>
          <tbody>{top_rows}</tbody>
        </table>
      </section>

      <section class="section warning">
        <h2>Seasonal Pattern</h2>
        <div class="chart"><img src="data:image/png;base64,{plots['seasonal']}" alt="Seasonal"></div>
      </section>
    </div>

    <section class="section">
      <h2>Top Articles</h2>
      <div class="chart"><img src="data:image/png;base64,{plots['top_articles']}" alt="Top Articles"></div>
    </section>

    <section class="section success">
      <h2>View Trajectories — Top 10 Articles</h2>
      <p style="color:{COLORS['muted']};font-size:.9rem;margin-bottom:.75rem">Annual total views for the 10 most-viewed articles of all time.</p>
      <div class="chart"><img src="data:image/png;base64,{plots['longitudinal']}" alt="Trajectories"></div>
    </section>

    <section class="section highlight">
      <h2>Biggest Spikes — All Time</h2>
      <p style="color:{COLORS['muted']};font-size:.9rem;margin-bottom:.75rem">Articles with the largest single-day view multiplier vs. their average.</p>
      {_spikes_table(spike_df, narratives)}
      {spike_chart}
    </section>
    '''

    nav = '<a href="index.html">← Archive Index</a>'
    return html_page(
        title='Wikipedia Pageviews — All Time',
        subtitle=f'{date_min} → {date_max}',
        nav=nav,
        body=body,
    )


def generate_index_html(df: pd.DataFrame) -> str:
    df = df.copy()
    df['year'] = df['date'].dt.year

    # Archive coverage
    total_days = df['date'].dt.date.nunique()
    date_min = df['date'].min().strftime('%Y-%m-%d')
    date_max = df['date'].max().strftime('%Y-%m-%d')

    # Expected days vs actual
    from datetime import date as date_type
    d_min = df['date'].min().date()
    d_max = df['date'].max().date()
    expected = (d_max - d_min).days + 1
    gaps = expected - total_days

    # Year table
    annual = df.groupby('year').agg(
        total_views=('views', 'sum'),
        days=('date', pd.Series.nunique),
    ).reset_index().sort_values('year', ascending=False)

    year_rows = ''
    for _, row in annual.iterrows():
        year = int(row['year'])
        year_rows += (
            f'<tr>'
            f'<td><a href="by-year/{year}.html">{year}</a></td>'
            f'<td>{int(row["days"])}</td>'
            f'<td>{format_number(row["total_views"])}</td>'
            f'</tr>'
        )

    # Longitudinal analysis is generated by a separate script; only link it if it exists,
    # so the index never carries a dead link.
    dashboard_tile = ''
    if (REPORTS_DIR / 'dashboard.html').exists():
        dashboard_tile = (
            '<div class="metric">'
            '<div class="metric-value"><a href="dashboard.html">Dashboard →</a></div>'
            '<div class="metric-label">Pipeline Health</div>'
            '</div>'
        )

    longitudinal_tile = ''
    if (REPORTS_DIR / 'longitudinal.html').exists():
        longitudinal_tile = (
            '<div class="metric">'
            '<div class="metric-value"><a href="longitudinal.html">Longitudinal →</a></div>'
            '<div class="metric-label">Appearance Density</div>'
            '</div>'
        )

    body = f'''
    <section class="section">
      <h2>Archive Coverage</h2>
      <div class="metrics">
        <div class="metric"><div class="metric-value">{format_number(total_days)}</div><div class="metric-label">Days in Archive</div></div>
        <div class="metric"><div class="metric-value">{date_min[:4]}–{date_max[:4]}</div><div class="metric-label">Date Range</div></div>
        <div class="metric"><div class="metric-value">{gaps}</div><div class="metric-label">Missing Days</div></div>
        <div class="metric"><div class="metric-value"><a href="all-time.html">All Time →</a></div><div class="metric-label">Decade Overview</div></div>
        {longitudinal_tile}
        {dashboard_tile}
      </div>
    </section>

    <section class="section">
      <h2>By Year</h2>
      <table>
        <thead><tr><th>Year</th><th>Days</th><th>Total Views</th></tr></thead>
        <tbody>{year_rows}</tbody>
      </table>
    </section>
    '''

    return html_page(
        title='Wikipedia Pageviews Archive',
        subtitle=f'{date_min} → {date_max} &nbsp;·&nbsp; {format_number(total_days)} days',
        nav='',
        body=body,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate all-time report and index')
    parser.add_argument('--no-narratives', action='store_true', help='Skip LLM spike narratives')
    parser.add_argument('--spike-method', choices=['residual', 'flat'], default='residual',
                        help="How to score spikes. 'residual' (default) measures against an "
                             'expectation baseline, so fixtures and calendar recurrences stop '
                             "dominating. 'flat' is the original per-article mean.")
    parser.add_argument('--refresh-narratives', action='store_true',
                        help='Re-fetch cached narratives that are refusals rather than '
                             'explanations. Overwrites those cache entries; leaves good '
                             'ones untouched. Off by default — the cache is not reproducible.')
    args = parser.parse_args()

    if args.refresh_narratives and args.no_narratives:
        parser.error('--refresh-narratives cannot be combined with --no-narratives')

    print("Loading all data...")
    df_raw = load_all_data(DATA_DIR)
    print(f"  {len(df_raw):,} records, {df_raw['date'].dt.date.nunique()} days")

    df = filter_content(df_raw)
    print(f"  After filter: {len(df):,} records")

    print("Detecting spikes...")
    context: dict | None = None
    if args.spike_method == 'flat':
        spike_df = detect_spikes(df)
    else:
        print("  fitting expectation baseline...")
        enriched = baseline.add_expectation(df, seasonal=True)
        spike_df = baseline.detect_spikes_residual(enriched=enriched)
    print(f"  {len(spike_df)} spiking articles found")

    narratives: dict = {}
    if not args.no_narratives and not spike_df.empty:
        if args.spike_method != 'flat':
            context = baseline.co_spike_context(enriched, spike_df.head(20))
        narratives = batch_generate(spike_df.to_dict('records'), top_n=20,
                                    refresh_degraded=args.refresh_narratives,
                                    context=context)

    print("Computing longitudinal data (top 10 articles)...")
    long_df = longitudinal_data(df, top_n=10)

    print("Generating plots...")
    setup_plot_style()
    plots = {
        'annual':       plot_annual_traffic(df),
        'top_articles': plot_top_articles(df, n=20, title='All-Time Top 20 Articles'),
        'longitudinal': plot_longitudinal(long_df),
        'seasonal':     plot_seasonal(df),
    }
    spike_chart = plot_spike_examples(df, spike_df)
    if spike_chart:
        plots['spikes'] = spike_chart

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Writing reports/all-time.html...")
    html = generate_all_time_html(df, spike_df, narratives, plots)
    (REPORTS_DIR / 'all-time.html').write_text(html)

    print("Writing reports/index.html...")
    index_html = generate_index_html(df_raw)
    (REPORTS_DIR / 'index.html').write_text(index_html)

    print(f"\nDone.")
    print(f"  file://{(REPORTS_DIR / 'index.html').absolute()}")
    print(f"  file://{(REPORTS_DIR / 'all-time.html').absolute()}")


if __name__ == '__main__':
    main()
