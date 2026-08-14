#!/usr/bin/env python3
"""Generate per-year Wikipedia pageviews HTML report.

Usage:
    ./scripts/generate-year-report.py 2025         # Single year
    ./scripts/generate-year-report.py --all        # All years in archive
    ./scripts/generate-year-report.py 2025 --no-narratives   # Skip LLM calls
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.wikipedia.analysis import (
    load_all_data, load_year_data, filter_content, detect_spikes,
)
from shared.wikipedia.narratives import batch_generate
from shared.wikipedia.report_utils import (
    COLORS, setup_plot_style, fig_to_base64, format_number,
    wiki_link, make_badge, plot_top_articles, plot_spike_examples, html_page,
)

DATA_DIR = ROOT / 'data'
REPORTS_DIR = ROOT / 'reports' / 'by-year'


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_monthly_traffic(df: pd.DataFrame, year: int) -> str:
    df = df.copy()
    df['month'] = df['date'].dt.month
    monthly = df.groupby('month')['views'].sum()
    # Fill missing months with 0
    monthly = monthly.reindex(range(1, 13), fill_value=0)
    labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(labels, monthly.values, color=COLORS['info'], alpha=0.85)
    ax.set_ylabel('Total Views (Top 1000)')
    ax.set_title(f'{year} — Monthly Traffic', fontsize=13, fontweight='bold', color=COLORS['text'])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
    ax.grid(True, axis='y', alpha=0.3)
    max_v = monthly.max()
    for bar, val in zip(bars, monthly.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val + max_v * 0.01,
                    f'{val/1e6:.1f}M', ha='center', fontsize=8, color=COLORS['text'])
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_day_of_week(df: pd.DataFrame) -> str:
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    daily = df.groupby('date')['views'].sum().reset_index()
    daily['dow'] = daily['date'].dt.day_name()
    dow_avg = daily.groupby('dow')['views'].mean().reindex(dow_order)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(dow_order, dow_avg.values, color=COLORS['success'], alpha=0.85)
    ax.set_ylabel('Avg Daily Views')
    ax.set_title('Traffic by Day of Week', fontsize=13, fontweight='bold', color=COLORS['text'])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
    for i, bar in enumerate(bars):
        if i >= 5:
            bar.set_color(COLORS['warning'])
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _spikes_table(spike_df: pd.DataFrame, narratives: dict) -> str:
    if spike_df.empty:
        return '<p style="color:#a0a0a0">No significant spikes detected.</p>'
    rows = ''
    for _, row in spike_df.head(10).iterrows():
        date_str = str(row['spike_date'])[:10]
        key = f"{row['article']}::{date_str}"
        narrative = narratives.get(key, '')
        narrative_html = (f'<div class="narrative">{narrative}</div>'
                          if narrative else '')
        rows += f'''
        <tr>
          <td>{wiki_link(row['article'])}{narrative_html}</td>
          <td>{date_str}</td>
          <td>{format_number(row['spike_views'])}</td>
          <td>{make_badge(f"{row['multiplier']:.0f}x", COLORS['highlight'])}</td>
        </tr>'''
    return f'''
    <table>
      <thead><tr><th>Article</th><th>Date</th><th>Peak Views</th><th>Spike</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>'''


def generate_year_html(df: pd.DataFrame, year: int, prior_total: int | None,
                       spike_df: pd.DataFrame, narratives: dict, plots: dict) -> str:
    days = df['date'].dt.date.nunique()
    total = int(df['views'].sum())
    unique_articles = df['article'].nunique()
    peak_day = df.groupby('date')['views'].sum().idxmax()
    peak_views = int(df.groupby('date')['views'].sum().max())

    # YoY
    if prior_total and prior_total > 0:
        pct = (total - prior_total) / prior_total * 100
        sign = '+' if pct >= 0 else ''
        cls = 'yoy-up' if pct >= 0 else 'yoy-down'
        yoy_html = f'<p style="margin-bottom:.5rem">vs {year-1}: <span class="{cls}"><strong>{sign}{pct:.1f}%</strong></span> total views</p>'
    else:
        yoy_html = ''

    date_min = df['date'].min().strftime('%Y-%m-%d')
    date_max = df['date'].max().strftime('%Y-%m-%d')

    # Top articles table
    top = df.groupby('article')['views'].sum().nlargest(20).reset_index()
    top_rows = ''
    for i, row in top.iterrows():
        top_rows += f'<tr><td>{make_badge(i+1, COLORS["accent"])}</td><td>{wiki_link(row["article"])}</td><td>{format_number(row["views"])}</td></tr>'

    spike_chart = (f'<div class="chart"><img src="data:image/png;base64,{plots["spikes"]}" alt="Spikes"></div>'
                   if plots.get('spikes') else '')

    body = f'''
    <section class="section">
      <h2>Overview</h2>
      {yoy_html}
      <div class="metrics">
        <div class="metric"><div class="metric-value">{format_number(total)}</div><div class="metric-label">Total Views</div></div>
        <div class="metric"><div class="metric-value">{days}</div><div class="metric-label">Days Covered</div></div>
        <div class="metric"><div class="metric-value">{format_number(unique_articles)}</div><div class="metric-label">Unique Articles</div></div>
        <div class="metric"><div class="metric-value">{format_number(peak_views)}</div><div class="metric-label">Peak Day Views</div></div>
      </div>
      <p style="color:{COLORS['muted']};font-size:.85rem">Peak day: {str(peak_day)[:10]}</p>
    </section>

    <section class="section">
      <h2>Monthly Traffic</h2>
      <div class="chart"><img src="data:image/png;base64,{plots['monthly']}" alt="Monthly"></div>
    </section>

    <div class="grid">
      <section class="section highlight">
        <h2>Top 20 Articles</h2>
        <table>
          <thead><tr><th>#</th><th>Article</th><th>Views</th></tr></thead>
          <tbody>{top_rows}</tbody>
        </table>
      </section>

      <section class="section warning">
        <h2>Traffic by Day of Week</h2>
        <div class="chart"><img src="data:image/png;base64,{plots['dow']}" alt="Day of Week"></div>
      </section>
    </div>

    <section class="section">
      <h2>Top Articles by Views</h2>
      <div class="chart"><img src="data:image/png;base64,{plots['top_articles']}" alt="Top Articles"></div>
    </section>

    <section class="section highlight">
      <h2>Biggest Traffic Spikes</h2>
      {_spikes_table(spike_df, narratives)}
      {spike_chart}
    </section>
    '''

    nav = '<a href="../index.html">← Archive</a> <a href="../all-time.html">All Time</a>'
    return html_page(
        title=f'{year} Wikipedia Pageviews',
        subtitle=f'{date_min} → {date_max} &nbsp;·&nbsp; {days} days',
        nav=nav,
        body=body,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_year(year: int, all_df: pd.DataFrame | None, narratives_flag: bool,
                 refresh_degraded: bool = False) -> None:
    print(f"\n=== {year} ===")

    if all_df is not None:
        df_year = all_df[all_df['date'].dt.year == year].copy()
    else:
        df_year = load_year_data(DATA_DIR, year)

    if df_year.empty:
        print(f"  No data, skipping.")
        return

    df = filter_content(df_year)

    # Prior year total (for YoY)
    prior_total = None
    if all_df is not None:
        prior_df = filter_content(all_df[all_df['date'].dt.year == year - 1])
        if not prior_df.empty:
            prior_total = int(prior_df['views'].sum())
    else:
        try:
            prior_raw = load_year_data(DATA_DIR, year - 1)
            prior_total = int(filter_content(prior_raw)['views'].sum())
        except FileNotFoundError:
            pass

    # Spikes
    spike_df = detect_spikes(df)

    # Narratives
    narratives: dict = {}
    if narratives_flag and not spike_df.empty:
        narratives = batch_generate(spike_df.to_dict('records'),
                                    refresh_degraded=refresh_degraded)

    # Plots
    setup_plot_style()
    plots = {
        'monthly':      plot_monthly_traffic(df, year),
        'top_articles': plot_top_articles(df, n=20, title=f'{year} — Top 20 Articles by Views'),
        'dow':          plot_day_of_week(df),
    }
    spike_chart = plot_spike_examples(df, spike_df)
    if spike_chart:
        plots['spikes'] = spike_chart

    # Render
    html = generate_year_html(df, year, prior_total, spike_df, narratives, plots)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f'{year}.html'
    out.write_text(html)
    print(f"  → {out}")


def main():
    parser = argparse.ArgumentParser(description='Generate per-year Wikipedia pageviews report')
    parser.add_argument('year', nargs='?', type=int, help='Year to generate (e.g. 2025)')
    parser.add_argument('--all', action='store_true', help='Generate reports for all years in archive')
    parser.add_argument('--no-narratives', action='store_true', help='Skip LLM spike narratives')
    parser.add_argument('--refresh-narratives', action='store_true',
                        help='Re-fetch cached narratives that are refusals rather than '
                             'explanations. Overwrites those cache entries; leaves good '
                             'ones untouched. Off by default — the cache is not reproducible.')
    args = parser.parse_args()

    if not args.year and not args.all:
        parser.error('Provide a year or --all')

    narratives_flag = not args.no_narratives

    if args.refresh_narratives and not narratives_flag:
        parser.error('--refresh-narratives cannot be combined with --no-narratives')

    if args.all:
        print("Loading all data...")
        all_df = load_all_data(DATA_DIR)
        years = sorted(all_df['date'].dt.year.unique())
        print(f"Years in archive: {years[0]}–{years[-1]}")
        for year in years:
            process_year(year, all_df, narratives_flag, args.refresh_narratives)
    else:
        process_year(args.year, None, narratives_flag, args.refresh_narratives)

    print("\nDone.")


if __name__ == '__main__':
    main()
