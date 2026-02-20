#!/usr/bin/env python3
"""Generate professional HTML reports with seaborn visualizations.

Usage:
    ./scripts/generate-report.py                    # Last 30 days
    ./scripts/generate-report.py --days 90          # Last 90 days
    ./scripts/generate-report.py --all              # All available data
"""

import argparse
import base64
import io
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.wikipedia.filters import is_content

DATA_DIR = Path(__file__).parent.parent / 'data'
REPORTS_DIR = Path(__file__).parent.parent / 'reports'

# Dark theme colors
COLORS = {
    'bg': '#1a1a2e',
    'card': '#16213e',
    'accent': '#0f3460',
    'highlight': '#e94560',
    'text': '#eaeaea',
    'muted': '#a0a0a0',
    'success': '#4ecca3',
    'warning': '#ffc93c',
    'info': '#00adb5',
}

def load_data(days: int = None) -> pd.DataFrame:
    """Load pageview data from JSON files."""
    files = sorted(glob(str(DATA_DIR / 'pageviews_*.json')))

    if not files:
        raise FileNotFoundError(f"No data files found in {DATA_DIR}")

    if days:
        files = files[-days:]

    records = []
    for filepath in files:
        with open(filepath) as f:
            data = json.load(f)
            records.extend(data)

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df


def filter_content(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out non-article content using shared filtering logic."""
    mask = df['article'].apply(is_content)
    return df[mask].copy()


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=COLORS['card'], edgecolor='none')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_str


def setup_plot_style():
    """Configure seaborn/matplotlib for dark theme."""
    plt.rcParams.update({
        'figure.facecolor': COLORS['card'],
        'axes.facecolor': COLORS['bg'],
        'axes.edgecolor': COLORS['muted'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['accent'],
        'grid.alpha': 0.3,
        'legend.facecolor': COLORS['card'],
        'legend.edgecolor': COLORS['accent'],
    })
    sns.set_palette([COLORS['highlight'], COLORS['info'], COLORS['success'],
                     COLORS['warning'], '#9b59b6', '#3498db'])


def plot_daily_traffic(df: pd.DataFrame) -> str:
    """Plot total daily pageviews over time."""
    daily = df.groupby('date')['views'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(daily['date'], daily['views'], alpha=0.3, color=COLORS['info'])
    ax.plot(daily['date'], daily['views'], color=COLORS['info'], linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Views (Top 1000)')
    ax.set_title('Daily Wikipedia Traffic', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig_to_base64(fig)


def plot_top_articles(df: pd.DataFrame, n: int = 15) -> str:
    """Plot top articles by total views."""
    top = df.groupby('article')['views'].sum().nlargest(n).reset_index()
    top = top.iloc[::-1]  # Reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top['article'], top['views'], color=COLORS['highlight'], alpha=0.8)
    ax.set_xlabel('Total Views')
    ax.set_title(f'Top {n} Articles by Total Views', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))

    # Add value labels
    for bar, val in zip(bars, top['views']):
        ax.text(val + top['views'].max() * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val/1e6:.1f}M', va='center', fontsize=9, color=COLORS['text'])

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_day_of_week(df: pd.DataFrame) -> str:
    """Plot average views by day of week."""
    df = df.copy()
    df['dow'] = df['date'].dt.day_name()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    daily_total = df.groupby('date')['views'].sum().reset_index()
    daily_total['dow'] = daily_total['date'].dt.day_name()
    dow_avg = daily_total.groupby('dow')['views'].mean().reindex(dow_order)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(dow_order, dow_avg.values, color=COLORS['success'], alpha=0.8)
    ax.set_ylabel('Average Daily Views')
    ax.set_title('Traffic by Day of Week', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    plt.xticks(rotation=45, ha='right')

    # Highlight weekend
    for i, bar in enumerate(bars):
        if i >= 5:  # Saturday, Sunday
            bar.set_color(COLORS['warning'])

    plt.tight_layout()
    return fig_to_base64(fig)


def detect_spikes(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Detect articles with unusual spikes in views."""
    # Calculate mean and std for each article
    stats = df.groupby('article')['views'].agg(['mean', 'std', 'max']).reset_index()
    stats = stats[stats['std'] > 0]  # Need variance

    # Find days with spikes
    spikes = []
    for _, row in stats.iterrows():
        article_data = df[df['article'] == row['article']]
        z_scores = (article_data['views'] - row['mean']) / row['std']
        spike_days = article_data[z_scores > threshold]

        if len(spike_days) > 0:
            max_spike = spike_days.loc[spike_days['views'].idxmax()]
            spikes.append({
                'article': row['article'],
                'spike_date': max_spike['date'],
                'spike_views': max_spike['views'],
                'avg_views': row['mean'],
                'multiplier': max_spike['views'] / row['mean']
            })

    spike_df = pd.DataFrame(spikes)
    if len(spike_df) > 0:
        spike_df = spike_df.sort_values('multiplier', ascending=False)
    return spike_df


def plot_spike_examples(df: pd.DataFrame, spike_df: pd.DataFrame, n: int = 4) -> str:
    """Plot time series for top spiking articles."""
    if len(spike_df) == 0:
        return None

    top_spikes = spike_df.head(n)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    colors = [COLORS['highlight'], COLORS['info'], COLORS['success'], COLORS['warning']]

    for i, (_, spike) in enumerate(top_spikes.iterrows()):
        if i >= len(axes):
            break
        ax = axes[i]
        article_data = df[df['article'] == spike['article']].sort_values('date')

        ax.fill_between(article_data['date'], article_data['views'], alpha=0.3, color=colors[i])
        ax.plot(article_data['date'], article_data['views'], color=colors[i], linewidth=1.5)
        ax.axvline(spike['spike_date'], color=COLORS['highlight'], linestyle='--', alpha=0.7)

        title = spike['article'].replace('_', ' ')[:30]
        ax.set_title(f"{title}\n{spike['multiplier']:.0f}x spike on {spike['spike_date'].strftime('%Y-%m-%d')}",
                    fontsize=10, color=COLORS['text'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K' if x < 1e6 else f'{x/1e6:.1f}M'))
        ax.grid(True, alpha=0.3)

    plt.suptitle('Biggest Traffic Spikes', fontsize=14, fontweight='bold', color=COLORS['text'], y=1.02)
    plt.tight_layout()
    return fig_to_base64(fig)


def calculate_stats(df: pd.DataFrame, filtered_df: pd.DataFrame) -> dict:
    """Calculate summary statistics."""
    date_range = (df['date'].min(), df['date'].max())
    num_days = (date_range[1] - date_range[0]).days + 1

    return {
        'date_range': f"{date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}",
        'num_days': num_days,
        'total_views': filtered_df['views'].sum(),
        'unique_articles': filtered_df['article'].nunique(),
        'avg_daily_views': filtered_df.groupby('date')['views'].sum().mean(),
        'peak_day': filtered_df.groupby('date')['views'].sum().idxmax().strftime('%Y-%m-%d'),
        'peak_views': filtered_df.groupby('date')['views'].sum().max(),
    }


def compute_day_of_week_stats(df: pd.DataFrame) -> dict:
    """Compute weekday vs weekend traffic statistics for narrative generation."""
    daily_total = df.groupby('date')['views'].sum().reset_index()
    daily_total['dow'] = daily_total['date'].dt.day_name()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_avg = daily_total.groupby('dow')['views'].mean().reindex(dow_order)

    weekday_avg = dow_avg.iloc[:5].mean()
    weekend_avg = dow_avg.iloc[5:].mean()

    return {
        'weekday_avg': weekday_avg,
        'weekend_avg': weekend_avg,
        'weekend_diff_pct': ((weekend_avg - weekday_avg) / weekday_avg) * 100 if weekday_avg > 0 else 0,
    }


def generate_narrative(stats: dict, spike_df: pd.DataFrame,
                       top_articles: pd.DataFrame, consistency: pd.DataFrame,
                       dow_stats: dict) -> dict:
    """Generate data-driven narrative HTML paragraphs with causal explanations.

    Returns dict with keys:
        'overview_insight': After overview metrics, before daily traffic chart
        'traffic_pattern_insight': In the day-of-week section
        'spike_insight': In the spike detection section
    """
    narrative = {}

    # Section A — Overview insight with peak day context
    if spike_df is not None and not spike_df.empty:
        top_spike = spike_df.iloc[0]
        article_name = top_spike['article'].replace('_', ' ')
        narrative['overview_insight'] = (
            f'<p style="color: {COLORS["muted"]}; margin-bottom: 1rem;">'
            f'Peak traffic of {stats["peak_views"]:,} views occurred on {stats["peak_day"]}, '
            f'likely driven by {article_name} reaching {top_spike["multiplier"]:.0f}x its average. '
            f'This suggests a major news event or cultural moment triggered widespread interest.'
            f'</p>'
        )
    else:
        narrative['overview_insight'] = (
            f'<p style="color: {COLORS["muted"]}; margin-bottom: 1rem;">'
            f'Peak traffic of {stats["peak_views"]:,} views occurred on {stats["peak_day"]}. '
            f'No significant spikes were detected, which suggests steady baseline interest '
            f'rather than event-driven traffic during this period.'
            f'</p>'
        )

    # Section B — Day-of-week traffic pattern with causal language
    if dow_stats and not pd.isna(dow_stats.get('weekend_diff_pct', float('nan'))):
        diff = abs(dow_stats['weekend_diff_pct'])
        if dow_stats['weekend_diff_pct'] < 0:
            narrative['traffic_pattern_insight'] = (
                f'<p style="color: {COLORS["muted"]}; margin-bottom: 1rem;">'
                f'Weekday traffic averages {diff:.0f}% higher than weekends. '
                f'This is probably due to work and school-related browsing patterns, '
                f'as Wikipedia serves as a primary reference during professional hours.'
                f'</p>'
            )
        else:
            narrative['traffic_pattern_insight'] = (
                f'<p style="color: {COLORS["muted"]}; margin-bottom: 1rem;">'
                f'Weekend traffic averages {diff:.0f}% higher than weekdays, '
                f'likely driven by increased leisure browsing on Saturdays and Sundays.'
                f'</p>'
            )

    # Section C — Spike context with causal explanation
    if spike_df is not None and not spike_df.empty:
        top_spike = spike_df.iloc[0]
        article_name = top_spike['article'].replace('_', ' ')
        spike_date_str = (top_spike['spike_date'].strftime('%Y-%m-%d')
                          if hasattr(top_spike['spike_date'], 'strftime')
                          else str(top_spike['spike_date']))
        narrative['spike_insight'] = (
            f'<p style="color: {COLORS["muted"]}; margin-bottom: 1rem;">'
            f'The largest spike was {article_name} on {spike_date_str} '
            f'at {top_spike["spike_views"]:,} views ({top_spike["multiplier"]:.0f}x above average). '
            f'Because spikes of this magnitude typically correlate with breaking news coverage, '
            f'this likely indicates a major media event on or near that date.'
            f'</p>'
        )

    return narrative


def generate_html(stats: dict, plots: dict, top_articles: pd.DataFrame,
                  spike_df: pd.DataFrame, consistency: pd.DataFrame,
                  narrative: dict = None) -> str:
    """Generate the HTML report."""

    def format_number(n):
        if n >= 1e9:
            return f'{n/1e9:.1f}B'
        if n >= 1e6:
            return f'{n/1e6:.1f}M'
        if n >= 1e3:
            return f'{n/1e3:.1f}K'
        return f'{n:.0f}'

    def make_badge(text, color):
        return f'<span class="badge" style="background: {color};">{text}</span>'

    def wiki_link(article):
        display_name = article.replace('_', ' ')
        return f'<a href="https://en.wikipedia.org/wiki/{article}" target="_blank">{display_name}</a>'

    # Top articles table
    top_rows = ''
    for i, row in top_articles.head(10).iterrows():
        top_rows += f'''
        <tr>
            <td>{make_badge(row['rank'], COLORS['accent'])}</td>
            <td>{wiki_link(row['article'])}</td>
            <td>{format_number(row['views'])}</td>
        </tr>'''

    # Spikes table
    spike_rows = ''
    for i, row in spike_df.head(10).iterrows():
        spike_rows += f'''
        <tr>
            <td>{wiki_link(row['article'])}</td>
            <td>{row['spike_date'].strftime('%Y-%m-%d')}</td>
            <td>{format_number(row['spike_views'])}</td>
            <td>{make_badge(f"{row['multiplier']:.0f}x", COLORS['highlight'])}</td>
        </tr>'''

    # Consistency table (articles appearing most days)
    consistency_rows = ''
    for i, row in consistency.head(10).iterrows():
        pct = row['days_appeared'] / stats['num_days'] * 100
        consistency_rows += f'''
        <tr>
            <td>{wiki_link(row['article'])}</td>
            <td>{row['days_appeared']}</td>
            <td>{make_badge(f"{pct:.0f}%", COLORS['success'])}</td>
            <td>{format_number(row['total_views'])}</td>
        </tr>'''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wikipedia Pageviews Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: {COLORS['bg']};
            color: {COLORS['text']};
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid {COLORS['accent']};
        }}
        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, {COLORS['info']}, {COLORS['highlight']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .subtitle {{ color: {COLORS['muted']}; font-size: 1.1rem; }}
        .timestamp {{ color: {COLORS['muted']}; font-size: 0.85rem; margin-top: 0.5rem; }}

        .section {{
            background: {COLORS['card']};
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid {COLORS['info']};
        }}
        .section h2 {{
            font-size: 1.3rem;
            margin-bottom: 1rem;
            color: {COLORS['text']};
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .section.highlight {{ border-left-color: {COLORS['highlight']}; }}
        .section.success {{ border-left-color: {COLORS['success']}; }}
        .section.warning {{ border-left-color: {COLORS['warning']}; }}

        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        .metric {{
            background: {COLORS['bg']};
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: {COLORS['info']};
        }}
        .metric-label {{
            font-size: 0.85rem;
            color: {COLORS['muted']};
            margin-top: 0.25rem;
        }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            color: white;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid {COLORS['accent']};
        }}
        th {{
            color: {COLORS['muted']};
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
        }}
        tr:hover {{ background: {COLORS['bg']}; }}

        a {{ color: {COLORS['info']}; text-decoration: none; }}
        a:hover {{ color: {COLORS['highlight']}; text-decoration: underline; }}

        .chart {{
            margin: 1rem 0;
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            border-radius: 8px;
        }}

        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
        @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}

        footer {{
            text-align: center;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid {COLORS['accent']};
            color: {COLORS['muted']};
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Wikipedia Pageviews Report</h1>
            <p class="subtitle">{stats['date_range']}</p>
            <p class="timestamp">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <section class="section">
            <h2>Overview</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{format_number(stats['total_views'])}</div>
                    <div class="metric-label">Total Views</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{stats['num_days']}</div>
                    <div class="metric-label">Days Analyzed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{format_number(stats['unique_articles'])}</div>
                    <div class="metric-label">Unique Articles</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{format_number(stats['avg_daily_views'])}</div>
                    <div class="metric-label">Avg Daily Views</div>
                </div>
            </div>
            {(narrative or {}).get('overview_insight', '')}
            <div class="chart">
                <img src="data:image/png;base64,{plots['daily_traffic']}" alt="Daily Traffic">
            </div>
        </section>

        <div class="grid">
            <section class="section highlight">
                <h2>Top Articles</h2>
                <table>
                    <thead>
                        <tr><th>Rank</th><th>Article</th><th>Views</th></tr>
                    </thead>
                    <tbody>{top_rows}</tbody>
                </table>
            </section>

            <section class="section success">
                <h2>Most Consistent (Always Trending)</h2>
                <table>
                    <thead>
                        <tr><th>Article</th><th>Days</th><th>Coverage</th><th>Views</th></tr>
                    </thead>
                    <tbody>{consistency_rows}</tbody>
                </table>
            </section>
        </div>

        <section class="section">
            <h2>Top Articles Visualization</h2>
            <div class="chart">
                <img src="data:image/png;base64,{plots['top_articles']}" alt="Top Articles">
            </div>
        </section>

        <section class="section warning">
            <h2>Traffic Patterns</h2>
            <p style="color: {COLORS['muted']}; margin-bottom: 1rem;">
                Weekend traffic (highlighted in yellow) typically shows different patterns than weekdays.
            </p>
            {(narrative or {}).get('traffic_pattern_insight', '')}
            <div class="chart">
                <img src="data:image/png;base64,{plots['day_of_week']}" alt="Day of Week">
            </div>
        </section>

        <section class="section highlight">
            <h2>Biggest Traffic Spikes</h2>
            <p style="color: {COLORS['muted']}; margin-bottom: 1rem;">
                Articles with unusual view surges (3+ standard deviations above their average).
            </p>
            {(narrative or {}).get('spike_insight', '')}
            <table>
                <thead>
                    <tr><th>Article</th><th>Spike Date</th><th>Peak Views</th><th>Multiplier</th></tr>
                </thead>
                <tbody>{spike_rows}</tbody>
            </table>
        </section>

        {f'<section class="section"><h2>Spike Analysis</h2><div class="chart"><img src="data:image/png;base64,{plots["spikes"]}" alt="Spike Analysis"></div></section>' if plots.get('spikes') else ''}

        <footer>
            Wikipedia Pageviews Analytics &bull; Data from Wikimedia REST API
        </footer>
    </div>
</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(description='Generate Wikipedia pageviews HTML report')
    parser.add_argument('--days', '-d', type=int, default=30, help='Number of days to analyze (default: 30)')
    parser.add_argument('--all', '-a', action='store_true', help='Analyze all available data')
    parser.add_argument('--output', '-o', help='Output filename (default: auto-generated)')
    args = parser.parse_args()

    setup_plot_style()

    # Load data
    days = None if args.all else args.days
    print(f"Loading data{'' if args.all else f' (last {args.days} days)'}...")
    df = load_data(days)
    print(f"  Loaded {len(df):,} records from {df['date'].min().date()} to {df['date'].max().date()}")

    # Filter content
    filtered_df = filter_content(df)
    print(f"  After filtering: {len(filtered_df):,} article records")

    # Calculate stats
    stats = calculate_stats(df, filtered_df)

    # Generate plots
    print("Generating visualizations...")
    plots = {
        'daily_traffic': plot_daily_traffic(filtered_df),
        'top_articles': plot_top_articles(filtered_df),
        'day_of_week': plot_day_of_week(filtered_df),
    }

    # Detect spikes
    print("Analyzing spikes...")
    spike_df = detect_spikes(filtered_df)
    if len(spike_df) > 0:
        plots['spikes'] = plot_spike_examples(filtered_df, spike_df)
        print(f"  Found {len(spike_df)} articles with significant spikes")

    # Top articles
    top_articles = filtered_df.groupby('article').agg({
        'views': 'sum',
        'rank': 'min'
    }).reset_index().sort_values('views', ascending=False)
    top_articles['rank'] = range(1, len(top_articles) + 1)

    # Consistency analysis
    consistency = filtered_df.groupby('article').agg({
        'date': 'nunique',
        'views': 'sum'
    }).reset_index()
    consistency.columns = ['article', 'days_appeared', 'total_views']
    consistency = consistency.sort_values('days_appeared', ascending=False)

    # Generate narrative
    try:
        dow_stats = compute_day_of_week_stats(filtered_df)
    except (ValueError, ZeroDivisionError):
        dow_stats = {}
    narrative = generate_narrative(stats, spike_df, top_articles, consistency, dow_stats)

    # Generate HTML
    print("Generating report...")
    html = generate_html(stats, plots, top_articles, spike_df, consistency, narrative)

    # Save report
    REPORTS_DIR.mkdir(exist_ok=True)
    if args.output:
        output_file = REPORTS_DIR / args.output
    else:
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = REPORTS_DIR / f'report_{date_str}.html'

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"\nReport saved to: {output_file}")
    print(f"Open in browser: file://{output_file.absolute()}")


if __name__ == '__main__':
    main()
