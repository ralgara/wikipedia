#!/usr/bin/env python3
"""Deep analysis with advanced correlation detection and causal inference.

Usage:
    ./scripts/analyze-deep.py                    # Last 365 days
    ./scripts/analyze-deep.py --days 730         # Last 2 years
    ./scripts/analyze-deep.py --all              # All available data
"""

import argparse
import base64
import io
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import zscore, pearsonr
from scipy.ndimage import gaussian_filter1d
import pywt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.wikipedia.filters import is_content, should_flag_for_review

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

# Filtering is now done via shared.wikipedia.filters module


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
    """Filter out non-article content and flagged articles using shared filtering logic."""
    # Filter based on content type and flagged items
    mask = df['article'].apply(lambda x: is_content(x) and not should_flag_for_review(x))

    return df[mask].copy()


def analyze_date_coverage(df: pd.DataFrame) -> dict:
    """Analyze date range coverage and find missing dates."""
    dates_in_data = df['date'].dt.date.unique()
    dates_set = set(dates_in_data)

    first_date = min(dates_in_data)
    last_date = max(dates_in_data)

    expected_dates = pd.date_range(first_date, last_date, freq='D').date
    expected_count = len(expected_dates)
    actual_count = len(dates_set)

    missing_dates = sorted([d for d in expected_dates if d not in dates_set])

    missing_ranges = []
    if missing_dates:
        range_start = missing_dates[0]
        range_end = missing_dates[0]
        for i in range(1, len(missing_dates)):
            if (missing_dates[i] - range_end).days == 1:
                range_end = missing_dates[i]
            else:
                missing_ranges.append((range_start, range_end))
                range_start = missing_dates[i]
                range_end = missing_dates[i]
        missing_ranges.append((range_start, range_end))

    return {
        'first_date': first_date,
        'last_date': last_date,
        'expected_days': expected_count,
        'actual_days': actual_count,
        'missing_days': expected_count - actual_count,
        'density': actual_count / expected_count if expected_count > 0 else 0,
        'missing_ranges': missing_ranges,
    }


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
    })
    sns.set_palette([COLORS['highlight'], COLORS['info'], COLORS['success'],
                     COLORS['warning'], '#9b59b6', '#3498db'])


def get_article_timeseries(df: pd.DataFrame, min_days: int = 30) -> dict:
    """Extract time series for articles with sufficient data."""
    date_range = pd.date_range(df['date'].min(), df['date'].max())

    timeseries = {}
    for article, group in df.groupby('article'):
        if len(group) < min_days:
            continue
        series = group.set_index('date')['views'].reindex(date_range, fill_value=0)
        timeseries[article] = series

    return timeseries


# =============================================================================
# ADVANCED CORRELATION ANALYSIS
# =============================================================================

def get_article_words(article: str) -> set:
    """Extract meaningful words from article name."""
    # Replace underscores, split on non-alphanumeric
    words = re.split(r'[_\-\(\)\[\]\{\}:,\.]', article.lower())
    # Filter short words and common terms
    stopwords = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'is', 'was', 'are', 'were'}
    return {w for w in words if len(w) > 2 and w not in stopwords}


def compute_word_similarity(article1: str, article2: str) -> float:
    """Compute Jaccard similarity between article names."""
    words1 = get_article_words(article1)
    words2 = get_article_words(article2)

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def compute_lag_correlation(series1: np.ndarray, series2: np.ndarray, max_lag: int = 7) -> dict:
    """Compute correlation at different lags to detect lead/lag relationships."""
    best_corr = 0
    best_lag = 0

    s1 = zscore(series1)
    s2 = zscore(series2)

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = np.corrcoef(s1[:lag], s2[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(s1[lag:], s2[:-lag])[0, 1]
        else:
            corr = np.corrcoef(s1, s2)[0, 1]

        if not np.isnan(corr) and abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag

    return {'correlation': best_corr, 'lag': best_lag}


def find_spike_overlap(series1: np.ndarray, series2: np.ndarray, threshold_std: float = 2.0) -> dict:
    """Find how often two articles spike together."""
    z1 = zscore(series1)
    z2 = zscore(series2)

    spikes1 = z1 > threshold_std
    spikes2 = z2 > threshold_std

    both_spike = np.sum(spikes1 & spikes2)
    either_spike = np.sum(spikes1 | spikes2)

    # Jaccard index of spike overlap
    overlap = both_spike / either_spike if either_spike > 0 else 0

    return {
        'overlap': overlap,
        'both_spike_days': int(both_spike),
        'either_spike_days': int(either_spike)
    }


def infer_relationship(article1: str, article2: str, lag: int,
                       word_similarity: float, spike_overlap: float) -> dict:
    """Infer the likely relationship between two correlated articles."""

    # Determine relationship type
    if word_similarity > 0.3:
        relationship_type = "obvious"
        explanation = "Names share common words - likely directly related (e.g., actor/show, person/organization)"
    elif word_similarity > 0.1:
        relationship_type = "semi-obvious"
        explanation = "Some name overlap - possibly related topics or franchises"
    else:
        relationship_type = "interesting"
        explanation = "No obvious name connection - may reveal hidden cultural link or shared external cause"

    # Determine causality direction
    if abs(lag) <= 1:
        causality = "simultaneous"
        causality_explanation = "Spike together - likely shared external cause (news event, cultural moment)"
    elif lag > 1:
        causality = f"{article1} leads"
        causality_explanation = f"Interest in {article1.replace('_', ' ')} precedes {article2.replace('_', ' ')} by ~{lag} days"
    else:
        causality = f"{article2} leads"
        causality_explanation = f"Interest in {article2.replace('_', ' ')} precedes {article1.replace('_', ' ')} by ~{abs(lag)} days"

    # Generate hypothesis
    if relationship_type == "interesting":
        if spike_overlap > 0.3:
            hypothesis = "Strong co-occurrence suggests shared cultural trigger or hidden thematic connection"
        else:
            hypothesis = "Occasional co-occurrence - may share audience demographic or appear in same news cycles"
    elif relationship_type == "obvious":
        hypothesis = "Direct relationship - one topic naturally leads to the other"
    else:
        hypothesis = "Moderate connection - possibly same franchise, era, or field"

    return {
        'type': relationship_type,
        'type_explanation': explanation,
        'causality': causality,
        'causality_explanation': causality_explanation,
        'hypothesis': hypothesis,
        'interest_score': (1 - word_similarity) * (0.5 + spike_overlap)  # Higher = more interesting
    }


def find_interesting_correlations(timeseries: dict, top_n: int = 100,
                                   min_correlation: float = 0.4) -> list:
    """Find correlated pairs with relationship analysis."""

    # Get articles with highest total views for analysis
    totals = {a: s.sum() for a, s in timeseries.items()}
    top_articles = sorted(totals.keys(), key=lambda x: totals[x], reverse=True)[:top_n]

    pairs = []
    n = len(top_articles)

    for i in range(n):
        for j in range(i + 1, n):
            a1, a2 = top_articles[i], top_articles[j]
            s1 = timeseries[a1].values
            s2 = timeseries[a2].values

            # Basic correlation
            corr = np.corrcoef(zscore(s1), zscore(s2))[0, 1]

            if np.isnan(corr) or corr < min_correlation:
                continue

            # Detailed analysis
            word_sim = compute_word_similarity(a1, a2)
            lag_info = compute_lag_correlation(s1, s2)
            spike_info = find_spike_overlap(s1, s2)
            relationship = infer_relationship(a1, a2, lag_info['lag'],
                                             word_sim, spike_info['overlap'])

            pairs.append({
                'article1': a1,
                'article2': a2,
                'correlation': corr,
                'word_similarity': word_sim,
                'lag': lag_info['lag'],
                'spike_overlap': spike_info['overlap'],
                'both_spike_days': spike_info['both_spike_days'],
                **relationship
            })

    # Sort by interest score (prefer interesting non-obvious relationships)
    pairs.sort(key=lambda x: (x['type'] == 'interesting', x['interest_score'], x['correlation']),
               reverse=True)

    return pairs


def plot_correlation_pair(timeseries: dict, pair: dict) -> str:
    """Plot detailed view of a correlated pair."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    s1 = timeseries[pair['article1']]
    s2 = timeseries[pair['article2']]

    # Normalize for comparison
    s1_norm = (s1 - s1.mean()) / s1.std()
    s2_norm = (s2 - s2.mean()) / s2.std()

    # Time series comparison
    ax1 = axes[0]
    ax1.plot(s1_norm.index, s1_norm.values, color=COLORS['highlight'],
             linewidth=1, alpha=0.8, label=pair['article1'].replace('_', ' ')[:30])
    ax1.plot(s2_norm.index, s2_norm.values, color=COLORS['info'],
             linewidth=1, alpha=0.8, label=pair['article2'].replace('_', ' ')[:30])
    ax1.set_ylabel('Normalized Views')
    ax1.set_title(f"Correlation: {pair['correlation']:.2f} | Lag: {pair['lag']} days | Type: {pair['type'].upper()}",
                  fontsize=11, color=COLORS['text'])
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(s1.values, s2.values, alpha=0.3, s=10, color=COLORS['success'])
    ax2.set_xlabel(pair['article1'].replace('_', ' ')[:30])
    ax2.set_ylabel(pair['article2'].replace('_', ' ')[:30])
    ax2.set_title(f"Spike Overlap: {pair['spike_overlap']:.1%} ({pair['both_spike_days']} days)",
                  fontsize=10, color=COLORS['text'])
    ax2.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(s1.values, s2.values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(s1.min(), s1.max(), 100)
    ax2.plot(x_line, p(x_line), color=COLORS['highlight'], linewidth=2, alpha=0.7)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_top_pairs_grid(timeseries: dict, pairs: list, n: int = 6) -> str:
    """Plot grid of top correlated pairs."""
    n = min(n, len(pairs))
    if n == 0:
        return None

    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n))
    if n == 1:
        axes = [axes]

    colors = [(COLORS['highlight'], COLORS['info']),
              (COLORS['success'], COLORS['warning']),
              ('#9b59b6', '#3498db')] * 3

    for i, pair in enumerate(pairs[:n]):
        ax = axes[i]

        s1 = timeseries[pair['article1']]
        s2 = timeseries[pair['article2']]

        s1_norm = (s1 - s1.mean()) / s1.std()
        s2_norm = (s2 - s2.mean()) / s2.std()

        ax.plot(s1_norm.index, s1_norm.values, color=colors[i % len(colors)][0],
                linewidth=1, alpha=0.8, label=pair['article1'].replace('_', ' ')[:25])
        ax.plot(s2_norm.index, s2_norm.values, color=colors[i % len(colors)][1],
                linewidth=1, alpha=0.8, label=pair['article2'].replace('_', ' ')[:25])

        type_badge = f"[{pair['type'].upper()}]"
        ax.set_title(f"{type_badge} r={pair['correlation']:.2f}, lag={pair['lag']}d - {pair['causality']}",
                     fontsize=10, color=COLORS['text'], loc='left')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylabel('Normalized')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Top Correlated Pairs by Interest Score', fontsize=14, fontweight='bold',
                 color=COLORS['text'], y=1.01)
    plt.tight_layout()
    return fig_to_base64(fig)


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_html(stats: dict, plots: dict, pairs: list) -> str:
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

    # Missing dates summary
    missing_ranges_html = ''
    if stats.get('missing_ranges'):
        for start, end in stats['missing_ranges'][:20]:
            if start == end:
                missing_ranges_html += f'<span class="badge" style="background: {COLORS["accent"]}; margin: 2px;">{start}</span> '
            else:
                days_in_range = (end - start).days + 1
                missing_ranges_html += f'<span class="badge" style="background: {COLORS["accent"]}; margin: 2px;">{start} to {end} ({days_in_range}d)</span> '
        if len(stats['missing_ranges']) > 20:
            missing_ranges_html += f'<span class="badge" style="background: {COLORS["muted"]};">+{len(stats["missing_ranges"]) - 20} more</span>'

    # Correlation pairs - separated by type
    interesting_pairs = [p for p in pairs if p['type'] == 'interesting'][:10]
    obvious_pairs = [p for p in pairs if p['type'] in ('obvious', 'semi-obvious')][:10]

    def make_pair_card(pair, index):
        type_color = COLORS['success'] if pair['type'] == 'interesting' else COLORS['warning'] if pair['type'] == 'semi-obvious' else COLORS['muted']
        causality_icon = "⟷" if pair['causality'] == 'simultaneous' else "→" if 'leads' in pair['causality'] else "←"

        return f'''
        <div class="pair-card">
            <div class="pair-header">
                <span class="pair-number">{index + 1}</span>
                {make_badge(pair['type'].upper(), type_color)}
                {make_badge(f"r={pair['correlation']:.2f}", COLORS['info'])}
            </div>
            <div class="pair-articles">
                {wiki_link(pair['article1'])}
                <span class="causality-arrow">{causality_icon}</span>
                {wiki_link(pair['article2'])}
            </div>
            <div class="pair-stats">
                <span>Lag: {pair['lag']}d</span>
                <span>Spike overlap: {pair['spike_overlap']:.0%}</span>
                <span>Word sim: {pair['word_similarity']:.0%}</span>
            </div>
            <div class="pair-analysis">
                <p><strong>Causality:</strong> {pair['causality_explanation']}</p>
                <p><strong>Hypothesis:</strong> {pair['hypothesis']}</p>
            </div>
        </div>'''

    interesting_cards = ''.join(make_pair_card(p, i) for i, p in enumerate(interesting_pairs))
    obvious_cards = ''.join(make_pair_card(p, i) for i, p in enumerate(obvious_pairs))

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wikipedia Deep Analysis</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: {COLORS['bg']};
            color: {COLORS['text']};
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
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
            background: linear-gradient(90deg, {COLORS['success']}, {COLORS['info']});
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
            font-size: 1.4rem;
            margin-bottom: 1rem;
            color: {COLORS['text']};
        }}
        .section h3 {{
            font-size: 1.1rem;
            margin: 1.5rem 0 1rem 0;
            color: {COLORS['info']};
        }}
        .section p {{ color: {COLORS['muted']}; margin-bottom: 1rem; }}
        .section.highlight {{ border-left-color: {COLORS['highlight']}; }}
        .section.success {{ border-left-color: {COLORS['success']}; }}
        .section.warning {{ border-left-color: {COLORS['warning']}; }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            color: white;
            margin-right: 0.5rem;
        }}

        .metric {{
            background: {COLORS['bg']};
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 1.3rem;
            font-weight: 700;
            color: {COLORS['info']};
        }}
        .metric-label {{
            font-size: 0.8rem;
            color: {COLORS['muted']};
            margin-top: 0.25rem;
        }}

        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }}
        @media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(2, 1fr); }} }}

        .chart {{
            margin: 1.5rem 0;
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            border-radius: 8px;
        }}

        a {{ color: {COLORS['info']}; text-decoration: none; }}
        a:hover {{ color: {COLORS['highlight']}; text-decoration: underline; }}

        .pair-card {{
            background: {COLORS['bg']};
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 3px solid {COLORS['accent']};
        }}
        .pair-card:hover {{
            border-left-color: {COLORS['info']};
        }}
        .pair-header {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }}
        .pair-number {{
            background: {COLORS['accent']};
            color: {COLORS['text']};
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        .pair-articles {{
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }}
        .causality-arrow {{
            color: {COLORS['warning']};
            font-size: 1.2rem;
            margin: 0 0.5rem;
        }}
        .pair-stats {{
            display: flex;
            gap: 1.5rem;
            font-size: 0.85rem;
            color: {COLORS['muted']};
            margin-bottom: 0.75rem;
        }}
        .pair-analysis {{
            font-size: 0.9rem;
            padding-top: 0.75rem;
            border-top: 1px solid {COLORS['accent']};
        }}
        .pair-analysis p {{
            margin-bottom: 0.5rem;
            color: {COLORS['text']};
        }}
        .pair-analysis strong {{
            color: {COLORS['info']};
        }}

        .pairs-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }}
        @media (max-width: 1000px) {{ .pairs-grid {{ grid-template-columns: 1fr; }} }}

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
            <h1>Wikipedia Deep Analysis</h1>
            <p class="subtitle">Advanced Correlation Detection & Causal Inference</p>
            <p class="timestamp">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <section class="section">
            <h2>Data Coverage</h2>
            <div class="grid">
                <div class="metric">
                    <div class="metric-value">{stats['first_date']}</div>
                    <div class="metric-label">First Date</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{stats['last_date']}</div>
                    <div class="metric-label">Last Date</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{stats['actual_days']:,} / {stats['expected_days']:,}</div>
                    <div class="metric-label">Days (Actual / Expected)</div>
                </div>
                <div class="metric">
                    <div class="metric-value" style="color: {COLORS['success'] if stats['density'] > 0.95 else COLORS['warning']};">{stats['density']*100:.1f}%</div>
                    <div class="metric-label">Coverage Density</div>
                </div>
            </div>
            {f'<p style="margin-top: 1rem;"><strong>Missing dates ({stats["missing_days"]}):</strong></p><p style="line-height: 2;">{missing_ranges_html}</p>' if stats['missing_days'] > 0 else ''}
        </section>

        <section class="section">
            <h2>Analysis Summary</h2>
            <div class="grid">
                <div class="metric">
                    <div class="metric-value">{stats['articles_analyzed']:,}</div>
                    <div class="metric-label">Articles Analyzed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{stats['total_pairs']:,}</div>
                    <div class="metric-label">Correlated Pairs (r>{stats['min_correlation']})</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{stats['interesting_pairs']}</div>
                    <div class="metric-label">Interesting (Non-Obvious)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{stats['nsfw_filtered']}</div>
                    <div class="metric-label">NSFW Filtered</div>
                </div>
            </div>
        </section>

        {f'<section class="section"><h2>Top Correlated Pairs</h2><div class="chart"><img src="data:image/png;base64,{plots["top_pairs"]}" alt="Top Pairs"></div></section>' if plots.get('top_pairs') else ''}

        <section class="section success">
            <h2>Interesting Correlations</h2>
            <p>Non-obvious relationships - articles with no apparent name connection that spike together.
               These may reveal hidden cultural links, shared audiences, or common external triggers.</p>

            {interesting_cards if interesting_cards else '<p>No interesting correlations found with current threshold.</p>'}
        </section>

        <section class="section warning">
            <h2>Expected Correlations</h2>
            <p>Obvious or semi-obvious relationships - articles that share naming elements or are directly related
               (actors/shows, people/organizations, sequels/franchises).</p>

            {obvious_cards if obvious_cards else '<p>No obvious correlations found.</p>'}
        </section>

        <section class="section">
            <h2>Methodology</h2>
            <p><strong>Correlation:</strong> Pearson correlation coefficient between normalized daily view counts.</p>
            <p><strong>Lag Analysis:</strong> Cross-correlation at ±7 day offsets to detect lead/lag relationships.</p>
            <p><strong>Word Similarity:</strong> Jaccard index of words in article names (high = obvious relationship).</p>
            <p><strong>Spike Overlap:</strong> Jaccard index of days both articles exceeded 2σ above their mean.</p>
            <p><strong>Interest Score:</strong> (1 - word_similarity) × (0.5 + spike_overlap) — prioritizes non-obvious, co-spiking pairs.</p>
            <p><strong>NSFW Filter:</strong> Articles containing explicit terms are excluded from analysis.</p>
        </section>

        <footer>
            Wikipedia Deep Analysis &bull; Advanced correlation detection with causal inference
        </footer>
    </div>
</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(description='Deep Wikipedia correlation analysis')
    parser.add_argument('--days', '-d', type=int, default=365, help='Days to analyze (default: 365)')
    parser.add_argument('--all', '-a', action='store_true', help='Analyze all available data')
    parser.add_argument('--output', '-o', help='Output filename')
    parser.add_argument('--top', '-t', type=int, default=100, help='Top N articles to analyze (default: 100)')
    parser.add_argument('--min-corr', type=float, default=0.4, help='Minimum correlation threshold (default: 0.4)')
    args = parser.parse_args()

    setup_plot_style()

    # Load data
    days = None if args.all else args.days
    print(f"Loading data{'' if args.all else f' (last {args.days} days)'}...")
    df = load_data(days)

    # Count NSFW before filtering
    nsfw_count = df['article'].apply(is_nsfw).sum()

    filtered_df = filter_content(df)
    print(f"  Loaded {len(filtered_df):,} records (filtered {nsfw_count:,} NSFW)")

    # Analyze date coverage
    print("Analyzing date coverage...")
    coverage = analyze_date_coverage(df)
    print(f"  Date range: {coverage['first_date']} to {coverage['last_date']}")
    print(f"  Coverage: {coverage['actual_days']}/{coverage['expected_days']} days ({coverage['density']*100:.1f}%)")

    # Get time series
    print("Building time series...")
    timeseries = get_article_timeseries(filtered_df, min_days=30)
    print(f"  {len(timeseries)} articles with 30+ days of data")

    # Find correlations
    print(f"Finding correlations (top {args.top} articles, r>{args.min_corr})...")
    pairs = find_interesting_correlations(timeseries, top_n=args.top, min_correlation=args.min_corr)

    interesting_count = sum(1 for p in pairs if p['type'] == 'interesting')
    print(f"  Found {len(pairs)} correlated pairs ({interesting_count} interesting)")

    stats = {
        'first_date': coverage['first_date'],
        'last_date': coverage['last_date'],
        'expected_days': coverage['expected_days'],
        'actual_days': coverage['actual_days'],
        'missing_days': coverage['missing_days'],
        'density': coverage['density'],
        'missing_ranges': coverage['missing_ranges'],
        'articles_analyzed': len(timeseries),
        'total_pairs': len(pairs),
        'interesting_pairs': interesting_count,
        'nsfw_filtered': nsfw_count,
        'min_correlation': args.min_corr,
    }

    # Generate plots
    plots = {}
    print("Generating visualizations...")

    if pairs:
        plots['top_pairs'] = plot_top_pairs_grid(timeseries, pairs[:6])

    # Generate HTML
    print("Generating report...")
    html = generate_html(stats, plots, pairs)

    # Save
    REPORTS_DIR.mkdir(exist_ok=True)
    if args.output:
        output_file = REPORTS_DIR / args.output
    else:
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = REPORTS_DIR / f'deep_analysis_{date_str}.html'

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"\nReport saved to: {output_file}")
    print(f"Open in browser: file://{output_file.absolute()}")


if __name__ == '__main__':
    main()
