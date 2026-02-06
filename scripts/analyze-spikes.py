#!/usr/bin/env python3
"""Advanced spike analysis with periodicity detection, shape metrics, and wavelets.

Usage:
    ./scripts/analyze-spikes.py                    # Last 365 days
    ./scripts/analyze-spikes.py --days 730         # Last 2 years
    ./scripts/analyze-spikes.py --all              # All available data
"""

import argparse
import base64
import io
import json
import os
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
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
import pywt

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

# Filtering is now done via shared.wikipedia.filters.is_content()


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


def analyze_date_coverage(df: pd.DataFrame) -> dict:
    """Analyze date range coverage and find missing dates."""
    dates_in_data = df['date'].dt.date.unique()
    dates_set = set(dates_in_data)

    first_date = min(dates_in_data)
    last_date = max(dates_in_data)

    # Generate all expected dates in range
    expected_dates = pd.date_range(first_date, last_date, freq='D').date
    expected_count = len(expected_dates)
    actual_count = len(dates_set)

    # Find missing dates
    missing_dates = sorted([d for d in expected_dates if d not in dates_set])

    # Group missing dates into ranges for cleaner display
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
        'missing_dates': missing_dates,
        'missing_ranges': missing_ranges,
    }


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


def get_article_timeseries(df: pd.DataFrame, min_days: int = 30) -> dict:
    """Extract time series for articles with sufficient data."""
    date_range = pd.date_range(df['date'].min(), df['date'].max())

    timeseries = {}
    for article, group in df.groupby('article'):
        if len(group) < min_days:
            continue

        # Create full date range with zeros for missing days
        series = group.set_index('date')['views'].reindex(date_range, fill_value=0)
        timeseries[article] = series

    return timeseries


# =============================================================================
# 1. AUTOCORRELATION ANALYSIS - Find periodic patterns
# =============================================================================

def compute_acf(series: np.ndarray, max_lag: int = 400) -> np.ndarray:
    """Compute autocorrelation function."""
    n = len(series)
    max_lag = min(max_lag, n - 1)

    # Normalize
    series = series - np.mean(series)

    # Use FFT for efficiency
    fft = np.fft.fft(series, n=2*n)
    acf = np.fft.ifft(fft * np.conj(fft))[:max_lag].real
    acf = acf / acf[0]  # Normalize

    return acf


def find_periodic_articles(timeseries: dict, min_peak_height: float = 0.3) -> list:
    """Find articles with strong periodic patterns."""
    periodic = []

    for article, series in timeseries.items():
        if len(series) < 100:
            continue

        acf = compute_acf(series.values, max_lag=min(400, len(series) - 1))

        # Find peaks in ACF (excluding lag 0)
        peaks, properties = signal.find_peaks(acf[1:], height=min_peak_height, distance=5)
        peaks = peaks + 1  # Adjust for skipping lag 0

        if len(peaks) > 0:
            # Check for yearly pattern (around day 365)
            yearly_peaks = peaks[(peaks > 350) & (peaks < 380)]
            weekly_peaks = peaks[(peaks > 5) & (peaks < 9)]

            best_peak_idx = np.argmax(properties['peak_heights'])
            best_lag = peaks[best_peak_idx]
            best_height = properties['peak_heights'][best_peak_idx]

            period_type = 'unknown'
            if len(yearly_peaks) > 0 and acf[yearly_peaks[0]] > 0.2:
                period_type = 'yearly'
            elif len(weekly_peaks) > 0 and acf[weekly_peaks[0]] > 0.2:
                period_type = 'weekly'
            elif best_lag > 25 and best_lag < 35:
                period_type = 'monthly'

            periodic.append({
                'article': article,
                'best_lag': best_lag,
                'acf_strength': best_height,
                'period_type': period_type,
                'acf': acf,
                'series': series
            })

    # Sort by ACF strength
    periodic.sort(key=lambda x: x['acf_strength'], reverse=True)
    return periodic


def plot_periodic_articles(periodic: list, n: int = 6) -> str:
    """Plot ACF and time series for top periodic articles."""
    if len(periodic) == 0:
        return None

    top = periodic[:n]
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n))

    for i, item in enumerate(top):
        # Time series
        ax1 = axes[i, 0]
        dates = item['series'].index
        values = item['series'].values
        ax1.fill_between(dates, values, alpha=0.3, color=COLORS['info'])
        ax1.plot(dates, values, color=COLORS['info'], linewidth=0.8)

        title = item['article'].replace('_', ' ')[:35]
        ax1.set_title(f"{title} ({item['period_type']})", fontsize=10, color=COLORS['text'])
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K' if x < 1e6 else f'{x/1e6:.1f}M'))
        ax1.grid(True, alpha=0.3)

        # ACF
        ax2 = axes[i, 1]
        lags = np.arange(len(item['acf']))
        ax2.bar(lags, item['acf'], width=1, color=COLORS['highlight'], alpha=0.7)
        ax2.axhline(y=0, color=COLORS['muted'], linewidth=0.5)
        ax2.axhline(y=0.2, color=COLORS['success'], linewidth=1, linestyle='--', alpha=0.5)
        ax2.axvline(x=7, color=COLORS['warning'], linewidth=1, linestyle=':', alpha=0.5, label='7 days')
        ax2.axvline(x=365, color=COLORS['info'], linewidth=1, linestyle=':', alpha=0.5, label='365 days')
        ax2.set_xlabel('Lag (days)')
        ax2.set_ylabel('ACF')
        ax2.set_title(f"Autocorrelation (peak at {item['best_lag']} days)", fontsize=10, color=COLORS['text'])
        ax2.set_xlim(0, min(400, len(item['acf'])))
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.suptitle('Articles with Periodic Patterns', fontsize=14, fontweight='bold',
                 color=COLORS['text'], y=1.01)
    plt.tight_layout()
    return fig_to_base64(fig)


# =============================================================================
# 2. SPIKE SHAPE ANALYSIS - Rise time, decay time, area
# =============================================================================

def detect_spikes_advanced(series: np.ndarray, threshold_std: float = 3.0) -> list:
    """Detect spikes and measure their characteristics."""
    if len(series) < 10:
        return []

    # Smooth for baseline
    baseline = gaussian_filter1d(series.astype(float), sigma=7)
    residual = series - baseline

    # Find spike points
    threshold = threshold_std * np.std(residual)
    spike_mask = residual > threshold

    # Group consecutive spike days
    spikes = []
    in_spike = False
    spike_start = 0

    for i, is_spike in enumerate(spike_mask):
        if is_spike and not in_spike:
            spike_start = i
            in_spike = True
        elif not is_spike and in_spike:
            spikes.append((spike_start, i - 1))
            in_spike = False

    if in_spike:
        spikes.append((spike_start, len(series) - 1))

    # Analyze each spike
    spike_info = []
    for start, end in spikes:
        # Extend to find rise and decay
        rise_start = max(0, start - 5)
        decay_end = min(len(series) - 1, end + 10)

        peak_idx = start + np.argmax(series[start:end + 1])
        peak_value = series[peak_idx]

        # Rise time (days from 20% to peak)
        rise_threshold = baseline[peak_idx] + 0.2 * (peak_value - baseline[peak_idx])
        rise_days = 0
        for j in range(peak_idx, rise_start - 1, -1):
            if series[j] < rise_threshold:
                rise_days = peak_idx - j
                break

        # Decay time (days from peak to 20%)
        decay_days = 0
        for j in range(peak_idx, decay_end + 1):
            if series[j] < rise_threshold:
                decay_days = j - peak_idx
                break

        # Area (excess views above baseline)
        area = np.sum(series[rise_start:decay_end + 1] - baseline[rise_start:decay_end + 1])
        area = max(0, area)

        spike_info.append({
            'peak_idx': peak_idx,
            'peak_value': peak_value,
            'baseline': baseline[peak_idx],
            'rise_days': rise_days,
            'decay_days': decay_days,
            'duration': end - start + 1,
            'area': area,
            'multiplier': peak_value / max(baseline[peak_idx], 1)
        })

    return spike_info


def analyze_spike_shapes(timeseries: dict, min_spikes: int = 1) -> pd.DataFrame:
    """Analyze spike shapes for all articles."""
    results = []

    for article, series in timeseries.items():
        spikes = detect_spikes_advanced(series.values)

        if len(spikes) >= min_spikes:
            for spike in spikes:
                results.append({
                    'article': article,
                    'peak_date': series.index[spike['peak_idx']],
                    **spike
                })

    return pd.DataFrame(results)


def classify_spike_types(spike_df: pd.DataFrame) -> pd.DataFrame:
    """Classify spikes into types based on shape."""
    if len(spike_df) == 0:
        return spike_df

    df = spike_df.copy()

    def classify(row):
        ratio = row['decay_days'] / max(row['rise_days'], 0.5)

        if row['rise_days'] <= 1 and row['decay_days'] <= 2:
            return 'flash'  # Very brief spike
        elif row['rise_days'] <= 1 and ratio > 2:
            return 'breaking_news'  # Sharp rise, slow decay
        elif row['rise_days'] > 3 and row['decay_days'] > 3:
            return 'sustained'  # Gradual rise and fall
        elif ratio < 0.5:
            return 'anticipation'  # Slow build, quick drop
        else:
            return 'standard'

    df['spike_type'] = df.apply(classify, axis=1)
    return df


def plot_spike_shape_distribution(spike_df: pd.DataFrame) -> str:
    """Plot distribution of spike characteristics."""
    if len(spike_df) == 0:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Rise vs Decay scatter
    ax1 = axes[0, 0]
    colors = [COLORS['highlight'], COLORS['info'], COLORS['success'], COLORS['warning'], COLORS['muted']]
    types = spike_df['spike_type'].unique()
    for i, stype in enumerate(types):
        subset = spike_df[spike_df['spike_type'] == stype]
        ax1.scatter(subset['rise_days'], subset['decay_days'],
                   alpha=0.6, s=30, label=stype, color=colors[i % len(colors)])
    ax1.set_xlabel('Rise Time (days)')
    ax1.set_ylabel('Decay Time (days)')
    ax1.set_title('Spike Shape: Rise vs Decay', fontsize=12, color=COLORS['text'])
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, min(30, spike_df['rise_days'].quantile(0.95)))
    ax1.set_ylim(0, min(30, spike_df['decay_days'].quantile(0.95)))

    # Spike type distribution
    ax2 = axes[0, 1]
    type_counts = spike_df['spike_type'].value_counts()
    bars = ax2.bar(type_counts.index, type_counts.values, color=colors[:len(type_counts)])
    ax2.set_ylabel('Count')
    ax2.set_title('Spike Type Distribution', fontsize=12, color=COLORS['text'])
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Multiplier distribution
    ax3 = axes[1, 0]
    multipliers = spike_df['multiplier'].clip(upper=50)
    ax3.hist(multipliers, bins=50, color=COLORS['highlight'], alpha=0.7, edgecolor='none')
    ax3.set_xlabel('Peak / Baseline Ratio')
    ax3.set_ylabel('Count')
    ax3.set_title('Spike Intensity Distribution', fontsize=12, color=COLORS['text'])
    ax3.axvline(x=multipliers.median(), color=COLORS['success'], linestyle='--',
                label=f'Median: {multipliers.median():.1f}x')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Duration distribution
    ax4 = axes[1, 1]
    durations = spike_df['duration'].clip(upper=20)
    ax4.hist(durations, bins=20, color=COLORS['info'], alpha=0.7, edgecolor='none')
    ax4.set_xlabel('Duration (days)')
    ax4.set_ylabel('Count')
    ax4.set_title('Spike Duration Distribution', fontsize=12, color=COLORS['text'])
    ax4.axvline(x=durations.median(), color=COLORS['success'], linestyle='--',
                label=f'Median: {durations.median():.0f} days')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Spike Shape Analysis', fontsize=14, fontweight='bold', color=COLORS['text'])
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_spike_examples_by_type(timeseries: dict, spike_df: pd.DataFrame) -> str:
    """Plot example spikes for each type."""
    if len(spike_df) == 0:
        return None

    types = ['breaking_news', 'sustained', 'anticipation', 'flash']
    types = [t for t in types if t in spike_df['spike_type'].values]

    if len(types) == 0:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    colors = {'breaking_news': COLORS['highlight'], 'sustained': COLORS['info'],
              'anticipation': COLORS['success'], 'flash': COLORS['warning'], 'standard': COLORS['muted']}

    for i, stype in enumerate(types[:4]):
        ax = axes[i]

        # Get best example
        subset = spike_df[spike_df['spike_type'] == stype].nlargest(1, 'multiplier')
        if len(subset) == 0:
            continue

        row = subset.iloc[0]
        series = timeseries[row['article']]

        # Plot window around spike
        peak_idx = row['peak_idx']
        start = max(0, peak_idx - 20)
        end = min(len(series), peak_idx + 30)

        window = series.iloc[start:end]
        ax.fill_between(window.index, window.values, alpha=0.3, color=colors.get(stype, COLORS['muted']))
        ax.plot(window.index, window.values, color=colors.get(stype, COLORS['muted']), linewidth=1.5)
        ax.axvline(x=row['peak_date'], color=COLORS['highlight'], linestyle='--', alpha=0.5)

        title = row['article'].replace('_', ' ')[:25]
        ax.set_title(f"{stype.replace('_', ' ').title()}: {title}\n"
                    f"Rise: {row['rise_days']:.0f}d, Decay: {row['decay_days']:.0f}d, {row['multiplier']:.0f}x",
                    fontsize=10, color=COLORS['text'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K' if x < 1e6 else f'{x/1e6:.1f}M'))
        ax.grid(True, alpha=0.3)

    plt.suptitle('Spike Types - Representative Examples', fontsize=14, fontweight='bold', color=COLORS['text'])
    plt.tight_layout()
    return fig_to_base64(fig)


# =============================================================================
# 3. CROSS-CORRELATION - Find co-spiking articles
# =============================================================================

def compute_cross_correlation_matrix(timeseries: dict, top_n: int = 50) -> tuple:
    """Compute cross-correlation matrix for top articles."""
    # Get articles with highest total views
    totals = {a: s.sum() for a, s in timeseries.items()}
    top_articles = sorted(totals.keys(), key=lambda x: totals[x], reverse=True)[:top_n]

    # Build matrix
    n = len(top_articles)
    corr_matrix = np.zeros((n, n))

    for i, a1 in enumerate(top_articles):
        for j, a2 in enumerate(top_articles):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif j > i:
                # Normalize and compute correlation
                s1 = zscore(timeseries[a1].values)
                s2 = zscore(timeseries[a2].values)
                corr = np.corrcoef(s1, s2)[0, 1]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

    return corr_matrix, top_articles


def find_correlated_pairs(corr_matrix: np.ndarray, articles: list,
                          threshold: float = 0.5) -> list:
    """Find highly correlated article pairs."""
    pairs = []
    n = len(articles)

    for i in range(n):
        for j in range(i + 1, n):
            if corr_matrix[i, j] >= threshold:
                pairs.append({
                    'article1': articles[i],
                    'article2': articles[j],
                    'correlation': corr_matrix[i, j]
                })

    pairs.sort(key=lambda x: x['correlation'], reverse=True)
    return pairs


def plot_correlation_matrix(corr_matrix: np.ndarray, articles: list) -> str:
    """Plot correlation heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Shorten labels
    labels = [a.replace('_', ' ')[:20] for a in articles]

    # Custom colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(corr_matrix, xticklabels=labels, yticklabels=labels,
                cmap=cmap, center=0, vmin=-0.5, vmax=1,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.5},
                ax=ax)

    ax.set_title('Article Cross-Correlation Matrix', fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=20)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_correlated_pairs(timeseries: dict, pairs: list, n: int = 4) -> str:
    """Plot time series of correlated pairs."""
    if len(pairs) == 0:
        return None

    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]

    colors = [(COLORS['highlight'], COLORS['info']),
              (COLORS['success'], COLORS['warning']),
              ('#9b59b6', '#3498db'),
              (COLORS['highlight'], COLORS['success'])]

    for i, pair in enumerate(pairs[:n]):
        ax = axes[i]

        s1 = timeseries[pair['article1']]
        s2 = timeseries[pair['article2']]

        # Normalize for comparison
        s1_norm = (s1 - s1.mean()) / s1.std()
        s2_norm = (s2 - s2.mean()) / s2.std()

        ax.plot(s1_norm.index, s1_norm.values, color=colors[i % len(colors)][0],
                linewidth=1, alpha=0.8, label=pair['article1'].replace('_', ' ')[:25])
        ax.plot(s2_norm.index, s2_norm.values, color=colors[i % len(colors)][1],
                linewidth=1, alpha=0.8, label=pair['article2'].replace('_', ' ')[:25])

        ax.set_title(f"Correlation: {pair['correlation']:.2f}", fontsize=10, color=COLORS['text'])
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylabel('Normalized Views')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Correlated Article Pairs', fontsize=14, fontweight='bold', color=COLORS['text'])
    plt.tight_layout()
    return fig_to_base64(fig)


# =============================================================================
# 4. WAVELET ANALYSIS - Multi-scale time-frequency
# =============================================================================

def compute_wavelet_scalogram(series: np.ndarray, wavelet: str = 'morl',
                               scales: np.ndarray = None) -> tuple:
    """Compute continuous wavelet transform."""
    if scales is None:
        # Scales corresponding to periods from 2 days to 400 days
        scales = np.arange(1, 200)

    coeffs, freqs = pywt.cwt(series, scales, wavelet)
    power = np.abs(coeffs) ** 2

    # Convert scales to periods (approximate)
    periods = scales * 2  # Rough conversion for Morlet wavelet

    return power, periods, scales


def plot_wavelet_scalogram(timeseries: dict, articles: list, n: int = 4) -> str:
    """Plot wavelet scalograms for selected articles."""
    if len(articles) == 0:
        return None

    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n))

    for i, article in enumerate(articles[:n]):
        if article not in timeseries:
            continue

        series = timeseries[article]

        # Time series
        ax1 = axes[i, 0]
        ax1.fill_between(series.index, series.values, alpha=0.3, color=COLORS['info'])
        ax1.plot(series.index, series.values, color=COLORS['info'], linewidth=0.8)

        title = article.replace('_', ' ')[:35]
        ax1.set_title(title, fontsize=10, color=COLORS['text'])
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K' if x < 1e6 else f'{x/1e6:.1f}M'))
        ax1.grid(True, alpha=0.3)

        # Wavelet scalogram
        ax2 = axes[i, 1]
        power, periods, scales = compute_wavelet_scalogram(series.values)

        # Use log scale for better visualization
        im = ax2.imshow(np.log10(power + 1), aspect='auto', cmap='magma',
                       extent=[0, len(series), periods[-1], periods[0]])

        ax2.set_ylabel('Period (days)')
        ax2.set_title('Wavelet Power Spectrum', fontsize=10, color=COLORS['text'])

        # Mark key periods
        for period, label in [(7, '1 week'), (30, '1 month'), (365, '1 year')]:
            if period < periods[-1]:
                ax2.axhline(y=period, color='white', linewidth=0.5, linestyle='--', alpha=0.5)
                ax2.text(len(series) * 0.02, period, label, fontsize=7, color='white', alpha=0.7)

    plt.suptitle('Wavelet Analysis - Multi-Scale Patterns', fontsize=14, fontweight='bold',
                 color=COLORS['text'], y=1.01)
    plt.tight_layout()
    return fig_to_base64(fig)


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_html(stats: dict, plots: dict, periodic: list, spike_df: pd.DataFrame,
                  correlated_pairs: list) -> str:
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

    # Periodic articles table
    periodic_rows = ''
    for item in periodic[:15]:
        periodic_rows += f'''
        <tr>
            <td>{wiki_link(item['article'])}</td>
            <td>{make_badge(item['period_type'], COLORS['info'])}</td>
            <td>{item['best_lag']} days</td>
            <td>{item['acf_strength']:.2f}</td>
        </tr>'''

    # Top spikes table
    top_spikes = spike_df.nlargest(15, 'multiplier') if len(spike_df) > 0 else pd.DataFrame()
    spike_rows = ''
    for _, row in top_spikes.iterrows():
        spike_rows += f'''
        <tr>
            <td>{wiki_link(row['article'])}</td>
            <td>{row['peak_date'].strftime('%Y-%m-%d')}</td>
            <td>{make_badge(row['spike_type'], COLORS['warning'])}</td>
            <td>{row['rise_days']:.0f}d / {row['decay_days']:.0f}d</td>
            <td>{make_badge(f"{row['multiplier']:.0f}x", COLORS['highlight'])}</td>
        </tr>'''

    # Correlated pairs table
    pair_rows = ''
    for pair in correlated_pairs[:15]:
        pair_rows += f'''
        <tr>
            <td>{wiki_link(pair['article1'])}</td>
            <td>{wiki_link(pair['article2'])}</td>
            <td>{make_badge(f"{pair['correlation']:.2f}", COLORS['success'])}</td>
        </tr>'''

    # Spike type summary
    type_summary = ''
    if len(spike_df) > 0:
        type_counts = spike_df['spike_type'].value_counts()
        for stype, count in type_counts.items():
            type_summary += f'<span class="stat-item"><strong>{stype.replace("_", " ").title()}:</strong> {count}</span>'

    # Missing dates summary
    missing_ranges_html = ''
    if stats.get('missing_ranges'):
        for start, end in stats['missing_ranges'][:20]:  # Limit to first 20 ranges
            if start == end:
                missing_ranges_html += f'<span class="badge" style="background: {COLORS["accent"]}; margin: 2px;">{start}</span> '
            else:
                days_in_range = (end - start).days + 1
                missing_ranges_html += f'<span class="badge" style="background: {COLORS["accent"]}; margin: 2px;">{start} to {end} ({days_in_range}d)</span> '
        if len(stats['missing_ranges']) > 20:
            missing_ranges_html += f'<span class="badge" style="background: {COLORS["muted"]};">+{len(stats["missing_ranges"]) - 20} more ranges</span>'

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wikipedia Spike Analysis</title>
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
            background: linear-gradient(90deg, {COLORS['warning']}, {COLORS['highlight']});
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
        }}
        .section p {{ color: {COLORS['muted']}; margin-bottom: 1rem; }}
        .section.highlight {{ border-left-color: {COLORS['highlight']}; }}
        .section.success {{ border-left-color: {COLORS['success']}; }}
        .section.warning {{ border-left-color: {COLORS['warning']}; }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            color: white;
        }}

        .stat-item {{
            display: inline-block;
            margin-right: 1.5rem;
            margin-bottom: 0.5rem;
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
        @media (max-width: 1000px) {{ .grid {{ grid-template-columns: 1fr; }} }}

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
            <h1>Wikipedia Spike Analysis</h1>
            <p class="subtitle">{stats['date_range']}</p>
            <p class="timestamp">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <section class="section">
            <h2>Data Coverage</h2>
            <div class="grid" style="grid-template-columns: 1fr 1fr 1fr 1fr;">
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
                    <div class="metric-value" style="color: {COLORS['success'] if stats['density'] > 0.95 else COLORS['warning'] if stats['density'] > 0.8 else COLORS['highlight']};">{stats['density']*100:.1f}%</div>
                    <div class="metric-label">Coverage Density</div>
                </div>
            </div>
            {f'<p style="margin-top: 1rem;"><strong>Missing dates ({stats["missing_days"]}):</strong></p><p style="margin-top: 0.5rem; line-height: 2;">{missing_ranges_html}</p>' if stats['missing_days'] > 0 else '<p style="color: ' + COLORS['success'] + '; margin-top: 1rem;">No missing dates - complete coverage!</p>'}
        </section>

        <section class="section">
            <h2>Analysis Summary</h2>
            <div class="stat-item"><strong>Articles analyzed:</strong> {stats['articles_analyzed']:,}</div>
            <div class="stat-item"><strong>Periodic patterns found:</strong> {len(periodic)}</div>
            <div class="stat-item"><strong>Spikes detected:</strong> {len(spike_df):,}</div>
            <div class="stat-item"><strong>Correlated pairs:</strong> {len(correlated_pairs)}</div>
            <p style="margin-top: 1rem;">{type_summary}</p>
        </section>

        <section class="section warning">
            <h2>1. Periodic Patterns (Autocorrelation Analysis)</h2>
            <p>Articles with recurring traffic patterns detected via autocorrelation.
               Yearly patterns often indicate anniversaries, holidays, or annual events.</p>
            <table>
                <thead>
                    <tr><th>Article</th><th>Pattern</th><th>Period</th><th>Strength</th></tr>
                </thead>
                <tbody>{periodic_rows}</tbody>
            </table>
        </section>

        {f'<section class="section"><h2>Periodic Pattern Visualization</h2><div class="chart"><img src="data:image/png;base64,{plots["periodic"]}" alt="Periodic Patterns"></div></section>' if plots.get('periodic') else ''}

        <section class="section highlight">
            <h2>2. Spike Shape Analysis</h2>
            <p>Spikes classified by their dynamics: <strong>Breaking News</strong> (sharp rise, slow decay),
               <strong>Sustained</strong> (gradual rise and fall), <strong>Anticipation</strong> (slow build, quick drop),
               <strong>Flash</strong> (very brief spike).</p>
        </section>

        {f'<section class="section"><div class="chart"><img src="data:image/png;base64,{plots["spike_shapes"]}" alt="Spike Shapes"></div></section>' if plots.get('spike_shapes') else ''}

        {f'<section class="section"><h2>Spike Type Examples</h2><div class="chart"><img src="data:image/png;base64,{plots["spike_examples"]}" alt="Spike Examples"></div></section>' if plots.get('spike_examples') else ''}

        <section class="section highlight">
            <h2>Largest Spikes</h2>
            <table>
                <thead>
                    <tr><th>Article</th><th>Date</th><th>Type</th><th>Rise/Decay</th><th>Multiplier</th></tr>
                </thead>
                <tbody>{spike_rows}</tbody>
            </table>
        </section>

        <section class="section success">
            <h2>3. Cross-Correlation Analysis</h2>
            <p>Articles that spike together, suggesting related topics or shared events.</p>
            <table>
                <thead>
                    <tr><th>Article 1</th><th>Article 2</th><th>Correlation</th></tr>
                </thead>
                <tbody>{pair_rows}</tbody>
            </table>
        </section>

        {f'<section class="section"><h2>Correlation Matrix (Top 50 Articles)</h2><div class="chart"><img src="data:image/png;base64,{plots["correlation"]}" alt="Correlation Matrix"></div></section>' if plots.get('correlation') else ''}

        {f'<section class="section"><h2>Correlated Pairs Time Series</h2><div class="chart"><img src="data:image/png;base64,{plots["corr_pairs"]}" alt="Correlated Pairs"></div></section>' if plots.get('corr_pairs') else ''}

        <section class="section">
            <h2>4. Wavelet Analysis</h2>
            <p>Multi-scale time-frequency decomposition reveals patterns at different timescales
               (days, weeks, months, years) and how they change over time.</p>
        </section>

        {f'<section class="section"><div class="chart"><img src="data:image/png;base64,{plots["wavelet"]}" alt="Wavelet Analysis"></div></section>' if plots.get('wavelet') else ''}

        <footer>
            Wikipedia Spike Analysis &bull; Advanced pattern detection using ACF, shape metrics, cross-correlation, and wavelets
        </footer>
    </div>
</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(description='Advanced Wikipedia spike analysis')
    parser.add_argument('--days', '-d', type=int, default=365, help='Days to analyze (default: 365)')
    parser.add_argument('--all', '-a', action='store_true', help='Analyze all available data')
    parser.add_argument('--output', '-o', help='Output filename')
    args = parser.parse_args()

    setup_plot_style()

    # Load data
    days = None if args.all else args.days
    print(f"Loading data{'' if args.all else f' (last {args.days} days)'}...")
    df = load_data(days)
    filtered_df = filter_content(df)
    print(f"  Loaded {len(filtered_df):,} records")

    # Analyze date coverage
    print("Analyzing date coverage...")
    coverage = analyze_date_coverage(df)
    print(f"  Date range: {coverage['first_date']} to {coverage['last_date']}")
    print(f"  Coverage: {coverage['actual_days']}/{coverage['expected_days']} days ({coverage['density']*100:.1f}%)")
    if coverage['missing_days'] > 0:
        print(f"  Missing: {coverage['missing_days']} days")

    # Get time series
    print("Building time series...")
    timeseries = get_article_timeseries(filtered_df, min_days=30)
    print(f"  {len(timeseries)} articles with 30+ days of data")

    stats = {
        'date_range': f"{coverage['first_date']} to {coverage['last_date']}",
        'first_date': coverage['first_date'],
        'last_date': coverage['last_date'],
        'expected_days': coverage['expected_days'],
        'actual_days': coverage['actual_days'],
        'missing_days': coverage['missing_days'],
        'density': coverage['density'],
        'missing_ranges': coverage['missing_ranges'],
        'articles_analyzed': len(timeseries)
    }

    plots = {}

    # 1. Autocorrelation / Periodicity
    print("Analyzing periodic patterns...")
    periodic = find_periodic_articles(timeseries)
    print(f"  Found {len(periodic)} articles with periodic patterns")
    if len(periodic) > 0:
        plots['periodic'] = plot_periodic_articles(periodic)

    # 2. Spike shapes
    print("Analyzing spike shapes...")
    spike_df = analyze_spike_shapes(timeseries)
    spike_df = classify_spike_types(spike_df)
    print(f"  Detected {len(spike_df)} spikes")
    if len(spike_df) > 0:
        plots['spike_shapes'] = plot_spike_shape_distribution(spike_df)
        plots['spike_examples'] = plot_spike_examples_by_type(timeseries, spike_df)

    # 3. Cross-correlation
    print("Computing cross-correlations...")
    corr_matrix, top_articles = compute_cross_correlation_matrix(timeseries, top_n=50)
    correlated_pairs = find_correlated_pairs(corr_matrix, top_articles, threshold=0.4)
    print(f"  Found {len(correlated_pairs)} correlated pairs (r > 0.4)")
    plots['correlation'] = plot_correlation_matrix(corr_matrix, top_articles)
    if len(correlated_pairs) > 0:
        plots['corr_pairs'] = plot_correlated_pairs(timeseries, correlated_pairs)

    # 4. Wavelet analysis (on interesting articles)
    print("Running wavelet analysis...")
    wavelet_articles = []
    # Add some periodic articles
    wavelet_articles.extend([p['article'] for p in periodic[:2]])
    # Add some high-spike articles
    if len(spike_df) > 0:
        wavelet_articles.extend(spike_df.nlargest(2, 'multiplier')['article'].tolist())
    wavelet_articles = list(dict.fromkeys(wavelet_articles))[:4]  # Dedupe, limit to 4

    if len(wavelet_articles) > 0:
        plots['wavelet'] = plot_wavelet_scalogram(timeseries, wavelet_articles)

    # Generate HTML
    print("Generating report...")
    html = generate_html(stats, plots, periodic, spike_df, correlated_pairs)

    # Save
    REPORTS_DIR.mkdir(exist_ok=True)
    if args.output:
        output_file = REPORTS_DIR / args.output
    else:
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = REPORTS_DIR / f'spike_analysis_{date_str}.html'

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"\nReport saved to: {output_file}")
    print(f"Open in browser: file://{output_file.absolute()}")


if __name__ == '__main__':
    main()
