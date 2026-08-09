"""Data loading and analysis functions — cloud neutral, report-agnostic."""

import json
from glob import glob
from pathlib import Path

import pandas as pd

from .filters import is_content


def load_all_data(data_dir: Path) -> pd.DataFrame:
    files = sorted(glob(str(data_dir / 'pageviews_*.json')))
    if not files:
        raise FileNotFoundError(f"No data files in {data_dir}")
    records = []
    for fp in files:
        with open(fp) as f:
            records.extend(json.load(f))
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_year_data(data_dir: Path, year: int) -> pd.DataFrame:
    files = sorted(glob(str(data_dir / f'pageviews_{year}*.json')))
    if not files:
        raise FileNotFoundError(f"No data for {year} in {data_dir}")
    records = []
    for fp in files:
        with open(fp) as f:
            records.extend(json.load(f))
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df


def filter_content(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['article'].apply(is_content)].copy()


def detect_spikes(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Vectorised spike detection. Returns rows sorted by multiplier descending."""
    stats = df.groupby('article')['views'].agg(['mean', 'std']).reset_index()
    stats = stats[stats['std'] > 0]

    merged = df.merge(stats, on='article')
    merged['z'] = (merged['views'] - merged['mean']) / merged['std']
    spikes = merged[merged['z'] > threshold]

    if spikes.empty:
        return pd.DataFrame(columns=['article', 'spike_date', 'spike_views', 'avg_views', 'multiplier'])

    idx = spikes.groupby('article')['views'].idxmax()
    top = spikes.loc[idx][['article', 'date', 'views', 'mean']].copy()
    top.columns = ['article', 'spike_date', 'spike_views', 'avg_views']
    top['multiplier'] = top['spike_views'] / top['avg_views']
    return top.sort_values('multiplier', ascending=False).reset_index(drop=True)


def longitudinal_data(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Annual total views for the top_n all-time articles.

    Returns DataFrame with columns: article, year, total_views.
    Years where an article never appeared have total_views = 0.
    """
    df = df.copy()
    df['year'] = df['date'].dt.year

    top_articles = (
        df.groupby('article')['views'].sum()
        .nlargest(top_n)
        .index.tolist()
    )
    years = sorted(df['year'].unique())

    records = []
    for article in top_articles:
        art = df[df['article'] == article]
        for year in years:
            views = int(art[art['year'] == year]['views'].sum())
            records.append({'article': article, 'year': year, 'total_views': views})

    return pd.DataFrame(records)
