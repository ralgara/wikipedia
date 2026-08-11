#!/usr/bin/env python3
"""Operational dashboard — ingestion health, dataset inventory, recent trends.

The other reports are *analytical*: they answer questions about Wikipedia. This one is
*operational*: it answers questions about the pipeline itself. Is ingestion current? Did
the last run publish? What has moved lately?

    uv run python scripts/generate-dashboard.py                 # 90-day trend window
    uv run python scripts/generate-dashboard.py --days 30
    uv run python scripts/generate-dashboard.py --no-gcs        # skip bucket inventory

Deliberately memory-light: a full all-time load peaks at ~2 GB and the runner has 4 GB and
no swap, so ingestion and inventory are derived from filenames and stat() alone. Only the
trend window actually parses rows.

Environment:
    GCS_BUCKET  bucket to inventory (default: wikipedia-cortex-data)
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.wikipedia.analysis import detect_spikes, filter_content
from shared.wikipedia.report_utils import (
    COLORS, format_number, html_page, make_badge, plot_top_articles, wiki_link,
)

DATA_DIR = ROOT / 'data'
REPORTS_DIR = ROOT / 'reports'
DEFAULT_BUCKET = 'wikipedia-cortex-data'

PAGEVIEWS_RE = re.compile(r'pageviews_(\d{8})\.json$')

# The Wikimedia REST API publishes a day's totals a day or two in arrears, so "current"
# means the newest file is within this many days of today — not that it is today.
API_LAG_DAYS = 2
STALE_AFTER_DAYS = 4


# ---------------------------------------------------------------- ingestion

def scan_archive(data_dir: Path) -> dict:
    """Archive state from filenames and stat() only — no JSON parsing."""
    entries = []
    for fp in sorted(glob(str(data_dir / 'pageviews_*.json'))):
        m = PAGEVIEWS_RE.search(fp)
        if not m:
            continue
        entries.append((datetime.strptime(m.group(1), '%Y%m%d'), Path(fp).stat().st_size))

    if not entries:
        return {'days': 0, 'dates': [], 'bytes': 0, 'gaps': [], 'first': None, 'last': None}

    dates = [d for d, _ in entries]
    first, last = dates[0], dates[-1]
    present = set(dates)
    span = (last - first).days + 1
    gaps = [first + timedelta(days=i) for i in range(span)
            if (first + timedelta(days=i)) not in present]

    return {
        'days': len(dates),
        'dates': dates,
        'bytes': sum(b for _, b in entries),
        'gaps': gaps,
        'first': first,
        'last': last,
        'span': span,
    }


def freshness(archive: dict) -> tuple[int, str, str]:
    """(days_behind, badge_colour, label) relative to what the API can actually offer."""
    if not archive['last']:
        return (999, 'highlight', 'no data')
    behind = (datetime.now() - archive['last']).days
    lag = max(0, behind - API_LAG_DAYS)
    if behind <= API_LAG_DAYS + 1:
        return (behind, 'success', 'current')
    if behind <= STALE_AFTER_DAYS:
        return (behind, 'warning', f'{lag}d behind')
    return (behind, 'highlight', f'{lag}d behind')


def ingestion_section(archive: dict) -> str:
    behind, colour, label = freshness(archive)
    gaps = archive['gaps']
    coverage = (archive['days'] / archive['span'] * 100) if archive.get('span') else 0

    gap_html = (
        f'<p style="color:{COLORS["success"]}">No gaps — every day between '
        f'{archive["first"]:%Y-%m-%d} and {archive["last"]:%Y-%m-%d} is present.</p>'
    ) if not gaps else (
        f'<p style="color:{COLORS["warning"]}">{len(gaps)} missing day(s). '
        f'<code>download-pageviews.py</code> fills these on the next run.</p>'
        '<p style="font-size:.85rem;color:#a0a0a0">'
        + ', '.join(d.strftime('%Y-%m-%d') for d in gaps[:20])
        + (f' … and {len(gaps) - 20} more' if len(gaps) > 20 else '') + '</p>'
    )

    # Per-year coverage: days present vs days the year can actually offer.
    by_year = {}
    for d in archive['dates']:
        by_year[d.year] = by_year.get(d.year, 0) + 1
    rows = []
    today = datetime.now()
    for year in sorted(by_year):
        start = max(datetime(year, 1, 1), archive['first'])
        end = min(datetime(year, 12, 31), today - timedelta(days=API_LAG_DAYS))
        expected = max(1, (end - start).days + 1)
        got = by_year[year]
        pct = got / expected * 100
        bar_colour = COLORS['success'] if pct >= 99.5 else COLORS['warning']
        rows.append(
            f'<tr><td>{year}</td><td>{got:,}</td><td>{expected:,}</td>'
            f'<td><div style="background:#1a1a2e;border-radius:4px;height:14px;width:160px">'
            f'<div style="background:{bar_colour};height:14px;border-radius:4px;'
            f'width:{min(100, pct):.0f}%"></div></div></td>'
            f'<td>{pct:.1f}%</td></tr>'
        )

    strip = day_strip(archive, days=120)

    return f"""
    <div class="section {'success' if colour == 'success' else colour}">
      <h2>Ingestion</h2>
      <div class="metrics">
        <div class="metric"><div class="metric-value">{archive['days']:,}</div>
          <div class="metric-label">Days archived</div></div>
        <div class="metric"><div class="metric-value">{coverage:.1f}%</div>
          <div class="metric-label">Coverage of span</div></div>
        <div class="metric"><div class="metric-value">{len(gaps)}</div>
          <div class="metric-label">Missing days</div></div>
        <div class="metric"><div class="metric-value">{archive['bytes']/1e9:.2f} GB</div>
          <div class="metric-label">Archive on disk</div></div>
        <div class="metric"><div class="metric-value">{make_badge(label, COLORS[colour])}</div>
          <div class="metric-label">Latest: {archive['last']:%Y-%m-%d}</div></div>
      </div>
      {gap_html}
      <h3 style="margin-top:1.5rem;font-size:1rem;color:#a0a0a0">Last 120 days</h3>
      {strip}
      <table>
        <tr><th>Year</th><th>Days</th><th>Expected</th><th>Coverage</th><th></th></tr>
        {''.join(rows)}
      </table>
    </div>"""


def day_strip(archive: dict, days: int = 120) -> str:
    """One cell per recent day — present, missing, or not-yet-available."""
    if not archive['last']:
        return ''
    present = set(archive['dates'])
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cells = []
    for i in range(days - 1, -1, -1):
        d = today - timedelta(days=i)
        if d in present:
            c, t = COLORS['success'], f'{d:%Y-%m-%d} present'
        elif (today - d).days <= API_LAG_DAYS:
            c, t = '#2a2a44', f'{d:%Y-%m-%d} not yet published by the API'
        else:
            c, t = COLORS['highlight'], f'{d:%Y-%m-%d} MISSING'
        cells.append(
            f'<span title="{t}" style="display:inline-block;width:8px;height:20px;'
            f'background:{c};margin-right:1px;border-radius:2px"></span>'
        )
    return f'<div style="line-height:1;margin:.5rem 0">{"".join(cells)}</div>'


# ---------------------------------------------------------------- datasets

def dataset_section(archive: dict, bucket_name: str, use_gcs: bool) -> str:
    local_reports = sorted(REPORTS_DIR.rglob('*.html'))
    stats_path = DATA_DIR / 'longitudinal' / 'stats.json'
    cache_path = DATA_DIR / 'narratives_cache.json'

    n_narratives = 0
    if cache_path.exists():
        try:
            n_narratives = len(json.loads(cache_path.read_text()))
        except (json.JSONDecodeError, OSError):
            n_narratives = 0

    sqlite_path = DATA_DIR / 'pageviews.db'

    rows = [
        _ds_row('Pageviews archive', 'local', f"{archive['days']:,} files",
                f"{archive['bytes']/1e9:.2f} GB",
                f"{archive['first']:%Y-%m-%d} → {archive['last']:%Y-%m-%d}"
                if archive['first'] else '—'),
        # On-demand v0.4 enrichment artifact, not a pipeline output — no stage builds or
        # consumes it, and absence is the normal state on every host including the VM.
        _ds_row('SQLite mirror', 'local',
                'pageviews.db' if sqlite_path.exists() else 'absent',
                f"{sqlite_path.stat().st_size/1e6:.0f} MB" if sqlite_path.exists() else '—',
                (f"built {datetime.fromtimestamp(sqlite_path.stat().st_mtime):%Y-%m-%d %H:%M}"
                 f" &middot; on-demand, rebuild with convert-to-sqlite.py") if sqlite_path.exists()
                else 'on-demand (v0.4 tooling) &middot; rebuild with convert-to-sqlite.py'),
        _ds_row('HTML reports', 'local', f'{len(local_reports)} files',
                f"{sum(p.stat().st_size for p in local_reports)/1e6:.1f} MB", 'regenerated each run'),
        _ds_row('Longitudinal stats', 'local', 'stats.json' if stats_path.exists() else 'absent',
                f"{stats_path.stat().st_size/1e3:.0f} KB" if stats_path.exists() else '—',
                f"{datetime.fromtimestamp(stats_path.stat().st_mtime):%Y-%m-%d %H:%M}"
                if stats_path.exists() else '—'),
        _ds_row('Narrative cache', 'local', f'{n_narratives:,} entries',
                f"{cache_path.stat().st_size/1e3:.0f} KB" if cache_path.exists() else '—',
                f"{datetime.fromtimestamp(cache_path.stat().st_mtime):%Y-%m-%d %H:%M}"
                if cache_path.exists() else '—'),
    ]

    note = ''
    if use_gcs:
        gcs_rows, note = gcs_inventory(bucket_name)
        rows.extend(gcs_rows)
    else:
        note = '<p style="color:#a0a0a0;font-size:.85rem">Bucket inventory skipped (--no-gcs).</p>'

    return f"""
    <div class="section">
      <h2>Live datasets</h2>
      <p style="color:#a0a0a0;font-size:.85rem">
        <strong>local</strong> means the host that generated this page. On the runner that is the
        pipeline's working copy; on a laptop it is whatever that machine last synced, which may
        legitimately lag the bucket. <strong>gcs</strong> is the durable copy and the shared truth.
      </p>
      <table>
        <tr><th>Dataset</th><th>Where</th><th>Objects</th><th>Size</th><th>Detail</th></tr>
        {''.join(rows)}
      </table>
      {note}
    </div>"""


def _ds_row(name, where, objects, size, detail) -> str:
    tag = make_badge(where, COLORS['info'] if where == 'local' else COLORS['success'])
    return (f'<tr><td><strong>{name}</strong></td><td>{tag}</td>'
            f'<td>{objects}</td><td>{size}</td><td style="color:#a0a0a0">{detail}</td></tr>')


def gcs_inventory(bucket_name: str) -> tuple[list[str], str]:
    """Bucket contents. Degrades to a note rather than failing the whole dashboard."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        groups = {'wikipedia/pageviews': [0, 0, None], 'reports': [0, 0, None]}
        for prefix in groups:
            for blob in bucket.list_blobs(prefix=prefix):
                g = groups[prefix]
                g[0] += 1
                g[1] += blob.size or 0
                if blob.updated and (g[2] is None or blob.updated > g[2]):
                    g[2] = blob.updated

        rows = []
        for prefix, (n, size, newest) in groups.items():
            label = 'Pageviews archive' if 'pageviews' in prefix else 'Published reports'
            rows.append(_ds_row(label, 'gcs', f'{n:,} objects', f'{size/1e6:.1f} MB',
                                f'newest {newest:%Y-%m-%d %H:%M UTC}' if newest else '—'))
        return rows, (
            f'<p style="color:#a0a0a0;font-size:.85rem">Public root: '
            f'<a href="https://storage.googleapis.com/{bucket_name}/reports/index.html">'
            f'storage.googleapis.com/{bucket_name}/reports/</a></p>'
        )
    except Exception as exc:  # noqa: BLE001 — the dashboard must render regardless
        return [], (f'<p style="color:{COLORS["warning"]};font-size:.85rem">'
                    f'Bucket inventory unavailable ({type(exc).__name__}). '
                    f'The rest of the dashboard is unaffected.</p>')


# ---------------------------------------------------------------- trends

def load_window(data_dir: Path, start: datetime, end: datetime) -> pd.DataFrame:
    """Parse only the files inside [start, end] — the memory-light path."""
    records = []
    for fp in sorted(glob(str(data_dir / 'pageviews_*.json'))):
        m = PAGEVIEWS_RE.search(fp)
        if not m:
            continue
        d = datetime.strptime(m.group(1), '%Y%m%d')
        if start <= d <= end:
            with open(fp) as f:
                records.extend(json.load(f))
    if not records:
        return pd.DataFrame(columns=['article', 'views', 'rank', 'date'])
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df


def trends_section(archive: dict, days: int) -> str:
    if not archive['last']:
        return '<div class="section warning"><h2>Recent trends</h2><p>No data.</p></div>'

    end = archive['last']
    start = end - timedelta(days=days - 1)
    prior_end = start - timedelta(days=1)
    prior_start = prior_end - timedelta(days=days - 1)

    cur = filter_content(load_window(DATA_DIR, start, end))
    prev = filter_content(load_window(DATA_DIR, prior_start, prior_end))

    if cur.empty:
        return '<div class="section warning"><h2>Recent trends</h2><p>No rows in window.</p></div>'

    top_chart = plot_top_articles(cur, n=15, title=f'Top articles — last {days} days')

    movers_html = build_movers(cur, prev, days)

    spikes = detect_spikes(cur, threshold=3.0).head(10)
    spike_rows = ''.join(
        f'<tr><td>{wiki_link(r.article)}</td><td>{r.spike_date:%Y-%m-%d}</td>'
        f'<td>{format_number(r.spike_views)}</td>'
        f'<td style="color:{COLORS["highlight"]}">{r.multiplier:.1f}×</td></tr>'
        for r in spikes.itertuples()
    ) or '<tr><td colspan="4" style="color:#a0a0a0">No spikes above 3σ in this window.</td></tr>'

    total_views = int(cur['views'].sum())
    prev_total = int(prev['views'].sum()) if not prev.empty else 0
    delta = ((total_views - prev_total) / prev_total * 100) if prev_total else 0
    delta_cls = 'yoy-up' if delta >= 0 else 'yoy-down'

    return f"""
    <div class="section highlight">
      <h2>Recent trends — last {days} days</h2>
      <p style="color:#a0a0a0;font-size:.9rem">
        {start:%Y-%m-%d} → {end:%Y-%m-%d}, compared against the preceding {days} days
        ({prior_start:%Y-%m-%d} → {prior_end:%Y-%m-%d}). Content articles only.
      </p>
      <div class="metrics">
        <div class="metric"><div class="metric-value">{format_number(total_views)}</div>
          <div class="metric-label">Views in window</div></div>
        <div class="metric"><div class="metric-value {delta_cls}">{delta:+.1f}%</div>
          <div class="metric-label">vs prior window</div></div>
        <div class="metric"><div class="metric-value">{cur['article'].nunique():,}</div>
          <div class="metric-label">Distinct articles</div></div>
        <div class="metric"><div class="metric-value">{len(spikes)}</div>
          <div class="metric-label">Spikes &gt;3&sigma;</div></div>
      </div>
      <div class="chart"><img src="data:image/png;base64,{top_chart}" alt="Top articles"></div>
      {movers_html}
      <h3 style="margin-top:1.5rem;font-size:1rem;color:#a0a0a0">Recent spikes</h3>
      <table>
        <tr><th>Article</th><th>Date</th><th>Views</th><th>Multiplier</th></tr>
        {spike_rows}
      </table>
    </div>"""


def build_movers(cur: pd.DataFrame, prev: pd.DataFrame, days: int) -> str:
    """Biggest risers and fallers between the two windows.

    Restricted to articles present in both windows with a meaningful base: an article that
    appears for the first time has an undefined percentage change, not an infinite one.
    """
    if prev.empty:
        return ('<p style="color:#a0a0a0;font-size:.85rem">'
                'No prior window available for comparison.</p>')

    c = cur.groupby('article')['views'].sum()
    p = prev.groupby('article')['views'].sum()
    both = pd.DataFrame({'now': c, 'before': p}).dropna()
    both = both[both['before'] >= 50_000]
    if both.empty:
        return ('<p style="color:#a0a0a0;font-size:.85rem">'
                'Not enough overlap between windows to compute movers.</p>')

    both['pct'] = (both['now'] - both['before']) / both['before'] * 100
    risers = both.nlargest(10, 'pct')
    fallers = both.nsmallest(10, 'pct')

    def table(frame, title, cls):
        rows = ''.join(
            f'<tr><td>{wiki_link(a)}</td><td>{format_number(int(r.now))}</td>'
            f'<td class="{cls}">{r.pct:+.0f}%</td></tr>'
            for a, r in frame.iterrows()
        )
        return (f'<div><h3 style="font-size:1rem;color:#a0a0a0">{title}</h3><table>'
                f'<tr><th>Article</th><th>Views</th><th>Change</th></tr>{rows}</table></div>')

    return (f'<div class="grid" style="margin-top:1.5rem">'
            f'{table(risers, "Rising", "yoy-up")}'
            f'{table(fallers, "Falling", "yoy-down")}</div>')


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description='Operational dashboard for the pipeline')
    parser.add_argument('--days', type=int, default=90, help='Trend window in days (default: 90)')
    parser.add_argument('--no-gcs', action='store_true', help='Skip the bucket inventory')
    parser.add_argument('--bucket', default=os.environ.get('GCS_BUCKET', DEFAULT_BUCKET))
    parser.add_argument('--output', type=Path, default=REPORTS_DIR / 'dashboard.html')
    args = parser.parse_args()

    print('Scanning archive (filenames only)...')
    archive = scan_archive(DATA_DIR)
    print(f"  {archive['days']:,} days, {len(archive['gaps'])} gap(s), "
          f"{archive['bytes']/1e9:.2f} GB")

    print('Building ingestion + dataset panels...')
    body = ingestion_section(archive)
    body += dataset_section(archive, args.bucket, use_gcs=not args.no_gcs)

    print(f'Loading {args.days}-day trend window...')
    body += trends_section(archive, args.days)

    nav = ('<a href="index.html">Archive index</a>'
           '<a href="all-time.html">All-time</a>'
           '<a href="longitudinal.html">Longitudinal</a>')
    html = html_page(
        'Pipeline Dashboard',
        'Ingestion health, live datasets, and recent movement',
        nav, body,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f'\nWrote {args.output}  ({args.output.stat().st_size/1024:.0f} KB)')
    print(f'open {args.output}')


if __name__ == '__main__':
    main()
