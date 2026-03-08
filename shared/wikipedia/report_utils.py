"""Shared HTML/plot utilities for Wikipedia reports.

All report scripts import from here to ensure a consistent dark theme,
number formatting, and chart style.
"""

import base64
import io
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

COLORS = {
    'bg':        '#1a1a2e',
    'card':      '#16213e',
    'accent':    '#0f3460',
    'highlight': '#e94560',
    'text':      '#eaeaea',
    'muted':     '#a0a0a0',
    'success':   '#4ecca3',
    'warning':   '#ffc93c',
    'info':      '#00adb5',
}

PALETTE = [
    COLORS['highlight'], COLORS['info'], COLORS['success'],
    COLORS['warning'], '#9b59b6', '#3498db', '#e67e22', '#1abc9c',
    '#e74c3c', '#2ecc71',
]


def setup_plot_style():
    plt.rcParams.update({
        'figure.facecolor':  COLORS['card'],
        'axes.facecolor':    COLORS['bg'],
        'axes.edgecolor':    COLORS['muted'],
        'axes.labelcolor':   COLORS['text'],
        'text.color':        COLORS['text'],
        'xtick.color':       COLORS['text'],
        'ytick.color':       COLORS['text'],
        'grid.color':        COLORS['accent'],
        'grid.alpha':        0.3,
        'legend.facecolor':  COLORS['card'],
        'legend.edgecolor':  COLORS['accent'],
    })
    sns.set_palette(PALETTE)


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=COLORS['card'], edgecolor='none')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img


def format_number(n) -> str:
    n = float(n)
    if n >= 1e9:  return f'{n/1e9:.1f}B'
    if n >= 1e6:  return f'{n/1e6:.1f}M'
    if n >= 1e3:  return f'{n/1e3:.1f}K'
    return f'{n:.0f}'


def wiki_link(article: str) -> str:
    display = article.replace('_', ' ')
    return f'<a href="https://en.wikipedia.org/wiki/{article}" target="_blank">{display}</a>'


def make_badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{text}</span>'


# ---------------------------------------------------------------------------
# Shared plot functions
# ---------------------------------------------------------------------------

def plot_top_articles(df: pd.DataFrame, n: int = 15, title: str = 'Top Articles') -> str:
    top = df.groupby('article')['views'].sum().nlargest(n).reset_index().iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.4)))
    bars = ax.barh(top['article'], top['views'], color=COLORS['highlight'], alpha=0.8)
    ax.set_xlabel('Total Views')
    ax.set_title(title, fontsize=14, fontweight='bold', color=COLORS['text'])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
    max_v = top['views'].max()
    for bar, val in zip(bars, top['views']):
        ax.text(val + max_v * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val/1e6:.1f}M', va='center', fontsize=8, color=COLORS['text'])
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_spike_examples(df: pd.DataFrame, spike_df: pd.DataFrame, n: int = 4) -> str | None:
    if spike_df.empty:
        return None
    top = spike_df.head(n)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, (_, spike) in enumerate(top.iterrows()):
        if i >= len(axes):
            break
        ax = axes[i]
        art = df[df['article'] == spike['article']].sort_values('date')
        color = PALETTE[i % len(PALETTE)]
        ax.fill_between(art['date'], art['views'], alpha=0.3, color=color)
        ax.plot(art['date'], art['views'], color=color, linewidth=1.5)
        ax.axvline(spike['spike_date'], color=COLORS['highlight'], linestyle='--', alpha=0.7)
        title = spike['article'].replace('_', ' ')[:28]
        ax.set_title(f"{title}\n{spike['multiplier']:.0f}x on {str(spike['spike_date'])[:10]}",
                     fontsize=9, color=COLORS['text'])
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K' if x < 1e6 else f'{x/1e6:.1f}M'))
        ax.grid(True, alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Biggest Traffic Spikes', fontsize=13, fontweight='bold',
                 color=COLORS['text'], y=1.01)
    plt.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML scaffolding
# ---------------------------------------------------------------------------

def _css() -> str:
    C = COLORS
    return f"""
        *{{margin:0;padding:0;box-sizing:border-box}}
        body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
              background:{C['bg']};color:{C['text']};line-height:1.6;padding:2rem}}
        .container{{max-width:1200px;margin:0 auto}}
        header{{text-align:center;margin-bottom:2rem;padding-bottom:1rem;
                border-bottom:2px solid {C['accent']}}}
        h1{{font-size:2.5rem;font-weight:700;margin-bottom:.5rem;
            background:linear-gradient(90deg,{C['info']},{C['highlight']});
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
        h2{{font-size:1.3rem;margin-bottom:1rem;color:{C['text']}}}
        .subtitle{{color:{C['muted']};font-size:1.1rem}}
        .timestamp{{color:{C['muted']};font-size:.85rem;margin-top:.5rem}}
        .nav{{margin-bottom:.5rem}}
        .nav a{{color:{C['info']};font-size:.9rem;margin-right:1rem}}
        .section{{background:{C['card']};border-radius:12px;padding:1.5rem;
                  margin-bottom:1.5rem;border-left:4px solid {C['info']}}}
        .section.highlight{{border-left-color:{C['highlight']}}}
        .section.success{{border-left-color:{C['success']}}}
        .section.warning{{border-left-color:{C['warning']}}}
        .metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
                  gap:1rem;margin-bottom:1rem}}
        .metric{{background:{C['bg']};padding:1rem;border-radius:8px;text-align:center}}
        .metric-value{{font-size:1.8rem;font-weight:700;color:{C['info']}}}
        .metric-label{{font-size:.85rem;color:{C['muted']};margin-top:.25rem}}
        .badge{{display:inline-block;padding:.25rem .6rem;border-radius:20px;
                font-size:.8rem;font-weight:600;color:#fff}}
        table{{width:100%;border-collapse:collapse;margin-top:1rem}}
        th,td{{padding:.75rem;text-align:left;border-bottom:1px solid {C['accent']}}}
        th{{color:{C['muted']};font-weight:600;font-size:.85rem;text-transform:uppercase}}
        tr:hover{{background:{C['bg']}}}
        a{{color:{C['info']};text-decoration:none}}
        a:hover{{color:{C['highlight']};text-decoration:underline}}
        .chart{{margin:1rem 0;text-align:center}}
        .chart img{{max-width:100%;border-radius:8px}}
        .narrative{{color:{C['muted']};font-size:.88rem;font-style:italic;
                    padding:.3rem 0 .3rem .5rem;border-left:2px solid {C['accent']}}}
        .grid{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}}
        .yoy-up{{color:{C['success']}}} .yoy-down{{color:{C['highlight']}}}
        footer{{text-align:center;margin-top:2rem;padding-top:1rem;
                border-top:1px solid {C['accent']};color:{C['muted']};font-size:.85rem}}
        @media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
    """


def html_page(title: str, subtitle: str, nav: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>{title}</title>
  <style>{_css()}</style>
</head>
<body>
  <div class="container">
    <header>
      <div class="nav">{nav}</div>
      <h1>{title}</h1>
      <p class="subtitle">{subtitle}</p>
      <p class="timestamp">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </header>
    {body}
    <footer>Wikipedia Pageviews Analytics &bull; Data from Wikimedia REST API</footer>
  </div>
</body>
</html>"""
