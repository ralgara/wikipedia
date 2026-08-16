"""Expectation baselines, residual spike scoring, view bands, and co-spike context.

WHY THIS EXISTS

`analysis.detect_spikes` scores z = (views - mean) / std against a flat baseline: one mean
and one standard deviation per article for the whole archive. Two complaints fall out of
that single choice, and they are the same complaint:

  * Fixtures (articles present near-daily at high rank) have small sigma because they are
    stable, so a modest bump clears z > 3. A flat baseline makes them spike-PRONE.
  * Calendar recurrences sit near zero for 364 days, dragging the mean down, so
    Independence_Day on 4 July produces an enormous multiplier every single year and
    crowds out everything else.

Filtering both out would work and is what one reaches for first, but it throws away the
rows a co-occurrence kernel needs, and it needs a list somebody maintains. Modelling the
expectation instead handles both without an exclusion list:

    expected = level(article) x weekday_factor x day_of_year_factor
    residual = observed / expected

    Independence_Day on 4 Jul     expected is already high  -> residual ~1, boring
    Independence_Day in February  expected is low           -> residual high, interesting
    Independence_Day 3x last year expected is this-date-shaped -> residual high, and THAT
                                  is the case a flat baseline cannot see at all

Periodicity is not discarded; it is absorbed into `expected` and still available as a
column, so the kernel keeps the correlated cluster it needs.

TWO HONEST LIMITS

1. CENSORING. The archive is the daily top 1000. Absence is not zero views, it is "below
   the line", and the line moves — on a heavy news day the top absorbs attention and the
   rank-1000 cutoff rises, pushing mid-tier articles out of the archive for reasons that
   have nothing to do with them. `censoring_threshold` publishes that line per day so any
   tiered result can be read against it.

2. SELECTION BIAS IN THE LEVEL. An article's level is estimated only from days it was
   visible, and for a rare article those are by definition its highest days. Its level is
   therefore biased UP and its residuals biased DOWN. This is inherited from
   `detect_spikes`, not introduced here, but it matters more as you slice into lower
   bands. `days_present` travels with every level so callers can weigh it; treat residuals
   for articles with few appearances as indicative rather than quantitative.

Nothing here modifies `analysis.detect_spikes`. Existing reports keep their current
numbers; this is additive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# An article needs at least this many appearances before a weekday profile is worth
# fitting. Below it, seven buckets over a handful of days is noise dressed as structure.
MIN_DAYS_FOR_DOW = 28

# Day-of-year factors need the same calendar position observed in more than one year,
# otherwise "last year's spike" becomes "this date is expected to be big" from a single
# observation — which would make every one-off event permanently unremarkable.
MIN_YEARS_FOR_DOY = 2

# Deliberately low, and the first version of this file got it badly wrong at 200.
#
# A seasonal article is BY DEFINITION absent most of the year: Independence_Day charts for
# a few days around 4 July and is below the top-1000 line the rest of the time, so five
# years of history is only ~50 appearances. A 200-day gate therefore excluded precisely
# the articles day-of-year factors exist to model, and the anniversaries it was built to
# quieten sailed straight through unchanged.
#
# What actually matters is repetition at the same calendar position across years, which
# MIN_YEARS_FOR_DOY already enforces. This gate only drops articles too sparse for any
# profile to mean anything.
MIN_DAYS_FOR_DOY = 20

# Day-of-year is smoothed over +/- this many days. Absorbs the leap-year off-by-one (a
# fixed date shifts one day-of-year after February in a leap year) and the fact that
# real-world observances drift across weekends.
DOY_WINDOW = 3

# Residuals below this observed-view floor are not reported. A tiny expected value makes
# the ratio explode; this keeps the output about articles with real traffic.
MIN_VIEWS_FOR_SPIKE = 10_000

# Month-day -> index in a fixed non-leap reference year, so a calendar date always lands
# in the same bucket regardless of leap years. Built once at import.
_MD_TO_INDEX = {
    d.month * 100 + d.day: i + 1
    for i, d in enumerate(pd.date_range('2001-01-01', '2001-12-31'))
}


def article_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Per-article baseline level.

    Median rather than mean, deliberately: the mean of a series that contains the spike
    is pulled toward the spike, which shrinks the very residual we are trying to measure.

    Returns columns: article, level, days_present.
    """
    grouped = df.groupby('article')['views']
    levels = grouped.median().rename('level').reset_index()
    levels['days_present'] = grouped.size().values
    # A zero level would make every residual infinite.
    levels = levels[levels['level'] > 0]
    return levels


def weekday_factors(df: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """Per-article, per-weekday multiplier relative to that article's level.

    Wikipedia traffic has a real weekly shape — reference and school-adjacent topics sag
    at weekends, entertainment does the opposite — and leaving it in the residual makes
    every Monday look mildly eventful.

    Returns columns: article, dow, dow_factor.
    """
    eligible = levels[levels['days_present'] >= MIN_DAYS_FOR_DOW][['article', 'level']]
    if eligible.empty:
        return pd.DataFrame(columns=['article', 'dow', 'dow_factor'])

    work = df.merge(eligible, on='article')
    work['dow'] = work['date'].dt.dayofweek
    factors = (
        work.groupby(['article', 'dow'])
        .apply(lambda g: g['views'].median() / g['level'].iloc[0], include_groups=False)
        .rename('dow_factor')
        .reset_index()
    )
    return factors


def dayofyear_factors(df: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """Per-article, per-day-of-year multiplier relative to that article's level.

    This is the piece that makes an anniversary boring on its anniversary. Restricted to
    articles with substantial history: fitting 366 buckets for an article seen twice would
    be pure overfit, and the restriction also keeps the work proportional — only a few
    thousand articles qualify out of ~200k.

    Requires the same calendar position in at least MIN_YEARS_FOR_DOY distinct years, then
    smooths over a +/- DOY_WINDOW day window.

    Returns columns: article, doy, doy_factor.
    """
    eligible = levels[levels['days_present'] >= MIN_DAYS_FOR_DOY][['article', 'level']]
    if eligible.empty:
        return pd.DataFrame(columns=['article', 'doy', 'doy_factor'])

    work = df.merge(eligible, on='article').copy()
    # Key on MONTH-DAY, not raw day-of-year. 4 July is day-of-year 185 in a common year
    # and 186 in a leap year, so raw doy splits a fixed anniversary into two buckets that
    # each look like they were seen half as often. Projecting onto a fixed non-leap
    # reference year keeps 4 July in one bucket forever. 29 February folds onto 28
    # February, which is close enough for a factor.
    md = work['date'].dt.month * 100 + work['date'].dt.day
    work['doy'] = md.map(_MD_TO_INDEX).fillna(_MD_TO_INDEX[228]).astype(int)
    work['year'] = work['date'].dt.year

    agg = work.groupby(['article', 'doy']).agg(
        med=('views', 'median'),
        years=('year', 'nunique'),
        level=('level', 'first'),
    ).reset_index()
    agg = agg[agg['years'] >= MIN_YEARS_FOR_DOY]
    if agg.empty:
        return pd.DataFrame(columns=['article', 'doy', 'doy_factor'])

    agg['raw_factor'] = agg['med'] / agg['level']

    # Smooth across neighbouring calendar days, vectorised over all qualifying articles at
    # once. Lowering MIN_DAYS_FOR_DOY admits tens of thousands of articles, and a Python
    # loop doing reindex(366) per article is far too slow at that width.
    #
    # Robustness where it counts is already handled: `med` above is a median ACROSS YEARS,
    # so one freak year cannot define the profile. This second pass only bridges adjacent
    # days, which a mean does well enough — and it is what merges the leap-year split,
    # where a fixed date lands on two different day-of-year values (4 July is doy 185 in a
    # common year and 186 in a leap year, so the observations arrive in two buckets).
    wide = agg.pivot(index='doy', columns='article', values='raw_factor')
    wide = wide.reindex(range(1, 366))
    # MAX, not mean. A mean over +/- 3 days averages 4 July's factor against 1-3 and 5-7
    # July, which are ordinary, and so dilutes the peak to a third of its real height —
    # the first version of this did exactly that and left Independence_Day still reading
    # as a 4.9x spike on 4 July. Max keeps the peak intact where it belongs and also
    # covers observances that move within a week (Thanksgiving is the fourth Thursday of
    # November, so it lands anywhere in 22-28 Nov).
    #
    # The cost is deliberate and worth stating: days ADJACENT to a large anniversary
    # inherit its expectation, so genuine news on 3 July is suppressed. Quietening the
    # anniversary is the goal, and this errs toward that.
    smoothed = wide.rolling(window=2 * DOY_WINDOW + 1, center=True, min_periods=1).max()

    out = (
        smoothed.stack(future_stack=True)
        .rename('doy_factor')
        .reset_index()
        .dropna(subset=['doy_factor'])
    )
    out['doy'] = out['doy'].astype(int)
    return out[['article', 'doy', 'doy_factor']]


def add_expectation(df: pd.DataFrame, seasonal: bool = True) -> pd.DataFrame:
    """Attach expected / residual / band columns to a pageviews frame.

    Args:
        df: columns article, date, views (rank optional).
        seasonal: fit day-of-year factors. Off is materially faster and still removes
            fixtures and the weekly cycle — useful for a single year, where most articles
            cannot meet MIN_YEARS_FOR_DOY anyway.

    Returns a copy with added columns:
        level, dow_factor, doy_factor, expected, residual, band.
    """
    levels = article_levels(df)
    out = df.merge(levels, on='article', how='inner').copy()

    out['dow'] = out['date'].dt.dayofweek
    dow = weekday_factors(df, levels)
    out = out.merge(dow, on=['article', 'dow'], how='left')
    out['dow_factor'] = out['dow_factor'].fillna(1.0)

    if seasonal:
        out['doy'] = out['date'].dt.dayofyear
        doy = dayofyear_factors(df, levels)
        out = out.merge(doy, on=['article', 'doy'], how='left')
        out['doy_factor'] = out['doy_factor'].fillna(1.0)
    else:
        out['doy_factor'] = 1.0

    # A factor of zero would divide by zero downstream; a median of zero on some weekday
    # is real (an article that only ever charts midweek) but useless as a divisor.
    for col in ('dow_factor', 'doy_factor'):
        out[col] = out[col].replace(0, 1.0).fillna(1.0)

    out['expected'] = out['level'] * out['dow_factor'] * out['doy_factor']
    out['residual'] = out['views'] / out['expected']
    out['band'] = view_band(out['views'])
    return out


def detect_spikes_residual(df: pd.DataFrame, threshold: float = 3.0,
                           min_views: int = MIN_VIEWS_FOR_SPIKE,
                           seasonal: bool = True,
                           per_article: bool = True) -> pd.DataFrame:
    """Spikes scored against the expectation baseline rather than a flat mean.

    Deliberately NOT a modification of analysis.detect_spikes: existing reports keep
    their current numbers, and the two can be compared on the same archive.

    Args:
        threshold: minimum residual (observed / expected) to count as a spike.
        min_views: floor on observed views, so a small expectation cannot manufacture a
            large ratio out of noise.
        per_article: keep only each article's largest spike, matching detect_spikes'
            shape. False returns every article-day above threshold, which is what
            co-spike analysis wants.

    Returns columns: article, spike_date, spike_views, expected, residual, multiplier,
    band, days_present — a superset of detect_spikes' columns, so report code that reads
    'multiplier' keeps working.
    """
    enriched = add_expectation(df, seasonal=seasonal)
    spikes = enriched[(enriched['residual'] > threshold) &
                      (enriched['views'] >= min_views)].copy()

    if spikes.empty:
        return pd.DataFrame(columns=['article', 'spike_date', 'spike_views', 'expected',
                                     'residual', 'multiplier', 'band', 'days_present'])

    if per_article:
        spikes = spikes.loc[spikes.groupby('article')['residual'].idxmax()]

    spikes = spikes.rename(columns={'date': 'spike_date', 'views': 'spike_views'})
    # 'multiplier' is what the existing report templates and the narrative cache key off.
    spikes['multiplier'] = spikes['residual']
    cols = ['article', 'spike_date', 'spike_views', 'expected', 'residual',
            'multiplier', 'band', 'days_present']
    return spikes[cols].sort_values('residual', ascending=False).reset_index(drop=True)


# ── Tiering ───────────────────────────────────────────────────────────────────

# Absolute view bands rather than rank bands. Rank is relative to a truncated window, so
# "remove the top N" re-ranks inside the archive rather than revealing anything below it,
# and band membership would shift whenever the top moved. Absolute views do not.
BANDS = [
    (1_000_000, 'attractor'),   # Main_Page-class; a handful per day
    (100_000, 'major'),         # genuine mass-interest news
    (10_000, 'middle'),         # where most of the interesting signal lives
    (0, 'marginal'),            # near the censoring line; presence is itself fragile
]


def view_band(views) -> pd.Series:
    """Label views by order of magnitude. Accepts a Series or a scalar."""
    if np.isscalar(views):
        for floor, name in BANDS:
            if views >= floor:
                return name
        return BANDS[-1][1]
    out = pd.Series(BANDS[-1][1], index=views.index, dtype=object)
    for floor, name in reversed(BANDS):
        out[views >= floor] = name
    return out


def censoring_threshold(df: pd.DataFrame) -> pd.DataFrame:
    """Daily view count of the lowest-ranked article present — the censoring line.

    "What did it take to chart today?" Without this, a tiered analysis cannot tell a
    genuine drop-off from an article being pushed out of a top-1000 window that got more
    competitive. Rising threshold means the archive is less inclusive that day.

    Returns columns: date, threshold_views, articles_present.
    """
    grouped = df.groupby('date')['views']
    out = grouped.min().rename('threshold_views').reset_index()
    out['articles_present'] = grouped.size().values
    return out


# ── Co-spike context ──────────────────────────────────────────────────────────

def co_spikes(enriched: pd.DataFrame, date, exclude_article: str | None = None,
              top_n: int = 5, min_residual: float = 2.0,
              min_views: int = MIN_VIEWS_FOR_SPIKE) -> pd.DataFrame:
    """Articles that were also unusual on the same day, ranked by residual.

    Returns the union of two rankings — most abnormal for itself, and simply largest —
    because neither alone is sufficient, and finding that out cost a test run.

    Residual alone was the original design, on the reasoning that raw views just return
    the same handful of giants daily. But testing on 2026-02-20 showed residual DROPS
    Alysa Liu (4.6x) even though her Olympic gold that day is the actual reason Tiananmen
    Square spiked. Her February is a competition season, so the model is right that a
    February spike is unremarkable FOR HER — and still wrong about what mattered that day.

    Raw views alone has the opposite failure: it returns whatever is perennially huge.

    So: take both, label which ranking produced each, and let the caller see the shape.
    Fixtures largely fall out anyway once `is_content` has run and the residual floor is
    applied to the residual half.

    Args:
        enriched: output of add_expectation (or detect_spikes_residual with
            per_article=False).
        date: the day to look at; anything pandas can compare to the date column.
        exclude_article: the article being explained, omitted from its own context.
    """
    day = enriched[enriched['date'] == pd.Timestamp(date)]
    day = day[day['views'] >= min_views]
    if exclude_article is not None:
        day = day[day['article'] != exclude_article]
    if day.empty:
        return day.assign(why=pd.Series(dtype=object))[
            ['article', 'views', 'residual', 'band', 'why']]

    by_residual = day[day['residual'] >= min_residual].nlargest(top_n, 'residual')
    by_views = day.nlargest(top_n, 'views')

    combined = pd.concat([by_residual.assign(why='unusual'),
                          by_views.assign(why='large')])
    # An article surfacing on both rankings is the strongest signal there is; keep the
    # 'unusual' label, which is the more informative of the two.
    combined = combined.drop_duplicates(subset='article', keep='first')
    return combined.sort_values('residual', ascending=False)[
        ['article', 'views', 'residual', 'band', 'why']]


def co_spike_context(enriched: pd.DataFrame, spikes: pd.DataFrame,
                     top_n: int = 6) -> dict[str, str]:
    """Build "what else was unusual that day" blurbs, keyed "article::YYYY-MM-DD".

    The narrator otherwise sees one article, one date and a multiplier, in isolation — it
    cannot know that Alysa Liu won Olympic gold the same day Tiananmen Square spiked, even
    though both rows are sitting in the archive. This hands it the day's context so the
    connection is available rather than having to be recalled.

    Keys match the narrative cache, so a caller can zip them together directly.
    """
    out: dict[str, str] = {}
    for _, spike in spikes.iterrows():
        date = spike['spike_date']
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        others = co_spikes(enriched, date, exclude_article=spike['article'], top_n=top_n)
        if others.empty:
            continue
        lines = [
            f"  - {r['article'].replace('_', ' ')} ({r['views']:,} views, "
            f"{r['residual']:.1f}x its norm)"
            for _, r in others.iterrows()
        ]
        out[f"{spike['article']}::{date_str}"] = '\n'.join(lines)
    return out
