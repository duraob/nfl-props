"""
EXPERIMENTAL - season-long player prop projections and edge vs Kalshi's KXNFLSEASON*
markets. Built entirely from the existing weekly model (projections.project(),
exceed_probability), summing 17-18 single-week point estimates and discounting by
GAMES_PLAYED_RATE (a measured per-position games-missed correction - see below).

Backtested once, leak-free: project(2025, 1, seasons=[2024,2025]) - week 1 of 2025
uses only 2024 as trailing history regardless of what else is loaded, so this is a
true "as if run in August 2025" snapshot - multiplied by each team's real 17-game
season, vs actual 2025 outcomes. This is the only season this can be tested on
without crashing: depth_chart_ranks() only supports the 2025+ schema (see
nfl_source.py), so 2023/2024 backtests hit the no-history rookie-prior path and raise.
n=1 season - real evidence, not a robust multi-year study; 2026 in-season results are
the next genuinely clean check, same caveat as the weekly model's own 2025 holdout.

Results, restricted to the position(s) that actually produce each stat (mixing in
positions that never touch the ball dilutes the error to near-zero trivially):

  stat         n    raw bias   discounted bias   MAE improvement   spearman
  rush_yd    127     +26.5%          +0.9%            -13.4%         0.723
  rec_yd     315     +30.3%          +1.0%            -18.2%         0.771
  receptions 315     +30.8%          +1.7%            -17.4%         0.753
  pass_yd     33     +21.7%          -9.5%             -3.0%         0.251

GAMES_PLAYED_RATE nearly fully corrects bias for rush_yd/rec_yd/receptions and
meaningfully improves MAE on all three - real, validated confidence, not just theory.
It overcorrects for pass_yd (flips positive bias into negative) and barely moves
MAE - QB is already documented as the model's weakest category (see CLAUDE.md); this
season-level check reinforces that with fresh evidence rather than new speculation.
Treat pass_yd/QB output here as the least trustworthy of the four - not fixed, since
retuning a single position's rate off one season's 33 QBs would be fitting noise, not
a measurement.

Kept as its own module, separate from projections.py/market_odds.py, so this stays
easy to ignore or delete without touching anything the test suite / backtest gate
depends on.
"""

from __future__ import annotations

import pandas as pd

import market_odds as M
import projections as P

# The KXNFLSEASON* series odds_capture.py captures -> the build() stat they price.
# Title template confirmed against real captured rows: "Will {Player} record {N}+
# {description} during [the] 2026-27 Pro Football regular season?" - different from
# the weekly "{Player}: {N}+ {description}" template in market_odds.py, so parsed
# separately rather than folding into that function's regex.
SEASON_SERIES_TO_STAT = {
    "KXNFLSEASONPASSYDS": "pass_yd",
    "KXNFLSEASONPASSTDS": "pass_td",
    "KXNFLSEASONRECTD": "rec_td",
    "KXNFLSEASONRSHTD": "rush_td",
    # Volume props - confirmed live 2026-08-17, not previously captured (see
    # odds_capture.py). exceed_probability supports all three directly, unlike the
    # TD series above.
    "KXNFLSEASONRECYDS": "rec_yd",
    "KXNFLSEASONRSHYDS": "rush_yd",
    "KXNFLSEASONREC": "receptions",
}
_SEASON_TITLE_RE = r"^Will (?P<player_name>.+?) record (?P<line>\d+(?:\.\d+)?)\+"

SUM_COLUMNS = ["pass_yd", "rush_yd", "rec_yd", "receptions", "pass_td", "rush_td",
               "rec_td", "interceptions", "e_targets", "e_carries", "e_attempts"]

# Measured on nflverse 2022-2025: for a player with a real Week 1 role (QB
# attempts>=15, RB carries>=8, WR/TE targets>=4), the fraction of his team's
# remaining scheduled games he actually appeared in that season - for any reason
# (injury, benching, trade, IR), not just measurable injuries. This is the piece a
# raw sum of weekly point estimates has no way to know: it assumes a player plays
# every week he's rostered for, which real players never do. n=128/155/287/93.
GAMES_PLAYED_RATE = {"QB": 0.7433, "RB": 0.8067, "WR": 0.7681, "TE": 0.7747}


def project_season(season: int, weeks=range(1, 19), seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Sum project(season, week) across `weeks` per player, discounted by
    GAMES_PLAYED_RATE for the player's position.

    A bye week naturally contributes nothing: project()'s placeholder rows only
    cover teams with a scheduled game that week, so a bye-week player simply has no
    row that week rather than a spurious zero - no special-casing needed. Confidence
    columns are carried from whichever week a player's row happened to sort first,
    not recomputed for the season horizon - see the module docstring for why that's
    an approximation, not a measured number.

    GAMES_PLAYED_RATE is applied uniformly across every week in `weeks`, including
    week 1 itself - slightly conservative there, since an established Week 1
    starter's chance of playing *that* specific game is closer to 100% than the
    season-long average. Accepted as a simplification rather than splitting the
    discount by week.
    """
    weekly = pd.concat([P.project(season, w, seasons) for w in weeks], ignore_index=True)
    totals = weekly.groupby("player_id", as_index=False)[SUM_COLUMNS].sum(min_count=1)
    meta = weekly.groupby("player_id", as_index=False).first()[
        ["player_id", "player_display_name", "position", "team",
         "confidence_yardage", "confidence_touchdown"]
    ]
    combined = totals.merge(meta, on="player_id")
    rate = combined.position.map(GAMES_PLAYED_RATE)
    combined[SUM_COLUMNS] = combined[SUM_COLUMNS].mul(rate, axis=0)
    return combined


def normalize_kalshi_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Same output shape as market_odds.normalize_kalshi, for the four season series."""
    columns = ["player_id", "stat", "line", "implied_probability", "venue", "captured_at"]
    rows = df[df.series.isin(SEASON_SERIES_TO_STAT)].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows["stat"] = rows.series.map(SEASON_SERIES_TO_STAT)
    rows = pd.concat([rows, rows.title.str.extract(_SEASON_TITLE_RE)], axis=1)
    rows = rows.dropna(subset=["player_name", "line", "mid"])
    rows["line"] = rows["line"].astype(float)

    names = M._roster_name_lookup(season)
    rows["player_id"] = rows.player_name.map(names)
    unmatched = rows.player_id.isna().sum()
    if unmatched:
        print(f"normalize_kalshi_season: {unmatched} row(s) had no exact roster name match, dropped")
    rows = rows.dropna(subset=["player_id"])

    rows = rows.rename(columns={"mid": "implied_probability"}).assign(venue="kalshi")[columns]
    # Multiple captures over time land in the same kalshi.csv - keep only the most
    # recent price per line, matching market_odds.market_lines()'s convention.
    return rows.sort_values("captured_at").drop_duplicates(
        subset=["player_id", "stat", "line"], keep="last"
    )


def season_edge(season: int) -> pd.DataFrame:
    """
    Season projections vs captured KXNFLSEASON* lines.

    Edge (model probability - implied probability) is only computed for pass_yd -
    the one season stat exceed_probability supports (see STAT_DISPERSION in
    projections.py). Touchdowns are deliberately excluded there at any horizon,
    because their 0.09 reliability ceiling makes a probability false precision - if
    anything more so summed across a season than for one week. The three season TD
    series still come back with a `projection` column so they can be read next to
    the captured line, just with model_probability/edge left NaN rather than faked.
    """
    columns = ["player_id", "player_display_name", "stat", "line", "implied_probability",
               "venue", "captured_at", "projection", "model_probability", "edge"]
    if not M.KALSHI_HISTORY.exists():
        raise FileNotFoundError(f"{M.KALSHI_HISTORY} not found - run odds_capture.py first")

    market = normalize_kalshi_season(pd.read_csv(M.KALSHI_HISTORY), season)
    if market.empty:
        return pd.DataFrame(columns=columns)

    proj = project_season(season).set_index("player_id")
    frames = []
    for stat in market.stat.unique():
        rows = market[(market.stat == stat) & market.player_id.isin(proj.index)].copy()
        if rows.empty:
            continue
        rows["player_display_name"] = proj.loc[rows.player_id, "player_display_name"].to_numpy()
        rows["projection"] = proj.loc[rows.player_id, stat].to_numpy()
        if stat in P.STAT_DISPERSION:
            matched = proj.loc[rows.player_id].set_axis(rows.index)
            rows["model_probability"] = P.exceed_probability(matched, stat, rows.line.to_numpy())
            rows["edge"] = rows.model_probability - rows.implied_probability
        else:
            rows["model_probability"] = float("nan")
            rows["edge"] = float("nan")
        frames.append(rows)

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]
