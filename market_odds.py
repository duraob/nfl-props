"""
Normalize Kalshi and DraftKings captured lines into one shape:
(player_id, stat, line, implied_probability, venue, captured_at).

Kalshi and DraftKings both price props as thresholds, but their captured data looks
nothing alike (see odds_capture.py / dk_capture.py) - this is the one place that
reconciles them, so projections.exceed_probability and report.py never need to know
which venue a line came from.

Only pass_yd/rush_yd/rec_yd/receptions are covered - the stats projections.py can
turn into a probability (see STAT_DISPERSION) and that either venue actually prices
weekly. Kalshi's KXNFLRSHATT/PASSATT/PASSCOMP and DraftKings' player_rush_attempts/
player_pass_attempts have no corresponding build() stat and are skipped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import nfl_source as src
import projections as P

KALSHI_HISTORY = Path("data/odds_history/kalshi.csv")
DRAFTKINGS_HISTORY = Path("data/odds_history/draftkings.csv")

# Kalshi's weekly per-game player-prop series -> the build() stat they price.
# Confirmed against real settled preseason markets (odds_capture.fetch_markets):
# titles for these series are a strict template, "{Player Name}: {N}+ {description}"
# - not free text - unlike the game-market titles (KXNFLGAME etc.), which is why
# those are not parsed here or anywhere else in this codebase.
KALSHI_SERIES_TO_STAT = {
    "KXNFLRECYDS": "rec_yd", "KXNFLRSHYDS": "rush_yd",
    "KXNFLPASSYDS": "pass_yd", "KXNFLREC": "receptions",
}
_KALSHI_TITLE_RE = r"^(?P<player_name>.+?):\s*(?P<line>\d+(?:\.\d+)?)\+"

# DraftKings market -> build() stat. Only receptions is captured today - see
# dk_capture.py's MARKETS - so this is deliberately a subset of the four stats above.
DK_MARKET_TO_STAT = {"player_receptions": "receptions"}


def _roster_name_lookup(season: int) -> pd.Series:
    return src.rosters([season]).set_index("full_name").gsis_id


def _american_to_prob(price: float) -> float:
    return 100 / (price + 100) if price >= 0 else -price / (-price + 100)


def normalize_kalshi(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Kalshi's mid is already close to a fair price (~1% fee) - no de-vig needed,
    unlike DraftKings below."""
    rows = df[df.series.isin(KALSHI_SERIES_TO_STAT)].copy()
    if rows.empty:
        return pd.DataFrame(columns=["player_id", "stat", "line", "implied_probability",
                                     "venue", "captured_at"])
    rows["stat"] = rows.series.map(KALSHI_SERIES_TO_STAT)
    rows = pd.concat([rows, rows.title.str.extract(_KALSHI_TITLE_RE)], axis=1)
    rows = rows.dropna(subset=["player_name", "line", "mid"])
    rows["line"] = rows["line"].astype(float)

    names = _roster_name_lookup(season)
    rows["player_id"] = rows.player_name.map(names)
    unmatched = rows.player_id.isna().sum()
    if unmatched:
        print(f"normalize_kalshi: {unmatched} row(s) had no exact roster name match, dropped")
    rows = rows.dropna(subset=["player_id"])

    return rows.rename(columns={"mid": "implied_probability"}).assign(venue="kalshi")[
        ["player_id", "stat", "line", "implied_probability", "venue", "captured_at"]
    ]


def normalize_draftkings(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    DraftKings' price still carries the bookmaker's ~4.5% hold - paired Over/Under
    American odds for the same (event, market, line, capture) are de-vigged before
    reporting a probability, then only the Over side is kept (matching Kalshi's "X+"
    framing and exceed_probability's P(stat >= threshold) semantics).
    """
    rows = df[df.market.isin(DK_MARKET_TO_STAT)].copy()
    if rows.empty:
        return pd.DataFrame(columns=["player_id", "stat", "line", "implied_probability",
                                     "venue", "captured_at"])
    rows["stat"] = rows.market.map(DK_MARKET_TO_STAT)
    rows["raw_prob"] = rows.price.apply(_american_to_prob)

    key = ["event_id", "market", "line", "captured_at"]
    rows["fair_prob"] = rows.raw_prob / rows.groupby(key).raw_prob.transform("sum")

    over = rows[rows.side.str.lower() == "over"].copy()
    names = _roster_name_lookup(season)
    over["player_id"] = over.player.map(names)
    unmatched = over.player_id.isna().sum()
    if unmatched:
        print(f"normalize_draftkings: {unmatched} row(s) had no exact roster name match, dropped")
    over = over.dropna(subset=["player_id"])

    return over.rename(columns={"fair_prob": "implied_probability"}).assign(venue="draftkings")[
        ["player_id", "stat", "line", "implied_probability", "venue", "captured_at"]
    ]


def market_lines(season: int) -> pd.DataFrame:
    """Both venues, normalized and combined - each venue's most recently captured
    price per (player, stat, line)."""
    frames = []
    if KALSHI_HISTORY.exists():
        frames.append(normalize_kalshi(pd.read_csv(KALSHI_HISTORY), season))
    if DRAFTKINGS_HISTORY.exists():
        frames.append(normalize_draftkings(pd.read_csv(DRAFTKINGS_HISTORY), season))
    columns = ["player_id", "stat", "line", "implied_probability", "venue", "captured_at"]
    if not frames:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("captured_at").drop_duplicates(
        subset=["player_id", "stat", "line", "venue"], keep="last"
    )[columns]


def compute_edge(proj_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every captured market line matching a (player, stat) in proj_df, compute the
    model's P(stat >= line) and edge = model_probability - implied_probability.

    Positive edge means the model thinks the outcome is more likely than the market
    is pricing it. This is not a bet recommendation on its own - report edge
    alongside confidence, never in place of it (see report.py).
    """
    empty = market_df.iloc[0:0].assign(
        model_probability=pd.Series(dtype=float), projection=pd.Series(dtype=float),
        confidence=pd.Series(dtype=float), edge=pd.Series(dtype=float),
    )
    if market_df.empty:
        return empty

    proj_indexed = proj_df.set_index("player_id")
    frames = []
    for stat in market_df.stat.unique():
        if stat not in P.STAT_DISPERSION:
            continue
        rows = market_df[(market_df.stat == stat) & market_df.player_id.isin(proj_indexed.index)]
        if rows.empty:
            continue
        matched_proj = proj_indexed.loc[rows.player_id].set_axis(rows.index)
        result = rows.copy()
        result["model_probability"] = P.exceed_probability(matched_proj, stat, rows.line.to_numpy())
        result["projection"] = matched_proj[stat].to_numpy()
        result["confidence"] = matched_proj[P.STAT_CONFIDENCE_COLUMN[stat]].to_numpy()
        frames.append(result)

    if not frames:
        return empty
    out = pd.concat(frames, ignore_index=True)
    out["edge"] = out.model_probability - out.implied_probability
    return out.dropna(subset=["model_probability"])
