"""
Prediction and bet ledger: append-only, timestamped before kickoff.

Nothing records what was predicted or bet today, so nothing can be graded later.
Pre-kickoff timestamps make retroactive editing structurally impossible, which is
the most common way these systems quietly lie to their owner - the same reasoning as
odds_capture.py/dk_capture.py never rewriting a past row. See settle.py for grading
these against actuals and captured closing lines.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

import projections as P

PREDICTIONS = Path("data/odds_history/predictions.csv")
RECOMMENDATIONS = Path("data/odds_history/recommendations.csv")
BETS = Path("data/odds_history/bets.csv")
RECOMMENDATION_COLUMNS = [
    "ts_utc", "season", "week", "game_date", "slot", "player_id", "stat", "venue",
    "line", "yes_ask", "model_probability", "implied_probability", "edge_at_ask",
    "spread", "depth",
]
BET_COLUMNS = [
    "ts_utc", "season", "week", "player_id", "stat", "venue", "side", "line",
    "price", "stake",
]


def log_predictions(proj_df: pd.DataFrame) -> Path:
    """
    Append one row per (player, stat) in a projections.project() result.

    Call this right before generating/sending a report, not on a schedule - the
    timestamp is the whole point, and it should reflect when the projection was
    actually acted on.
    """
    ts = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    rows = [
        pd.DataFrame({
            "ts_utc": ts,
            "model_version": P.MODEL_VERSION,
            "season": proj_df.season,
            "week": proj_df.week,
            "player_id": proj_df.player_id,
            "stat": stat,
            "projection": proj_df[stat],
            "confidence": proj_df[conf_col],
        })
        for stat, conf_col in P.STAT_CONFIDENCE_COLUMN.items()
    ]
    out = pd.concat(rows, ignore_index=True)
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDICTIONS, mode="a", header=not PREDICTIONS.exists(), index=False)
    print(f"Logged {len(out)} predictions ({proj_df.player_id.nunique()} players) -> {PREDICTIONS}")
    return PREDICTIONS


def log_recommendations(bets_df: pd.DataFrame, season: int, week: int,
                        game_date: str | None = None) -> Path:
    """
    Append the bet sheet exactly as it was sent, numbered.

    Two jobs, and the second is the one that will matter in February. First, `slot`
    is what `/bet 2 25` resolves against - a number is the only thing worth typing on
    a phone, and it removes any chance of a mistyped player or threshold reaching an
    append-only file. Second, this records every recommendation whether or not it was
    backed, which is the only way to ever grade the screen itself: bets.csv holds the
    handful actually placed, and a handful per season can never say whether the +20
    point cap or the 25-75% window are the right numbers. Same append-only,
    timestamped-before-kickoff discipline as log_predictions above.
    """
    if bets_df.empty:
        return RECOMMENDATIONS
    ts = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    out = bets_df.reset_index(drop=True).assign(
        ts_utc=ts, season=season, week=week, game_date=game_date,
        slot=lambda d: d.index + 1,
    )[RECOMMENDATION_COLUMNS]
    RECOMMENDATIONS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RECOMMENDATIONS, mode="a", header=not RECOMMENDATIONS.exists(), index=False)
    print(f"Logged {len(out)} recommendations -> {RECOMMENDATIONS}")
    return RECOMMENDATIONS


def latest_recommendations() -> pd.DataFrame:
    """The most recently logged sheet, which is what a /bet slot number refers to."""
    if not RECOMMENDATIONS.exists():
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)
    rows = pd.read_csv(RECOMMENDATIONS)
    if rows.empty:
        return rows
    return rows[rows.ts_utc == rows.ts_utc.max()]


def record_bet(season: int, week: int, player_id: str, stat: str, venue: str,
                side: str, line: float, price: float, stake: float,
                ts_utc: str | None = None) -> Path:
    """
    Append one manually-placed bet.

    side is "over" or "under" - for a Kalshi YES bet on a threshold, record "over";
    for NO, record "under". line is the threshold; price is the odds actually taken
    (American for DraftKings, Kalshi's own cents-on-the-dollar for Kalshi).

    ts_utc defaults to now. telegram_commands.py passes the Telegram message's own
    timestamp instead, so a bet recorded from a phone is stamped when it was sent
    rather than whenever the polling cron happened to collect it.
    """
    if venue not in ("kalshi", "draftkings"):
        raise ValueError(f"venue must be 'kalshi' or 'draftkings', got {venue!r}")
    if side not in ("over", "under"):
        raise ValueError(f"side must be 'over' or 'under', got {side!r}")

    row = pd.DataFrame([{
        "ts_utc": ts_utc or dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "season": season, "week": week, "player_id": player_id, "stat": stat,
        "venue": venue, "side": side, "line": line, "price": price, "stake": stake,
    }])[BET_COLUMNS]
    BETS.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(BETS, mode="a", header=not BETS.exists(), index=False)
    return BETS
