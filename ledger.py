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
BETS = Path("data/odds_history/bets.csv")
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


def record_bet(season: int, week: int, player_id: str, stat: str, venue: str,
                side: str, line: float, price: float, stake: float) -> Path:
    """
    Append one manually-placed bet.

    side is "over" or "under" - for a Kalshi YES bet on a threshold, record "over";
    for NO, record "under". line is the threshold; price is the odds actually taken
    (American for DraftKings, Kalshi's own cents-on-the-dollar for Kalshi).
    """
    if venue not in ("kalshi", "draftkings"):
        raise ValueError(f"venue must be 'kalshi' or 'draftkings', got {venue!r}")
    if side not in ("over", "under"):
        raise ValueError(f"side must be 'over' or 'under', got {side!r}")

    row = pd.DataFrame([{
        "ts_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "season": season, "week": week, "player_id": player_id, "stat": stat,
        "venue": venue, "side": side, "line": line, "price": price, "stake": stake,
    }])[BET_COLUMNS]
    BETS.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(BETS, mode="a", header=not BETS.exists(), index=False)
    return BETS
