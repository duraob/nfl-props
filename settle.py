"""
Grade a week's bets against actuals and captured closing lines.

Run once a week's games are done (Tuesday is the natural cadence). Joins ledger.py's
predictions.csv + bets.csv to real outcomes (nfl_source.weekly_stats) for error and
result, and to a captured closing line for CLV where one can be reliably matched.

CLV is DraftKings-only. dk_capture.py's rows are structured (player, market, line,
side), so a bet matches its closing line unambiguously on those four fields. Kalshi's
captured rows have no such field - `title` is free text ("Will Justin Jefferson have
75+ receiving yards?") - and guessing a player/threshold out of it risks silently
matching the wrong market's price, which is worse than reporting nothing. Kalshi bets
are graded for error/result exactly the same as DraftKings; closing_price/beat_close
are left null for them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import nfl_source as src

PREDICTIONS = Path("data/odds_history/predictions.csv")
BETS = Path("data/odds_history/bets.csv")
DRAFTKINGS = Path("data/odds_history/draftkings.csv")

# Raw weekly_stats column each tracked stat settles against.
STAT_RAW_COLUMN = {
    "pass_yd": "passing_yards", "rush_yd": "rushing_yards", "rec_yd": "receiving_yards",
    "receptions": "receptions", "pass_td": "passing_tds", "rush_td": "rushing_tds",
    "rec_td": "receiving_tds", "interceptions": "passing_interceptions",
}
# Only markets dk_capture.py actually captures (see its MARKETS list) have a closing
# line to match against - yardage props are not swept there today.
DK_MARKET_FOR_STAT = {"receptions": "player_receptions"}


def _actuals(season: int, week: int) -> pd.DataFrame:
    stats = src.weekly_stats([season])
    stats = stats[stats.week == week]
    return pd.concat([
        pd.DataFrame({"player_id": stats.player_id, "stat": stat, "actual": stats[col]})
        for stat, col in STAT_RAW_COLUMN.items()
    ], ignore_index=True)


def _latest_predictions(season: int, week: int) -> pd.DataFrame:
    preds = pd.read_csv(PREDICTIONS, parse_dates=["ts_utc"])
    preds = preds[(preds.season == season) & (preds.week == week)]
    # A player/stat logged more than once this week (e.g. a Wednesday and a Sunday
    # capture) - the most recent capture is the freshest read on the model.
    return preds.sort_values("ts_utc").drop_duplicates(
        subset=["player_id", "stat"], keep="last"
    )[["player_id", "stat", "projection", "confidence"]]


def _grade(actual: float, line: float, side: str) -> str | None:
    if pd.isna(actual):
        return None
    if actual == line:
        return "push"
    cleared = actual > line
    return "win" if (cleared if side == "over" else not cleared) else "loss"


def _dk_closing_price(bet: pd.Series, dk: pd.DataFrame, player_names: pd.Series) -> float | None:
    market = DK_MARKET_FOR_STAT.get(bet.stat)
    if market is None:
        return None
    name = player_names.get(bet.player_id)
    if name is None:
        return None
    match = dk[
        (dk.player == name) & (dk.market == market)
        & (dk.line == bet.line) & (dk.side.str.lower() == bet.side)
    ]
    if match.empty:
        return None
    return match.sort_values("captured_at").iloc[-1].price


def settle(season: int, week: int) -> pd.DataFrame:
    """One row per bet placed on (season, week): model error, result, and CLV."""
    if not BETS.exists():
        raise RuntimeError(f"No bets recorded yet at {BETS} - nothing to settle.")

    bets = pd.read_csv(BETS, parse_dates=["ts_utc"])
    bets = bets[(bets.season == season) & (bets.week == week)]
    if bets.empty:
        return bets

    out = bets.merge(_latest_predictions(season, week), on=["player_id", "stat"], how="left")
    out = out.merge(_actuals(season, week), on=["player_id", "stat"], how="left")
    out["error"] = out["actual"] - out["projection"]
    out["result"] = [_grade(a, l, s) for a, l, s in zip(out.actual, out.line, out.side)]

    out["closing_price"] = None
    out["beat_close"] = None
    dk_rows = out.venue == "draftkings"
    if dk_rows.any():
        player_names = src.rosters([season]).set_index("gsis_id").full_name
        dk = pd.read_csv(DRAFTKINGS, parse_dates=["captured_at"]) if DRAFTKINGS.exists() else pd.DataFrame()
        out.loc[dk_rows, "closing_price"] = out.loc[dk_rows].apply(
            lambda bet: _dk_closing_price(bet, dk, player_names), axis=1
        )
        has_close = out.closing_price.notna()
        out.loc[has_close, "beat_close"] = out.loc[has_close, "price"] > out.loc[has_close, "closing_price"]

    return out


if __name__ == "__main__":
    import sys

    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    week = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    result = settle(season, week)
    if result.empty:
        print(f"No bets recorded for {season} week {week}.")
    else:
        print(result.to_string(index=False))
