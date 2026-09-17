"""
Grade a week's bets against actuals and captured closing lines.

Run once a week's games are done (Tuesday is the natural cadence). Joins ledger.py's
predictions.csv + bets.csv to real outcomes (nfl_source.weekly_stats) for error and
result, and to a captured closing line for CLV where one can be reliably matched.

CLV covers both venues. It used to be DraftKings-only, on the grounds that Kalshi's
`title` is free text and unmatchable - true of the game markets ("Will Kansas City
win..."), but not of the weekly player props, whose titles are a strict template
("{Player Name}: {N}+ receiving yards"). market_odds.normalize_kalshi already parses
them into (player_id, stat, line, yes_ask), so this joins through there rather than
re-deriving it.

`beat_close` is venue-specific and the two directions are opposite. A DraftKings
price is American odds, where a bigger number pays more, so beating the close means
`price > closing_price`. A Kalshi price is what you paid in cents for a $1 contract,
where lower is better, so beating the close means `price < closing_price`. One shared
comparison would silently invert every Kalshi bet's CLV.

Both venues compare ask-to-ask: `yes_ask` at close against the ask actually paid.
Grading an entry ask against a closing *mid* would book the bid/ask spread as lost
CLV on every bet, which is an artifact of the measurement, not a cost you paid.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import market_odds as M
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
KALSHI_HISTORY = Path("data/odds_history/kalshi.csv")


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


def _kalshi_closing_lines(season: int) -> pd.DataFrame:
    """Last captured ask per (player_id, stat, line) - the closing price, as far as
    the capture history knows it.

    Kalshi markets vanish from the open-markets sweep once they close at kickoff, so
    the final captured row for a market is necessarily pre-kickoff and no post-game
    price can leak in here. Verified against the Week 1 history: zero captured rows
    have captured_at past their own close_time. This is only as good as the last
    sweep is close to kickoff, which is a property of the capture schedule, not of
    this function - see `closing_captured_at` in the output, which exists so a CLV
    number can never be read without also seeing how stale the price behind it is.
    """
    if not KALSHI_HISTORY.exists():
        return pd.DataFrame(columns=["player_id", "stat", "line", "yes_ask", "captured_at"])
    lines = M.normalize_kalshi(pd.read_csv(KALSHI_HISTORY), season)
    if lines.empty:
        return lines
    return lines.sort_values("captured_at").drop_duplicates(
        subset=["player_id", "stat", "line"], keep="last"
    )[["player_id", "stat", "line", "yes_ask", "captured_at"]]


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
    # Object dtype, holding tz-aware Timestamps: a typed datetime column cannot
    # take an all-NaT assignment when no Kalshi line matches, and the only use of
    # this column is the row-wise comparison against ts_utc below.
    out["closing_captured_at"] = None
    out["beat_close"] = None

    dk_rows = out.venue == "draftkings"
    if dk_rows.any():
        player_names = src.rosters([season]).set_index("gsis_id").full_name
        dk = pd.read_csv(DRAFTKINGS, parse_dates=["captured_at"]) if DRAFTKINGS.exists() else pd.DataFrame()
        out.loc[dk_rows, "closing_price"] = out.loc[dk_rows].apply(
            lambda bet: _dk_closing_price(bet, dk, player_names), axis=1
        )

    kalshi_rows = out.venue == "kalshi"
    if kalshi_rows.any():
        closes = _kalshi_closing_lines(season)
        if not closes.empty:
            matched = out.loc[kalshi_rows, ["player_id", "stat", "line"]].merge(
                closes, on=["player_id", "stat", "line"], how="left"
            )
            out.loc[kalshi_rows, "closing_price"] = matched.yes_ask.to_numpy()
            out.loc[kalshi_rows, "closing_captured_at"] = pd.to_datetime(
                matched.captured_at, utc=True
            ).to_numpy(dtype=object)

    has_close = out.closing_price.notna()
    # Opposite directions by venue - see the module docstring. Sharing one
    # comparison would invert every Kalshi bet.
    out.loc[has_close & dk_rows, "beat_close"] = (
        out.loc[has_close & dk_rows, "price"] > out.loc[has_close & dk_rows, "closing_price"]
    )
    out.loc[has_close & kalshi_rows, "beat_close"] = (
        out.loc[has_close & kalshi_rows, "price"] < out.loc[has_close & kalshi_rows, "closing_price"]
    )

    # A "closing" price captured before the bet was placed is the same snapshot the
    # bet was made from, so its CLV is 0.00 by construction and measures the capture
    # schedule, not the bet. Every Week 1 bet was like this - the last sweep runs
    # ~3h before kickoff and nothing is captured between it and the game. Flagged
    # per row rather than silently averaged into a CLV that would read as "no skill"
    # when it actually means "no closing line was ever taken".
    out["close_is_stale"] = [
        bool(captured is not None and not pd.isna(captured) and captured <= placed)
        for captured, placed in zip(out.closing_captured_at, out.ts_utc)
    ]

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
