"""
Run captures at the right time, driven by kickoff_windows() - not a fixed weekly
cron, which misses closing lines on a week with Wed/Thu games and acts too early on
Sun/Mon ones (see projections.kickoff_windows).

Designed to be invoked frequently and cheaply by an external scheduler (cron every
~15 min is the intended setup - see the "Automation" section of RUN.md for the
actual crontab line): this script itself does no waiting or sleeping. Each run
checks whether *now* falls inside any of this week's action windows
(suggest_capture_by through earliest_kickoff) that haven't been actioned yet, and if
so, captures both venues, logs predictions, and pushes the report - each step wrapped
so one failure (e.g. DK's credit budget) doesn't block the others.

Idempotency is a marker file per (season, week, game_date) under
data/scheduler_state/ (gitignored - see .gitignore), not a database: cron re-invokes
this script stateless every time, so "have we already acted on this window" has to
persist across runs on disk, and a whole season is at most ~20 markers.
"""

from __future__ import annotations

import datetime as dt
import zoneinfo
from pathlib import Path

import pandas as pd

import dk_capture as DK
import ledger as L
import nfl_source as src
import odds_capture as K
import projections as P
import report as R
import telegram_notify as T

STATE_DIR = Path("data/scheduler_state")

# kickoff_windows()'s timestamps are tz-naive and assumed Eastern (see
# nfl_source.py's gametime note - unverified against an authoritative source, but
# consistent across every game checked). `now` (tz-aware UTC) has to be converted
# into that same naive-Eastern space before comparing, not just have its tzinfo
# stripped - that would silently shift every window by 4-5 hours. zoneinfo (stdlib)
# handles the EDT/EST transition that falls inside every NFL season.
_EASTERN = zoneinfo.ZoneInfo("America/New_York")


def current_week(now: dt.datetime | None = None) -> tuple[int, int] | None:
    """
    (season, week) for whatever week is currently live, or None if there's no
    schedule to work from (deep offseason).

    "Season" is the year the NFL labels it by, which runs Sept - Feb: a January game
    still belongs to the season labeled the previous calendar year. "Current week" is
    the earliest week whose games haven't all finished yet, with a day of grace past
    the latest kickoff so Monday night's aftermath doesn't roll over before Tuesday.
    """
    now = now or dt.datetime.now(dt.UTC)
    season = now.year if now.month >= 3 else now.year - 1
    try:
        sched = src.schedule([season])
    except Exception:
        return None

    sched = sched.dropna(subset=["gameday"]).copy()
    if sched.empty:
        return None
    sched["gameday"] = pd.to_datetime(sched["gameday"]).dt.tz_localize(now.tzinfo)
    latest_by_week = sched.groupby("week")["gameday"].max()
    upcoming = latest_by_week[latest_by_week >= (now - pd.Timedelta(days=1))]
    if upcoming.empty:
        return None
    return season, int(upcoming.idxmin())


def _marker(season: int, week: int, game_date) -> Path:
    return STATE_DIR / f"{season}_{week}_{game_date}.done"


def _try(label: str, fn) -> None:
    try:
        fn()
    except Exception as exc:
        print(f"  {label}: FAILED ({type(exc).__name__}: {exc})")


def run(now: dt.datetime | None = None) -> None:
    now = now or dt.datetime.now(dt.UTC)
    week_info = current_week(now)
    if week_info is None:
        print(f"{now:%Y-%m-%d %H:%M} UTC: no current NFL week (offseason) - nothing to do.")
        return
    season, week = week_info

    now_eastern = now.astimezone(_EASTERN).replace(tzinfo=None)
    windows = P.kickoff_windows(season, week)
    due = windows[(windows.suggest_capture_by <= now_eastern)
                  & (now_eastern <= windows.earliest_kickoff)]

    for _, row in due.iterrows():
        marker = _marker(season, week, row.game_date)
        if marker.exists():
            continue

        print(f"{now:%Y-%m-%d %H:%M} UTC: acting on {season} week {week}, {row.game_date}")
        _try("Kalshi capture", K.capture)
        _try("DraftKings capture", lambda: DK.capture(within_hours=6))
        _try("log predictions", lambda: L.log_predictions(P.project(season, week)))
        _try("Telegram push", lambda: T.send_message(
            R.build_report(season, week, game_date=str(row.game_date))
        ))

        STATE_DIR.mkdir(parents=True, exist_ok=True)
        marker.touch()


if __name__ == "__main__":
    run()
