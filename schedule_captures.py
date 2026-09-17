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
import subprocess
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

# Actions worth waking you up for when they fail - see _alert.
ALERT_ACTIONS = {"predictions", "report"}

# How close to kickoff the second ("closing") Kalshi sweep fires.
#
# Without it there is no closing line at all: the only sweep ran at
# suggest_capture_by, ~3h out, the bet gets placed off that same sheet minutes
# later, and nothing is captured in the hours remaining before kickoff. Every 2026
# Week 1 bet priced against a snapshot taken *before* the bet was placed, making CLV
# 0.00 by construction (settle.py flags these `close_is_stale`).
#
# 30 minutes is a compromise, not an ideal. Closer to kickoff is a better closing
# price, but a full Kalshi sweep took ~12 min in Week 1 and grows as more markets
# open - starting at T-30 finishes around T-18, while T-15 risks running past
# kickoff. Running past is not harmful (closed markets simply drop out of the open
# sweep, so no post-kickoff price can be recorded) but it wastes the sweep.
CLOSING_SWEEP_LEAD = dt.timedelta(minutes=30)

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


def _marker(season: int, week: int, game_date, action: str) -> Path:
    return STATE_DIR / f"{season}_{week}_{game_date}.{action}"


def _try(label: str, fn) -> Exception | None:
    """Run fn, returning the exception it raised, or None if it succeeded."""
    try:
        fn()
        return None
    except Exception as exc:
        print(f"  {label}: FAILED ({type(exc).__name__}: {exc})")
        return exc


def _alert(season: int, week: int, game_date, action: str, label: str,
           exc: Exception) -> None:
    """
    Push a failure notice for the two actions whose silent failure costs something
    that cannot be recreated afterwards: the prediction log (a pre-kickoff timestamp
    cannot be backfilled honestly - see ledger.py) and the report itself.

    Sent at most once per (window, action): the retry loop runs every 15 minutes
    until kickoff, and twelve identical warnings would only train you to ignore the
    thirteenth. Best-effort - if Telegram is itself what's broken there is nothing
    better to fall back to, so a failed alert stays silent rather than masking the
    original error.
    """
    marker = _marker(season, week, game_date, f"{action}-alerted")
    if marker.exists():
        return
    try:
        T.send_message(f"\u26a0\ufe0f {season} Week {week} ({game_date}): {label} failed "
                       f"- {type(exc).__name__}: {exc}")
        marker.touch()
    except Exception:
        pass


def sync_captured_data() -> None:
    """
    Commit and push any new rows under data/odds_history/ (kalshi.csv,
    draftkings.csv, predictions.csv, bets.csv).

    Captured betting lines cannot be reconstructed after the fact - the whole point
    of odds_capture.py/dk_capture.py is never losing them - so leaving them only on
    this machine's disk is a real, permanent data-loss risk if the droplet is ever
    rebuilt or dies. Requires the droplet's git remote to have push access (a deploy
    key with write access, or an HTTPS token) - see deploy/setup.sh's comments.
    Best-effort like every other action here: a git/network failure must not be
    treated as a capture failure, since the data is still safe on local disk either
    way and will sync on the next successful run. `run()` does alert on failure,
    though - see there for why this one is not silent like the rest.

    Rebases onto origin before pushing. Without that, one commit made anywhere else
    (a fix pushed from a laptop) leaves this machine permanently diverged, and every
    later push is rejected non-fast-forward. That is not hypothetical - it happened
    after the 2026-09-10 capture and stranded a full week of Kalshi snapshots, bets
    and recommendations here, visible to nobody. Rebase rather than merge because
    this machine only ever commits appended CSV rows, never code, so replaying them
    onto whatever origin has is always the right resolution.
    """
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--", "data/odds_history"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if changed:
        subprocess.run(["git", "add", "data/odds_history"], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"capture: {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC"],
            check=True,
        )

    # Deliberately NOT gated on `changed`. A push that failed earlier leaves a
    # commit sitting here with nothing new to add, so returning early on "no new
    # rows" is precisely what turns one rejected push into a permanent outage.
    subprocess.run(["git", "fetch", "origin", "main"], check=True)
    ahead = subprocess.run(
        ["git", "rev-list", "--count", "origin/main..HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if ahead == "0":
        return
    print(f"  syncing {ahead} commit(s) to origin")
    subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)


def _push_report(season: int, week: int, game_date: str) -> None:
    """
    Send the bet sheet, then record it.

    Logged after the send, not before, so a sheet only ever exists in the ledger if
    it actually reached the phone - a /bet slot number that resolves against a sheet
    you never saw would silently record the wrong wager.
    """
    text = R.build_report(season, week, game_date=game_date)
    T.send_message(text)
    L.log_recommendations(getattr(R.build_report, "last_bets", pd.DataFrame()),
                          season, week, game_date)


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
        # A marker per action, not one per window. A window whose report failed has
        # to retry that report on the next tick - bounded by the window closing at
        # kickoff - without re-running what already succeeded: re-sweeping
        # DraftKings every 15 minutes would spend the whole month's credit budget in
        # one afternoon (see dk_capture.py's budget note).
        actions = [
            ("kalshi", "Kalshi capture", K.capture),
            ("dk", "DraftKings capture", lambda: DK.capture(within_hours=6)),
            ("predictions", "log predictions",
             lambda: L.log_predictions(P.project(season, week))),
            ("report", "Telegram push",
             lambda: _push_report(season, week, str(row.game_date))),
        ]
        # A second sweep near kickoff, and the only one that produces a price the
        # bet can honestly be graded against - see CLOSING_SWEEP_LEAD. Its own
        # marker, so it is not skipped by the first sweep's.
        if now_eastern >= row.earliest_kickoff - CLOSING_SWEEP_LEAD:
            actions.append(("kalshi_close", "Kalshi closing sweep", K.capture))
        pending = [a for a in actions
                   if not _marker(season, week, row.game_date, a[0]).exists()]
        if not pending:
            continue

        print(f"{now:%Y-%m-%d %H:%M} UTC: acting on {season} week {week}, {row.game_date}")
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        for action, label, fn in pending:
            exc = _try(label, fn)
            if exc is None:
                _marker(season, week, row.game_date, action).touch()
            elif action in ALERT_ACTIONS:
                _alert(season, week, row.game_date, action, label, exc)

        # Deliberately unmarkered: predictions logged on a later retry still need
        # pushing, and the function no-ops when there is nothing to sync. Alerted on
        # failure like predictions/report, and for the same reason - a sync that
        # quietly stopped working looks exactly like one that is working, right up
        # until the droplet dies with the only copy of the season's bets on it.
        exc = _try("sync captured data to git", sync_captured_data)
        if exc is not None:
            _alert(season, week, row.game_date, "sync", "git sync", exc)


if __name__ == "__main__":
    run()
