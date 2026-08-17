"""
Tests for the kickoff-window-driven scheduler.

current_week() is checked against real 2025 schedule dates (no mocking, matching
nfl_source's own test convention) rather than synthetic data, since its whole job is
correctly reading real calendar boundaries. run()'s action-dispatch logic is tested
with monkeypatched actions and a tmp_path STATE_DIR so nothing here ever calls a real
API or touches data/scheduler_state/.
"""

import datetime as dt

import pandas as pd

import schedule_captures as SC


def test_current_week_matches_real_schedule():
    """Week 5 2025 spans Oct 2-6; week 4 spans Sep 25-29 (verified via
    nfl_source.schedule([2025]))."""
    assert SC.current_week(dt.datetime(2025, 10, 5, 12, 0, tzinfo=dt.UTC)) == (2025, 5)
    assert SC.current_week(dt.datetime(2025, 9, 26, 12, 0, tzinfo=dt.UTC)) == (2025, 4)


def test_current_week_returns_the_upcoming_week_before_a_season_starts():
    """Well before 2026 Week 1 kicks off, the schedule exists but nothing has
    played - the earliest not-yet-finished week is 1, not None."""
    assert SC.current_week(dt.datetime(2026, 8, 16, tzinfo=dt.UTC)) == (2026, 1)


def _fake_window(now, capture_offset_hours=-1, kickoff_offset_hours=1, game_date="2099-09-10"):
    """Mirrors real kickoff_windows() output: tz-naive timestamps (assumed Eastern -
    see schedule_captures._EASTERN). `now` here is the tz-aware value passed to
    run(), converted the same way run() itself converts it."""
    now_eastern = now.astimezone(SC._EASTERN).replace(tzinfo=None)
    return pd.DataFrame({
        "game_date": [game_date],
        "earliest_kickoff": [now_eastern + dt.timedelta(hours=kickoff_offset_hours)],
        "suggest_capture_by": [now_eastern + dt.timedelta(hours=capture_offset_hours)],
    })


def _patch_actions(monkeypatch, calls, kalshi=None, dk=None):
    monkeypatch.setattr(SC.K, "capture", kalshi or (lambda: calls.append("kalshi")))
    monkeypatch.setattr(SC.DK, "capture", dk or (lambda within_hours: calls.append("dk")))
    monkeypatch.setattr(SC.L, "log_predictions", lambda proj: calls.append("predictions"))
    monkeypatch.setattr(SC.P, "project", lambda season, week: pd.DataFrame())
    monkeypatch.setattr(SC.R, "build_report", lambda season, week, game_date: "report text")
    monkeypatch.setattr(SC.T, "send_message", lambda text: calls.append("telegram"))


def test_run_actions_a_due_window_once(tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "STATE_DIR", tmp_path)
    monkeypatch.setattr(SC, "current_week", lambda now: (2099, 1))
    now = dt.datetime(2099, 9, 10, 12, 0, tzinfo=dt.UTC)
    monkeypatch.setattr(SC.P, "kickoff_windows", lambda season, week: _fake_window(now))

    calls = []
    _patch_actions(monkeypatch, calls)

    SC.run(now)
    assert set(calls) == {"kalshi", "dk", "predictions", "telegram"}
    assert (tmp_path / "2099_1_2099-09-10.done").exists()

    calls.clear()
    SC.run(now)
    assert calls == [], "an already-actioned window must not fire again"


def test_run_isolates_one_failing_action(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(SC, "STATE_DIR", tmp_path)
    monkeypatch.setattr(SC, "current_week", lambda now: (2099, 1))
    now = dt.datetime(2099, 9, 10, 12, 0, tzinfo=dt.UTC)
    monkeypatch.setattr(SC.P, "kickoff_windows", lambda season, week: _fake_window(now))

    def _boom():
        raise RuntimeError("budget exhausted")

    calls = []
    _patch_actions(monkeypatch, calls, kalshi=_boom)

    SC.run(now)

    assert set(calls) == {"dk", "predictions", "telegram"}, "one failure must not block the rest"
    assert "Kalshi capture: FAILED" in capsys.readouterr().out
    assert (tmp_path / "2099_1_2099-09-10.done").exists(), (
        "window still marked done even with a partial failure - matches capture "
        "scripts' own append-only, best-effort philosophy rather than retrying "
        "forever on a persistent error"
    )


def test_run_skips_windows_not_yet_due(tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "STATE_DIR", tmp_path)
    monkeypatch.setattr(SC, "current_week", lambda now: (2099, 1))
    now = dt.datetime(2099, 9, 10, 12, 0, tzinfo=dt.UTC)
    not_due = _fake_window(now, capture_offset_hours=72, kickoff_offset_hours=96,
                           game_date="2099-09-14")
    monkeypatch.setattr(SC.P, "kickoff_windows", lambda season, week: not_due)

    def _boom():
        raise AssertionError("must not fire before suggest_capture_by")

    monkeypatch.setattr(SC.K, "capture", _boom)
    SC.run(now)
    assert list(tmp_path.iterdir()) == []


def test_run_handles_the_offseason_gracefully(monkeypatch, capsys):
    monkeypatch.setattr(SC, "current_week", lambda now: None)
    SC.run(dt.datetime(2027, 3, 1, tzinfo=dt.UTC))
    assert "no current NFL week" in capsys.readouterr().out
