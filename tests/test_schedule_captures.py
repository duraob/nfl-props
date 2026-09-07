"""
Tests for the kickoff-window-driven scheduler.

current_week() is checked against real 2025 schedule dates (no mocking, matching
nfl_source's own test convention) rather than synthetic data, since its whole job is
correctly reading real calendar boundaries. run()'s action-dispatch logic is tested
with monkeypatched actions and a tmp_path STATE_DIR so nothing here ever calls a real
API or touches data/scheduler_state/.
"""

import datetime as dt
import subprocess

import pandas as pd
import pytest

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


def _patch_actions(monkeypatch, calls, kalshi=None, dk=None, patch_sync=True):
    monkeypatch.setattr(SC.K, "capture", kalshi or (lambda: calls.append("kalshi")))
    monkeypatch.setattr(SC.DK, "capture", dk or (lambda within_hours: calls.append("dk")))
    monkeypatch.setattr(SC.L, "log_predictions", lambda proj: calls.append("predictions"))
    monkeypatch.setattr(SC.P, "project", lambda season, week: pd.DataFrame())
    monkeypatch.setattr(SC.R, "build_report", lambda season, week, game_date: "report text")
    monkeypatch.setattr(SC.T, "send_message", lambda text: calls.append("telegram"))
    # Patched by default so a due-window test never shells out to real git against
    # whatever the real cwd happens to be - the tests that actually want to exercise
    # sync_captured_data() for real pass patch_sync=False AND use the working_repo
    # fixture (which chdirs into a throwaway repo first).
    if patch_sync:
        monkeypatch.setattr(SC, "sync_captured_data", lambda: calls.append("sync"))


def test_run_actions_a_due_window_once(tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "STATE_DIR", tmp_path)
    monkeypatch.setattr(SC, "current_week", lambda now: (2099, 1))
    now = dt.datetime(2099, 9, 10, 12, 0, tzinfo=dt.UTC)
    monkeypatch.setattr(SC.P, "kickoff_windows", lambda season, week: _fake_window(now))

    calls = []
    _patch_actions(monkeypatch, calls)

    SC.run(now)
    assert set(calls) == {"kalshi", "dk", "predictions", "telegram", "sync"}
    for action in ("kalshi", "dk", "predictions", "report"):
        assert (tmp_path / f"2099_1_2099-09-10.{action}").exists()

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

    assert set(calls) == {"dk", "predictions", "telegram", "sync"}, "one failure must not block the rest"
    assert "Kalshi capture: FAILED" in capsys.readouterr().out
    assert not (tmp_path / "2099_1_2099-09-10.kalshi").exists(), (
        "a failed action must stay unmarked so the next tick retries it - retries "
        "are bounded by the window closing at kickoff, not unbounded"
    )
    for succeeded in ("dk", "predictions", "report"):
        assert (tmp_path / f"2099_1_2099-09-10.{succeeded}").exists()


def test_a_retry_reruns_only_the_action_that_failed(tmp_path, monkeypatch):
    """
    The credit-budget guarantee. dk_capture.py spends real API credits per event, so
    a window retried every 15 minutes across its 3-hour span would sweep DraftKings
    a dozen times and spend the month's whole budget in one afternoon. Only the
    action that actually failed may run again.
    """
    monkeypatch.setattr(SC, "STATE_DIR", tmp_path)
    monkeypatch.setattr(SC, "current_week", lambda now: (2099, 1))
    now = dt.datetime(2099, 9, 10, 12, 0, tzinfo=dt.UTC)
    monkeypatch.setattr(SC.P, "kickoff_windows", lambda season, week: _fake_window(now))

    failing = {"kalshi": True}

    def _kalshi():
        if failing["kalshi"]:
            raise RuntimeError("network down")
        calls.append("kalshi")

    calls = []
    _patch_actions(monkeypatch, calls, kalshi=_kalshi)
    SC.run(now)
    assert "dk" in calls

    failing["kalshi"] = False
    calls.clear()
    SC.run(now)
    assert calls == ["kalshi", "sync"], (
        "the retry must re-run only Kalshi - never a second paid DraftKings sweep"
    )


def test_a_failing_critical_action_alerts_once(tmp_path, monkeypatch):
    """
    require_fresh() raising mid-week takes down the prediction log and the report
    together, and a pre-kickoff prediction timestamp cannot be backfilled honestly.
    That has to be loud - but only once, not on all twelve retries.
    """
    monkeypatch.setattr(SC, "STATE_DIR", tmp_path)
    monkeypatch.setattr(SC, "current_week", lambda now: (2099, 1))
    now = dt.datetime(2099, 9, 10, 12, 0, tzinfo=dt.UTC)
    monkeypatch.setattr(SC.P, "kickoff_windows", lambda season, week: _fake_window(now))

    calls = []
    _patch_actions(monkeypatch, calls)

    def _stale(proj):
        raise RuntimeError("Refusing to project on stale data.")

    monkeypatch.setattr(SC.L, "log_predictions", _stale)
    sent = []
    monkeypatch.setattr(SC.T, "send_message", lambda text: sent.append(text))

    SC.run(now)
    assert any("log predictions failed" in m and "stale" in m for m in sent), sent
    assert (tmp_path / "2099_1_2099-09-10.predictions-alerted").exists()

    sent.clear()
    SC.run(now)
    assert not any("failed" in m for m in sent), "must not re-alert on every retry"


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


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def working_repo(tmp_path, monkeypatch):
    """A real working repo pushing to a real bare 'origin' - exercises the actual
    git plumbing rather than mocking subprocess, since the whole point of
    sync_captured_data() is that the commands genuinely work."""
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    _git("init", "--bare", str(origin), cwd=tmp_path)
    _git("init", str(work), cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=work)
    _git("config", "user.name", "Test", cwd=work)
    _git("remote", "add", "origin", str(origin), cwd=work)
    (work / "README.md").write_text("seed")
    _git("add", "README.md", cwd=work)
    _git("commit", "-m", "seed", cwd=work)
    _git("push", "-u", "origin", "HEAD:main", cwd=work)

    monkeypatch.chdir(work)
    return work


def test_sync_captured_data_commits_and_pushes_new_rows(working_repo):
    odds_dir = working_repo / "data" / "odds_history"
    odds_dir.mkdir(parents=True)
    (odds_dir / "kalshi.csv").write_text("captured_at,mid\n2099-01-01,0.5\n")

    SC.sync_captured_data()

    log = subprocess.run(["git", "log", "--oneline", "-1"], cwd=working_repo,
                         capture_output=True, text=True, check=True).stdout
    assert "capture:" in log
    status = subprocess.run(["git", "status", "--porcelain"], cwd=working_repo,
                            capture_output=True, text=True, check=True).stdout
    assert status.strip() == "", "working tree should be clean after commit"

    remote_log = subprocess.run(
        ["git", "log", "--oneline", "-1", "origin/main"], cwd=working_repo,
        capture_output=True, text=True, check=True,
    ).stdout
    assert "capture:" in remote_log, "the commit must actually reach origin"


def test_sync_captured_data_is_a_noop_with_nothing_new(working_repo):
    before = subprocess.run(["git", "log", "--oneline", "-1"], cwd=working_repo,
                            capture_output=True, text=True, check=True).stdout

    SC.sync_captured_data()

    after = subprocess.run(["git", "log", "--oneline", "-1"], cwd=working_repo,
                           capture_output=True, text=True, check=True).stdout
    assert before == after, "no changes under data/odds_history/ means no commit"


def test_run_syncs_after_a_due_window(working_repo, monkeypatch):
    monkeypatch.setattr(SC, "STATE_DIR", working_repo / "scheduler_state")
    monkeypatch.setattr(SC, "current_week", lambda now: (2099, 1))
    now = dt.datetime(2099, 9, 10, 12, 0, tzinfo=dt.UTC)
    monkeypatch.setattr(SC.P, "kickoff_windows", lambda season, week: _fake_window(now))

    def _fake_kalshi_capture():
        odds_dir = working_repo / "data" / "odds_history"
        odds_dir.mkdir(parents=True, exist_ok=True)
        (odds_dir / "kalshi.csv").write_text("captured_at,mid\n2099-01-01,0.5\n")

    calls = []
    _patch_actions(monkeypatch, calls, kalshi=_fake_kalshi_capture, patch_sync=False)

    SC.run(now)

    log = subprocess.run(["git", "log", "--oneline", "-1"], cwd=working_repo,
                         capture_output=True, text=True, check=True).stdout
    assert "capture:" in log
