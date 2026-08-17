"""
Tests for the weekly scorecard.

Ledger/settle file paths are monkeypatched to tmp_path so these never touch the
real, tracked data/odds_history/ files - matching tests/test_settle.py and
tests/test_ledger.py. week_accuracy is seeded with real project() output via
ledger.log_predictions (already tested elsewhere) rather than a synthetic frame, to
exercise the real join against nflverse actuals end to end.
"""

import pandas as pd
import pytest

import ledger as L
import projections as P
import scorecard as C
import settle as S

REAL_PLAYER = "00-0033280"   # Christian McCaffrey
REAL_SEASON, REAL_WEEK = 2025, 5


def _write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_week_accuracy_scores_real_logged_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "PREDICTIONS", tmp_path / "predictions.csv")

    proj = P.project(REAL_SEASON, REAL_WEEK, seasons=[2024, 2025])
    L.log_predictions(proj)

    out = C.week_accuracy(REAL_SEASON, REAL_WEEK)

    assert not out.empty
    assert set(out.stat) <= set(P.STAT_CONFIDENCE_COLUMN)
    row = out[out.stat == "receptions"].iloc[0]
    assert row.n > 5
    assert -1.0 <= row.spearman <= 1.0


def test_week_accuracy_returns_empty_when_no_predictions_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "PREDICTIONS", tmp_path / "predictions.csv")
    assert C.week_accuracy(2099, 1).empty


def test_clv_to_date_reports_none_when_no_bets(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "BETS", tmp_path / "bets.csv")
    out = C.clv_to_date(2099)
    assert out == {"n_bets": 0, "win_rate": None, "n_with_close": 0, "beat_close_rate": None}


def test_clv_to_date_aggregates_settled_bets(tmp_path, monkeypatch):
    bets_path, preds_path, dk_path = (tmp_path / f for f in ("bets.csv", "predictions.csv", "draftkings.csv"))
    monkeypatch.setattr(S, "BETS", bets_path)
    monkeypatch.setattr(S, "PREDICTIONS", preds_path)
    monkeypatch.setattr(S, "DRAFTKINGS", dk_path)

    _write(bets_path, [
        {"ts_utc": "2025-10-05T12:00:00+00:00", "season": REAL_SEASON, "week": REAL_WEEK, "player_id": REAL_PLAYER,
         "stat": "rec_yd", "venue": "kalshi", "side": "over", "line": 40.0, "price": 50, "stake": 10},
        {"ts_utc": "2025-10-05T12:00:00+00:00", "season": REAL_SEASON, "week": REAL_WEEK, "player_id": REAL_PLAYER,
         "stat": "rec_yd", "venue": "kalshi", "side": "under", "line": 40.0, "price": 50, "stake": 10},
    ])
    _write(preds_path, [
        {"ts_utc": "2025-10-05T12:00:00+00:00", "model_version": "1.0", "season": REAL_SEASON, "week": REAL_WEEK,
         "player_id": REAL_PLAYER, "stat": "rec_yd", "projection": 75.0, "confidence": 0.25},
    ])

    out = C.clv_to_date(REAL_SEASON)

    assert out["n_bets"] == 2
    # actual was 82 receiving yards: the "over 40" bet wins, "under 40" loses.
    assert out["win_rate"] == pytest.approx(0.5)


def test_calibration_to_date_empty_when_no_bets(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "BETS", tmp_path / "bets.csv")
    assert C.calibration_to_date(2099).empty


def test_build_scorecard_handles_fully_empty_data(tmp_path, monkeypatch):
    """The real current state of the repo: no predictions logged, no bets placed
    yet. Must produce a readable message, not crash."""
    monkeypatch.setattr(L, "PREDICTIONS", tmp_path / "predictions.csv")
    monkeypatch.setattr(S, "BETS", tmp_path / "bets.csv")

    text = C.build_scorecard(2099, 1)

    assert "No graded predictions" in text
    assert "nothing settled yet" in text
    assert "not enough settled bets" in text


def test_most_recently_completed_week_matches_real_schedule():
    """Week 5 2025 ends Oct 6; week 6 doesn't start until Oct 9 - Oct 8 sits
    cleanly between them (verified via nfl_source.schedule([2025]))."""
    import datetime as dt

    result = C.most_recently_completed_week(dt.datetime(2025, 10, 8, 12, 0, tzinfo=dt.UTC))
    assert result == (2025, 5)


def test_most_recently_completed_week_is_none_before_any_games():
    import datetime as dt

    assert C.most_recently_completed_week(dt.datetime(2026, 8, 16, tzinfo=dt.UTC)) is None
