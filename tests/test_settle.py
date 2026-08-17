"""
Tests for settling bets against actuals and captured closing lines.

Ledger/history file paths are monkeypatched to tmp_path so these never touch the
real, tracked data/odds_history/ files. Actuals still come from real nflverse data
(a known 2025 week 5 stat line), matching this repo's convention of testing against
real historical data rather than mocking nfl_source.
"""

import pandas as pd
import pytest

import settle as S

REAL_PLAYER = "00-0033280"   # Christian McCaffrey
REAL_SEASON, REAL_WEEK = 2025, 5
REAL_RECEIVING_YARDS = 82.0
REAL_RECEPTIONS = 8.0


def _write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_grade_over_under_and_push():
    assert S._grade(82.0, 70.5, "over") == "win"
    assert S._grade(60.0, 70.5, "over") == "loss"
    assert S._grade(60.0, 70.5, "under") == "win"
    assert S._grade(82.0, 70.5, "under") == "loss"
    assert S._grade(70.0, 70.0, "over") == "push"
    assert S._grade(float("nan"), 70.5, "over") is None


def test_settle_raises_without_a_bets_file(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "BETS", tmp_path / "bets.csv")
    with pytest.raises(RuntimeError, match="No bets recorded"):
        S.settle(2099, 1)


def test_settle_computes_error_and_result_against_real_actuals(tmp_path, monkeypatch):
    bets_path, preds_path = tmp_path / "bets.csv", tmp_path / "predictions.csv"
    monkeypatch.setattr(S, "BETS", bets_path)
    monkeypatch.setattr(S, "PREDICTIONS", preds_path)
    monkeypatch.setattr(S, "DRAFTKINGS", tmp_path / "draftkings.csv")  # does not exist

    _write(bets_path, [{
        "ts_utc": "2025-10-05T12:00:00+00:00", "season": REAL_SEASON, "week": REAL_WEEK,
        "player_id": REAL_PLAYER, "stat": "rec_yd", "venue": "kalshi", "side": "over",
        "line": 70.5, "price": 55, "stake": 10,
    }])
    _write(preds_path, [{
        "ts_utc": "2025-10-05T10:00:00+00:00", "model_version": "1.0",
        "season": REAL_SEASON, "week": REAL_WEEK, "player_id": REAL_PLAYER,
        "stat": "rec_yd", "projection": 75.0, "confidence": 0.25,
    }])

    out = S.settle(REAL_SEASON, REAL_WEEK)

    assert len(out) == 1
    row = out.iloc[0]
    assert row.actual == pytest.approx(REAL_RECEIVING_YARDS)
    assert row.projection == pytest.approx(75.0)
    assert row.error == pytest.approx(REAL_RECEIVING_YARDS - 75.0)
    assert row.result == "win"          # 82 actual > 70.5 line, side=over
    assert pd.isna(row.closing_price), "kalshi has no structured field to match on"


def test_settle_matches_draftkings_closing_price(tmp_path, monkeypatch):
    bets_path = tmp_path / "bets.csv"
    preds_path = tmp_path / "predictions.csv"
    dk_path = tmp_path / "draftkings.csv"
    monkeypatch.setattr(S, "BETS", bets_path)
    monkeypatch.setattr(S, "PREDICTIONS", preds_path)
    monkeypatch.setattr(S, "DRAFTKINGS", dk_path)

    _write(bets_path, [{
        "ts_utc": "2025-10-05T12:00:00+00:00", "season": REAL_SEASON, "week": REAL_WEEK,
        "player_id": REAL_PLAYER, "stat": "receptions", "venue": "draftkings",
        "side": "over", "line": 6.5, "price": -115, "stake": 20,
    }])
    _write(preds_path, [{
        "ts_utc": "2025-10-05T10:00:00+00:00", "model_version": "1.0",
        "season": REAL_SEASON, "week": REAL_WEEK, "player_id": REAL_PLAYER,
        "stat": "receptions", "projection": 6.0, "confidence": 0.28,
    }])
    _write(dk_path, [
        {"captured_at": "2025-10-05T15:00:00+00:00", "event_id": "e1",
         "commence_time": "x", "home_team": "x", "away_team": "x",
         "bookmaker": "draftkings", "market": "player_receptions",
         "player": "Christian McCaffrey", "side": "Over", "line": 6.5, "price": -130},
        {"captured_at": "2025-10-05T17:00:00+00:00", "event_id": "e1",  # later: closing
         "commence_time": "x", "home_team": "x", "away_team": "x",
         "bookmaker": "draftkings", "market": "player_receptions",
         "player": "Christian McCaffrey", "side": "Over", "line": 6.5, "price": -120},
    ])

    out = S.settle(REAL_SEASON, REAL_WEEK)

    row = out.iloc[0]
    assert row.actual == pytest.approx(REAL_RECEPTIONS)
    assert row.result == "win"                # 8 actual > 6.5 line
    assert row.closing_price == -120           # latest captured_at, not the earlier row
    assert row.beat_close == True              # -115 (yours) beats -120 (close)
