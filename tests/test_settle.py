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


# Kalshi CLV. The player name must match nflverse's roster exactly - that is how
# market_odds.normalize_kalshi resolves a title to a player_id, and a near-miss is
# dropped rather than guessed.
KALSHI_TITLE = "Christian McCaffrey: 70+ receiving yards"


def _kalshi_history(path, captured_at, yes_ask):
    _write(path, [{
        "captured_at": captured_at, "series": "KXNFLRECYDS",
        "ticker": "KXNFLRECYDS-TEST", "title": KALSHI_TITLE,
        "close_time": "2025-10-05T17:00:00+00:00", "yes_bid": yes_ask - 0.02,
        "yes_ask": yes_ask, "spread": 0.02, "mid": yes_ask - 0.01,
        "yes_depth": 900, "no_depth": 900,
    }])


def _one_kalshi_bet(tmp_path, monkeypatch, bet_ts, close_ts, paid, closing_ask):
    bets_path, preds_path = tmp_path / "bets.csv", tmp_path / "predictions.csv"
    monkeypatch.setattr(S, "BETS", bets_path)
    monkeypatch.setattr(S, "PREDICTIONS", preds_path)
    monkeypatch.setattr(S, "DRAFTKINGS", tmp_path / "draftkings.csv")
    monkeypatch.setattr(S, "KALSHI_HISTORY", tmp_path / "kalshi.csv")

    _write(bets_path, [{
        "ts_utc": bet_ts, "season": REAL_SEASON, "week": REAL_WEEK,
        "player_id": REAL_PLAYER, "stat": "rec_yd", "venue": "kalshi",
        "side": "over", "line": 70.0, "price": paid, "stake": 25.0,
    }])
    _write(preds_path, [{
        "ts_utc": bet_ts, "model_version": "1.0", "season": REAL_SEASON,
        "week": REAL_WEEK, "player_id": REAL_PLAYER, "stat": "rec_yd",
        "projection": 85.0, "confidence": 0.4,
    }])
    _kalshi_history(tmp_path / "kalshi.csv", close_ts, closing_ask)
    return S.settle(REAL_SEASON, REAL_WEEK).iloc[0]


def test_kalshi_bet_gets_a_closing_price(tmp_path, monkeypatch):
    """The reason settle.py used to skip Kalshi - unparseable free-text titles - is
    not true of the weekly player props, which follow a strict template."""
    row = _one_kalshi_bet(tmp_path, monkeypatch, "2025-10-05T12:00:00+00:00",
                          "2025-10-05T16:30:00+00:00", paid=0.40, closing_ask=0.55)
    assert row.closing_price == 0.55


def test_kalshi_beat_close_is_lower_is_better(tmp_path, monkeypatch):
    """Kalshi prices are cents paid for a $1 contract, so buying BELOW the close is
    beating it - the opposite of DraftKings' American odds. Sharing one comparison
    would score every good Kalshi entry as a loss."""
    beat = _one_kalshi_bet(tmp_path, monkeypatch, "2025-10-05T12:00:00+00:00",
                           "2025-10-05T16:30:00+00:00", paid=0.40, closing_ask=0.55)
    assert beat.beat_close is True

    missed = _one_kalshi_bet(tmp_path, monkeypatch, "2025-10-05T12:00:00+00:00",
                             "2025-10-05T16:30:00+00:00", paid=0.60, closing_ask=0.55)
    assert missed.beat_close is False


def test_closing_price_captured_before_the_bet_is_flagged_stale(tmp_path, monkeypatch):
    """A closing price snapshotted before the bet is the price the bet was made at,
    so its CLV is 0.00 by construction. Every 2026 Week 1 bet was like this."""
    stale = _one_kalshi_bet(tmp_path, monkeypatch, "2025-10-05T12:00:00+00:00",
                            "2025-10-05T09:00:00+00:00", paid=0.40, closing_ask=0.40)
    assert bool(stale.close_is_stale)

    fresh = _one_kalshi_bet(tmp_path, monkeypatch, "2025-10-05T12:00:00+00:00",
                            "2025-10-05T16:30:00+00:00", paid=0.40, closing_ask=0.55)
    assert not bool(fresh.close_is_stale)
