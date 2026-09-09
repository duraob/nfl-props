"""
Tests for normalizing Kalshi/DraftKings captured lines into one shape.

Kalshi/DraftKings row fixtures below mirror real captured formats (see
market_odds.py's module docstring for how the Kalshi title template was confirmed
against real settled preseason markets). Player name matching uses real 2025 roster
data (no mocking) - Christian McCaffrey, matching the convention already used in
tests/test_settle.py.
"""

import pandas as pd
import pytest

import market_odds as M
import projections as P

REAL_PLAYER = "00-0033280"   # Christian McCaffrey
REAL_SEASON = 2025


def test_normalize_kalshi_parses_player_and_threshold_from_title():
    df = pd.DataFrame([
        {"series": "KXNFLRECYDS", "ticker": "KXNFLRECYDS-X-Y-70", "captured_at": "t1",
         "title": "Christian McCaffrey: 70+ receiving yards", "mid": 0.62,
         "yes_ask": 0.64, "spread": 0.04, "yes_depth": 1200.0},
        {"series": "KXNFLGAME", "ticker": "KXNFLGAME-X", "captured_at": "t1",
         "title": "Will Kansas City win the game?", "mid": 0.55,
         "yes_ask": 0.56, "spread": 0.02, "yes_depth": 900.0},   # unmapped series
    ])
    out = M.normalize_kalshi(df, REAL_SEASON)
    assert len(out) == 1, "only the mapped series should survive"
    row = out.iloc[0]
    assert row.player_id == REAL_PLAYER
    assert row.stat == "rec_yd"
    assert row.line == pytest.approx(70.0)
    assert row.implied_probability == pytest.approx(0.62), "mid is fair value"
    assert row.yes_ask == pytest.approx(0.64), "ask is what a buyer actually pays"
    assert row.spread == pytest.approx(0.04)
    assert row.depth == pytest.approx(1200.0)
    assert row.venue == "kalshi"


def test_normalize_kalshi_drops_unmatched_player_names():
    df = pd.DataFrame([{
        "series": "KXNFLRSHYDS", "ticker": "x", "captured_at": "t1",
        "title": "Totally Fictional Player: 50+ rushing yards", "mid": 0.4,
        "yes_ask": 0.42, "spread": 0.04, "yes_depth": 500.0,
    }])
    out = M.normalize_kalshi(df, REAL_SEASON)
    assert out.empty


def test_normalize_draftkings_devigs_symmetric_pricing():
    """-110/-110 is the standard even vig - fair probability must land exactly at
    50/50 once removed, regardless of which side the -110 landed on."""
    df = pd.DataFrame([
        {"event_id": "e1", "market": "player_receptions", "player": "Christian McCaffrey",
         "side": "Over", "line": 6.5, "price": -110, "captured_at": "t1"},
        {"event_id": "e1", "market": "player_receptions", "player": "Christian McCaffrey",
         "side": "Under", "line": 6.5, "price": -110, "captured_at": "t1"},
    ])
    out = M.normalize_draftkings(df, REAL_SEASON)
    assert len(out) == 1, "only the Over side is kept"
    assert out.iloc[0].implied_probability == pytest.approx(0.5)
    assert out.iloc[0].yes_ask == pytest.approx(110 / 210), (
        "the ask is the raw vigged price actually paid, not the de-vigged fair value"
    )
    assert out.iloc[0].spread == pytest.approx(2 * (110 / 210) - 1), "the hold"
    assert pd.isna(out.iloc[0].depth), "DraftKings publishes no resting size"
    assert out.iloc[0].venue == "draftkings"


def test_normalize_draftkings_drops_unmapped_markets():
    df = pd.DataFrame([{
        "event_id": "e1", "market": "player_rush_attempts", "player": "Christian McCaffrey",
        "side": "Over", "line": 15.5, "price": -120, "captured_at": "t1",
    }])
    assert M.normalize_draftkings(df, REAL_SEASON).empty


def test_market_lines_keeps_only_the_latest_capture(tmp_path, monkeypatch):
    kalshi_path = tmp_path / "kalshi.csv"
    pd.DataFrame([
        {"series": "KXNFLRECYDS", "ticker": "x", "captured_at": "2025-10-01T00:00:00Z",
         "title": "Christian McCaffrey: 70+ receiving yards", "mid": 0.50,
         "yes_ask": 0.52, "spread": 0.04, "yes_depth": 800.0},
        {"series": "KXNFLRECYDS", "ticker": "x", "captured_at": "2025-10-05T00:00:00Z",
         "title": "Christian McCaffrey: 70+ receiving yards", "mid": 0.65,
         "yes_ask": 0.67, "spread": 0.04, "yes_depth": 800.0},
    ]).to_csv(kalshi_path, index=False)
    monkeypatch.setattr(M, "KALSHI_HISTORY", kalshi_path)
    monkeypatch.setattr(M, "DRAFTKINGS_HISTORY", tmp_path / "draftkings.csv")  # absent

    out = M.market_lines(REAL_SEASON)

    assert len(out) == 1
    assert out.iloc[0].implied_probability == pytest.approx(0.65), "must keep the later capture"


def test_compute_edge_matches_model_probability_against_market():
    proj_df = pd.DataFrame({
        "player_id": [REAL_PLAYER], "rec_yd": [80.0], "e_targets": [8.0],
        "confidence_yardage": [0.3], "confidence_touchdown": [0.05],
    })
    market_df = pd.DataFrame({
        "player_id": [REAL_PLAYER], "stat": ["rec_yd"], "line": [60.0],
        "implied_probability": [0.5], "venue": ["kalshi"], "captured_at": ["t1"],
    })

    out = M.compute_edge(proj_df, market_df)

    assert len(out) == 1
    row = out.iloc[0]
    expected_p = P.exceed_probability(proj_df, "rec_yd", 60.0).iloc[0]
    assert row.model_probability == pytest.approx(expected_p)
    assert row.edge == pytest.approx(expected_p - 0.5)
    assert row.confidence == pytest.approx(0.3)


def test_compute_edge_ignores_unmatched_players():
    proj_df = pd.DataFrame({
        "player_id": ["someone_else"], "rec_yd": [80.0], "e_targets": [8.0],
        "confidence_yardage": [0.3], "confidence_touchdown": [0.05],
    })
    market_df = pd.DataFrame({
        "player_id": [REAL_PLAYER], "stat": ["rec_yd"], "line": [60.0],
        "implied_probability": [0.5], "venue": ["kalshi"], "captured_at": ["t1"],
    })
    assert M.compute_edge(proj_df, market_df).empty
