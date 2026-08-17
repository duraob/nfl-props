"""Tests for the prediction/bet ledger. File paths are monkeypatched to tmp_path so
these never touch the real, tracked data/odds_history/ files."""

import pandas as pd
import pytest

import ledger as L
import projections as P


@pytest.fixture
def fake_proj_df():
    return pd.DataFrame({
        "player_id": ["A", "B"],
        "season": [2099, 2099], "week": [1, 1],
        "pass_yd": [250.0, 0.0], "rush_yd": [10.0, 60.0], "rec_yd": [0.0, 20.0],
        "receptions": [0.0, 3.0],
        "pass_td": [1.5, 0.0], "rush_td": [0.1, 0.5], "rec_td": [0.0, 0.2],
        "interceptions": [0.5, 0.0],
        "confidence_yardage": [0.25, 0.30], "confidence_touchdown": [0.05, 0.06],
    })


def test_log_predictions_writes_one_row_per_stat(tmp_path, monkeypatch, fake_proj_df):
    path = tmp_path / "predictions.csv"
    monkeypatch.setattr(L, "PREDICTIONS", path)

    L.log_predictions(fake_proj_df)

    out = pd.read_csv(path, dtype={"model_version": str})
    assert len(out) == 2 * 8, "2 players x 8 tracked stats"
    row = out[(out.player_id == "B") & (out.stat == "rec_yd")].iloc[0]
    assert row.projection == pytest.approx(20.0)
    assert row.confidence == pytest.approx(0.30), "rec_yd uses yardage confidence"
    td_row = out[(out.player_id == "B") & (out.stat == "rush_td")].iloc[0]
    assert td_row.confidence == pytest.approx(0.06), "rush_td uses touchdown confidence"
    assert (out.model_version == P.MODEL_VERSION).all()


def test_log_predictions_appends_without_overwriting(tmp_path, monkeypatch, fake_proj_df):
    path = tmp_path / "predictions.csv"
    monkeypatch.setattr(L, "PREDICTIONS", path)
    L.log_predictions(fake_proj_df)
    L.log_predictions(fake_proj_df)
    assert len(pd.read_csv(path)) == 2 * 8 * 2


def test_record_bet_rejects_bad_venue(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "BETS", tmp_path / "bets.csv")
    with pytest.raises(ValueError, match="venue"):
        L.record_bet(2099, 1, "A", "rec_yd", "fanduel", "over", 50.5, -110, 10)


def test_record_bet_rejects_bad_side(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "BETS", tmp_path / "bets.csv")
    with pytest.raises(ValueError, match="side"):
        L.record_bet(2099, 1, "A", "rec_yd", "draftkings", "yes", 50.5, -110, 10)


def test_record_bet_appends_row(tmp_path, monkeypatch):
    path = tmp_path / "bets.csv"
    monkeypatch.setattr(L, "BETS", path)
    L.record_bet(2099, 1, "A", "rec_yd", "draftkings", "over", 50.5, -110, 10)
    L.record_bet(2099, 1, "B", "receptions", "kalshi", "under", 4.5, 60, 5)
    out = pd.read_csv(path)
    assert len(out) == 2
    assert list(out.columns) == L.BET_COLUMNS
    assert out.iloc[0].player_id == "A" and out.iloc[0].line == 50.5
