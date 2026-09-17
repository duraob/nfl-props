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

import backtest as B
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
    assert out == {"n_bets": 0, "n_graded": 0, "n_wins": 0, "win_rate": None,
                   "n_with_close": 0, "n_stale_close": 0, "beat_close_rate": None}


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


def test_screen_calibration_empty_when_nothing_recommended(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "RECOMMENDATIONS", tmp_path / "recommendations.csv")
    assert C.screen_calibration(2099).empty


def test_build_scorecard_handles_fully_empty_data(tmp_path, monkeypatch):
    """The real current state of the repo: no predictions logged, no bets placed
    yet. Must produce a readable message, not crash."""
    monkeypatch.setattr(L, "PREDICTIONS", tmp_path / "predictions.csv")
    monkeypatch.setattr(L, "RECOMMENDATIONS", tmp_path / "recommendations.csv")
    monkeypatch.setattr(S, "BETS", tmp_path / "bets.csv")

    text = C.build_scorecard(2099, 1)

    assert "No graded predictions" in text
    assert "nothing settled yet" in text
    assert "no graded recommendations yet" in text


def test_most_recently_completed_week_matches_real_schedule():
    """Week 5 2025 ends Oct 6; week 6 doesn't start until Oct 9 - Oct 8 sits
    cleanly between them (verified via nfl_source.schedule([2025]))."""
    import datetime as dt

    result = C.most_recently_completed_week(dt.datetime(2025, 10, 8, 12, 0, tzinfo=dt.UTC))
    assert result == (2025, 5)


def test_most_recently_completed_week_is_none_before_any_games():
    import datetime as dt

    assert C.most_recently_completed_week(dt.datetime(2026, 8, 16, tzinfo=dt.UTC)) is None


def test_week_accuracy_grades_each_stat_only_for_players_whose_role_includes_it(
        tmp_path, monkeypatch):
    """
    The regression this guards is subtle and was live for all of Week 1: without a
    role filter, every logged player is graded on all eight stats, because
    weekly_stats gives a player who played a row with zeros in the stats outside his
    role. Receivers then get scored on passing yards against an actual of 0.

    The visible symptom was an identical n on every stat - so that is what this
    asserts. Ranking quarterbacks above receivers in passing yards is free, and it
    inflated Week 1's reported pass_yd rank correlation from 0.02 to 0.46.
    """
    import nfl_source as src

    season, week = 2025, 5
    played = src.weekly_stats([season])
    played = played[played.week == week]

    # A ledger covering everyone who played, so any difference in n between stats
    # can only come from the role filter.
    preds = pd.DataFrame([
        {"ts_utc": "2025-10-05T12:00:00+00:00", "model_version": "1.0",
         "season": season, "week": week, "player_id": pid, "stat": stat,
         "projection": 1.0, "confidence": 0.4}
        for pid in played.player_id.unique()
        for stat, _, _ in B.STATS
    ])
    preds_path = tmp_path / "predictions.csv"
    preds.to_csv(preds_path, index=False)
    monkeypatch.setattr(L, "PREDICTIONS", preds_path)

    accuracy = C.week_accuracy(season, week).set_index("stat")
    assert accuracy.n.nunique() > 1, "identical n across stats means the role filter is gone"
    assert accuracy.loc["pass_yd", "n"] < accuracy.loc["receptions", "n"]


# Christian McCaffrey, 2025 week 5: 8 receptions, 82 receiving yards.
def _recommendation(**over):
    base = {"ts_utc": "2025-10-05T12:00:00+00:00", "season": REAL_SEASON,
            "week": REAL_WEEK, "game_date": "2025-10-05", "slot": 1,
            "player_id": REAL_PLAYER, "stat": "receptions", "venue": "kalshi",
            "line": 5.0, "yes_ask": 0.45, "model_probability": 0.60,
            "implied_probability": 0.44, "edge_at_ask": 0.15, "spread": 0.02,
            "depth": 900.0}
    base.update(over)
    return base


def test_screen_accuracy_grades_every_call_not_just_backed_ones(tmp_path, monkeypatch):
    """bets.csv can never grade the screen - it only contains calls that were
    backed, so it cannot say whether the screen's own bounds are right."""
    monkeypatch.setattr(L, "RECOMMENDATIONS", tmp_path / "recommendations.csv")
    _write(L.RECOMMENDATIONS, [
        _recommendation(line=5.0),                            # 8 receptions -> cleared
        _recommendation(stat="rec_yd", line=70.0, slot=2),    # 82 yards -> cleared
        _recommendation(stat="rush_yd", line=500.0, slot=3),  # -> did not clear
    ])

    out = C.screen_accuracy(REAL_SEASON, REAL_WEEK)

    assert out["n"] == 3
    assert out["cleared"] == 2
    assert out["actual_rate"] == pytest.approx(2 / 3)


def test_screen_accuracy_reports_whether_model_or_market_was_closer(tmp_path, monkeypatch):
    """The gap that decides whether an edge was real: a bet only pays when the model
    is nearer the truth than the price is."""
    monkeypatch.setattr(L, "RECOMMENDATIONS", tmp_path / "recommendations.csv")
    # Both calls clear, so the actual rate is 100% - the model's 90% is nearer that
    # than the market's 20%.
    _write(L.RECOMMENDATIONS, [
        _recommendation(line=1.0, model_probability=0.9, implied_probability=0.2),
        _recommendation(stat="rec_yd", line=2.0, slot=2,
                        model_probability=0.9, implied_probability=0.2),
    ])
    assert C.screen_accuracy(REAL_SEASON, REAL_WEEK)["closer"] == "model"

    _write(L.RECOMMENDATIONS, [
        _recommendation(line=99.0, model_probability=0.9, implied_probability=0.2),
        _recommendation(stat="rec_yd", line=998.0, slot=2,
                        model_probability=0.9, implied_probability=0.2),
    ])
    assert C.screen_accuracy(REAL_SEASON, REAL_WEEK)["closer"] == "market"


def test_a_call_resent_on_a_later_sheet_is_counted_once(tmp_path, monkeypatch):
    """A player plays once a week, so the same (player, stat) appearing on two
    sheets is one call re-sent, not two. Double-counting would silently weight
    whichever players happened to span several game days."""
    monkeypatch.setattr(L, "RECOMMENDATIONS", tmp_path / "recommendations.csv")
    _write(L.RECOMMENDATIONS, [
        _recommendation(ts_utc="2025-10-01T12:00:00+00:00", line=5.0),
        _recommendation(ts_utc="2025-10-04T12:00:00+00:00", line=6.0),
    ])

    out = C.screen_accuracy(REAL_SEASON, REAL_WEEK)

    assert out["n"] == 1
    assert out["cleared"] == 1, "the later sheet's line (6.0) is the one that stood"


def test_screen_calibration_uses_the_probability_as_logged(tmp_path, monkeypatch):
    """Recomputing it now would grade today's model against last week's decision."""
    monkeypatch.setattr(L, "RECOMMENDATIONS", tmp_path / "recommendations.csv")
    _write(L.RECOMMENDATIONS, [
        _recommendation(line=5.0, model_probability=0.75),                     # cleared
        _recommendation(stat="rush_yd", line=500.0, slot=2, model_probability=0.75),
    ])

    table = C.screen_calibration(REAL_SEASON, n_bins=1)

    assert len(table) == 1
    assert table.iloc[0].predicted == pytest.approx(0.75)
    assert table.iloc[0].actual_rate == pytest.approx(0.5)
    assert int(table.iloc[0].n) == 2
