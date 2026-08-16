"""
Tests for the nfl_source loader.

test_data_source.py checks the upstream nflverse contract; this checks that our
wrapper assembles it correctly - specifically the gsis->pfr crosswalk that attaches
snap counts, which is the one non-trivial join in the module.
"""

import datetime as dt

import pytest

import nfl_source as src

SEASON = 2024
WEEK = 10


@pytest.fixture(scope="module")
def weekly():
    return src.weekly_stats([SEASON])


def test_returns_pandas_with_expected_columns(weekly):
    import pandas as pd

    assert isinstance(weekly, pd.DataFrame), "loader must hand pandas to the rest of the project"
    for col in ("player_id", "target_share", "offense_pct", "carries", "receptions"):
        assert col in weekly.columns, f"missing model input column: {col}"


def test_snap_counts_are_attached(weekly):
    """The gsis->pfr crosswalk is the fragile step; a silent failure blanks offense_pct."""
    coverage = weekly["offense_pct"].notna().mean()
    assert coverage > 0.95, f"snap join produced only {coverage:.1%} coverage; crosswalk likely broken"


def test_full_receiver_pool_survives_loading(weekly):
    week = weekly[weekly.week == WEEK]
    receivers = (week.position == "WR").sum()
    assert receivers >= 80, f"only {receivers} WRs loaded for week {WEEK}"


def test_known_players_present_with_usage(weekly):
    week = weekly[weekly.week == WEEK]
    chase = week[week.player_display_name == "Ja'Marr Chase"]
    assert not chase.empty, "Ja'Marr Chase absent from week 10"
    assert chase.iloc[0]["target_share"] > 0, "usage signal is null for a known high-volume player"


def test_only_skill_positions_by_default(weekly):
    assert set(weekly.position.unique()) <= set(src.SKILL_POSITIONS)


def test_schedule_carries_closing_lines():
    """spread_line/total_line are the free market benchmark for the team model."""
    sched = src.schedule([SEASON])
    assert sched.spread_line.notna().mean() > 0.99
    assert sched.total_line.notna().mean() > 0.99


def test_require_fresh_rejects_stale_data(monkeypatch):
    """Staleness must raise, not warn - silent staleness is the failure we are engineering out."""
    old = dt.datetime.now() - dt.timedelta(days=30)
    monkeypatch.setattr(src, "freshness", lambda tag="stats_player": old)
    with pytest.raises(RuntimeError, match="Refusing to project on stale data"):
        src.require_fresh(max_age_days=3)
