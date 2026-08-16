"""
Guards the nflverse data contract.

The projection model is built on usage share (targets, carries, snaps), so these
columns are load-bearing. nflverse is an upstream project that can rename or drop
fields between releases; this test fails loudly if that happens instead of letting
a silently-empty column produce quiet garbage downstream.
"""

import polars as pl
import pytest

import nflreadpy as nfl

SEASON = 2024
WEEK = 10

# Columns the projection engine cannot work without.
REQUIRED_STAT_COLS = {
    "player_id", "player_display_name", "position", "team", "season", "week",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "carries", "rushing_yards", "rushing_tds",
    "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "target_share",
}

REQUIRED_SNAP_COLS = {"player", "position", "team", "season", "week", "offense_snaps", "offense_pct"}

SKILL_POSITIONS = ["QB", "RB", "WR", "TE", "FB"]


@pytest.fixture(scope="module")
def player_stats():
    return nfl.load_player_stats(seasons=[SEASON]).filter(pl.col("season_type") == "REG")


@pytest.fixture(scope="module")
def snap_counts():
    return nfl.load_snap_counts(seasons=[SEASON])


def test_player_stats_has_required_columns(player_stats):
    missing = REQUIRED_STAT_COLS - set(player_stats.columns)
    assert not missing, f"nflverse player_stats is missing required columns: {sorted(missing)}"


def test_snap_counts_has_required_columns(snap_counts):
    missing = REQUIRED_SNAP_COLS - set(snap_counts.columns)
    assert not missing, f"nflverse snap_counts is missing required columns: {sorted(missing)}"


def test_star_receivers_are_present(player_stats):
    """The old pipeline silently dropped every top WR. Never again."""
    stars = ["A.J. Brown", "Ja'Marr Chase", "Justin Jefferson", "CeeDee Lamb", "Amon-Ra St. Brown"]
    week = player_stats.filter(pl.col("week") == WEEK)
    found = set(week["player_display_name"].to_list())
    missing = [s for s in stars if s not in found]
    assert not missing, f"Star receivers absent from week {WEEK}: {missing}"


def test_weekly_pool_is_full_size(player_stats):
    """A real NFL week has ~100+ WRs with a stat line, not a handful."""
    week = player_stats.filter(pl.col("week") == WEEK)
    wrs = week.filter(pl.col("position") == "WR").height
    assert wrs >= 80, f"Only {wrs} WRs in week {WEEK}; expected >= 80"


def test_snap_coverage_for_skill_positions(snap_counts):
    """
    Root cause of the old 57-player pool: snap_pct was absent for ~56% of scraped
    rows. nflverse should be an order of magnitude better for skill positions.
    """
    off = snap_counts.filter(pl.col("position").is_in(SKILL_POSITIONS))
    blank = off.filter(pl.col("offense_pct").is_null() | (pl.col("offense_pct") == 0)).height
    ratio = blank / off.height
    assert ratio < 0.15, f"{ratio:.1%} of skill-position rows lack snap pct; expected < 15%"


def test_target_share_is_populated(player_stats):
    """target_share is the core usage signal; it must not be mostly null."""
    wrs = player_stats.filter((pl.col("position") == "WR") & (pl.col("targets") > 0))
    null_ratio = wrs.filter(pl.col("target_share").is_null()).height / wrs.height
    assert null_ratio < 0.05, f"target_share null for {null_ratio:.1%} of targeted WRs"
