"""
NFL data source, backed by nflverse.

Replaces the Pro-Football-Reference and ESPN Selenium scrapers. Everything here is
a thin wrapper over nflreadpy: no parsing, no retries, no bot evasion.

Two design rules matter downstream:

1. Freshness is verified, never assumed. nflverse publishes its own update stamp;
   `require_fresh()` reads it and raises rather than letting the pipeline project
   off stale numbers. Silent staleness is how the old engine shipped a 57-player
   pool for eight weeks.
2. Polars is an implementation detail. nflreadpy returns polars frames; everything
   crosses into pandas at this boundary so the rest of the project uses one
   dataframe library.
"""

from __future__ import annotations

import datetime as dt

import nflreadpy as nfl
import pandas as pd
import polars as pl
import requests

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]

_TIMESTAMP_URL = "https://github.com/nflverse/nflverse-data/releases/download/{tag}/timestamp.json"

# Columns the projection model reads. Selected rather than passed through whole so a
# silent upstream schema change surfaces here instead of three layers downstream.
_IDENTITY = ["player_id", "player_display_name", "position", "team", "season", "week", "opponent_team"]
_PASSING = [
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "passing_air_yards", "passing_epa", "passing_2pt_conversions", "sack_fumbles_lost",
]
_RUSHING = [
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles_lost",
    "rushing_2pt_conversions", "rushing_epa",
]
_RECEIVING = [
    "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
    "receiving_fumbles_lost", "receiving_2pt_conversions", "target_share",
    "air_yards_share", "receiving_epa",
]
_MISC = ["special_teams_tds"]

STAT_COLUMNS = _IDENTITY + _PASSING + _RUSHING + _RECEIVING + _MISC


def freshness(tag: str = "stats_player") -> dt.datetime:
    """
    Return when nflverse last published the given dataset.

    Reads the timestamp.json that nflverse ships alongside each release, so this
    reflects the data itself rather than when we happened to download it.
    """
    response = requests.get(_TIMESTAMP_URL.format(tag=tag), timeout=30)
    response.raise_for_status()
    stamp = response.json()["last_updated"]
    # Format is "YYYY-MM-DD HH:MM:SS TZ"; the trailing zone abbreviation is not
    # parseable by strptime, and the date is all we need for a staleness check.
    return dt.datetime.strptime(" ".join(stamp.split()[:2]), "%Y-%m-%d %H:%M:%S")


def require_fresh(max_age_days: int = 3, tag: str = "stats_player") -> dt.datetime:
    """
    Raise if nflverse data is older than max_age_days.

    Call this before generating projections during the season. In the offseason the
    upstream data legitimately goes quiet, so pass a wider window or skip the check.
    """
    updated = freshness(tag)
    age = (dt.datetime.now() - updated).days
    if age > max_age_days:
        raise RuntimeError(
            f"nflverse '{tag}' was last updated {updated:%Y-%m-%d} ({age} days ago), "
            f"exceeding the {max_age_days}-day limit. Refusing to project on stale data."
        )
    return updated


def clear_cache() -> None:
    """
    Drop nflreadpy's cache (default TTL is 24h).

    Call before a Sunday-morning run: a day-old cache would hide inactives and late
    injury news, which is exactly when freshness is worth the most.
    """
    nfl.clear_cache()


def _snap_crosswalk() -> pl.DataFrame:
    """Map gsis player_id (used by stats) to pfr_id (used by snap counts)."""
    return nfl.load_players().select(["gsis_id", "pfr_id"]).drop_nulls()


def weekly_stats(
    seasons: list[int],
    positions: list[str] | None = None,
    season_type: str = "REG",
) -> pd.DataFrame:
    """
    Weekly player stats joined to snap counts.

    This is the model's primary input. Usage columns (target_share, air_yards_share,
    offense_pct) are the predictable half of the volume x efficiency decomposition and
    are why this replaced the scraper: they were unavailable before, and snap_pct was
    absent for 56% of scraped rows versus ~6% here.

    Args:
        seasons: Seasons to load, e.g. [2023, 2024].
        positions: Positions to keep. Defaults to skill positions.
        season_type: "REG", "POST", or "ALL" for both.

    Returns:
        One row per player-week, with offense_snaps and offense_pct attached.
    """
    positions = positions or SKILL_POSITIONS

    stats = nfl.load_player_stats(seasons=seasons)
    if season_type != "ALL":
        stats = stats.filter(pl.col("season_type") == season_type)
    stats = stats.filter(pl.col("position").is_in(positions)).select(STAT_COLUMNS)

    snaps = nfl.load_snap_counts(seasons=seasons).select(
        ["pfr_player_id", "season", "week", "offense_snaps", "offense_pct"]
    )

    joined = (
        stats.join(_snap_crosswalk(), left_on="player_id", right_on="gsis_id", how="left")
        .join(
            snaps,
            left_on=["pfr_id", "season", "week"],
            right_on=["pfr_player_id", "season", "week"],
            how="left",
        )
        .drop("pfr_id")
    )
    return joined.to_pandas()


def schedule(seasons: list[int]) -> pd.DataFrame:
    """
    Game schedule with venue, weather, and historical closing lines.

    spread_line and total_line are the market's closing numbers. They are the fair-value
    benchmark for the team model and, unlike player props, come free with the schedule.
    """
    columns = [
        "game_id", "season", "week", "gameday", "gametime", "away_team", "home_team",
        "away_score", "home_score", "spread_line", "total_line", "roof", "surface",
        "temp", "wind",
    ]
    return nfl.load_schedules(seasons=seasons).select(columns).to_pandas()


def injuries(seasons: list[int]) -> pd.DataFrame:
    """
    Weekly injury reports, keyed by gsis_id so they join directly to weekly_stats.

    report_status is the game-day designation (Out / Doubtful / Questionable);
    practice_status reflects the Wed-Fri practice participation that precedes it.
    """
    columns = [
        "season", "week", "team", "gsis_id", "full_name", "position",
        "report_status", "report_primary_injury", "practice_status",
    ]
    return nfl.load_injuries(seasons=seasons).select(columns).to_pandas()
