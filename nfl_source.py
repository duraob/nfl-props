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


def published_seasons(seasons: list[int], loader=None) -> list[int]:
    """
    Filter to seasons nflverse has actually published stats for.

    Schedules publish months in advance; per-player stats and injuries only exist
    once games have been played. So in the weeks before a season opens, season N has
    a full schedule but requesting its stats raises a 404. Projecting week 1 of an
    upcoming season is a completely normal thing to do - it just has to run entirely
    on the prior season's history - so an unstarted season is filtered out here
    rather than crashing the caller.

    This is not a silent fallback: skipped seasons are printed, and requesting
    nothing but unstarted seasons still raises.
    """
    loader = loader or (lambda season: nfl.load_player_stats(seasons=[season]))
    available, skipped = [], []
    for season in seasons:
        try:
            loader(season)
            available.append(season)
        except Exception:
            skipped.append(season)

    if skipped:
        print(f"nfl_source: no published stats yet for {skipped} "
              f"(season not started); using {available}")
    if not available:
        raise RuntimeError(
            f"None of the requested seasons {seasons} have published stats yet. "
            "Include at least one completed or in-progress season."
        )
    return available


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
    # An upcoming season has a schedule but no stats until its first games are played.
    seasons = published_seasons(seasons)

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


# nflverse is *mostly* internally consistent on team abbreviations, but not entirely:
# the 2026 roster file uses "AZ" for Arizona while every schedule (and the 2023-25
# rosters) use "ARI". Left unnormalized, Arizona silently vanishes from any
# roster-to-schedule join - 0 projections for an entire team, no error raised. That
# is the same failure shape as the old 57-player-pool bug, so it is normalized here
# rather than trusted.
TEAM_ALIASES = {"AZ": "ARI"}


def rosters(seasons: list[int], positions: list[str] | None = None,
            active_only: bool = True) -> pd.DataFrame:
    """
    Who is on which team, keyed by gsis_id so it joins straight to weekly_stats.

    Unlike stats, rosters publish before a season starts - which is what makes it
    possible to project week 1 of an upcoming season at all. `status` "ACT" is the
    active roster; RES/RET/CUT are reserve, retired, and released.

    Team abbreviations are normalized via TEAM_ALIASES - see the note there.
    """
    positions = positions or SKILL_POSITIONS
    frame = nfl.load_rosters(seasons=seasons).filter(pl.col("position").is_in(positions))
    if active_only:
        frame = frame.filter(pl.col("status") == "ACT")
    out = frame.select(
        ["season", "team", "position", "gsis_id", "full_name", "status"]
    ).drop_nulls(subset=["gsis_id"]).to_pandas()
    out["team"] = out["team"].replace(TEAM_ALIASES)
    return out


def injuries(seasons: list[int], positions: list[str] | None = None) -> pd.DataFrame:
    """
    Weekly injury reports, keyed by gsis_id so they join directly to weekly_stats.

    report_status is the game-day designation (Out / Doubtful / Questionable);
    practice_status reflects the Wed-Fri practice participation that precedes it.

    Defaults to skill positions, matching weekly_stats()/rosters(): the unfiltered
    report is mostly offensive line and defense, who never appear in weekly_stats at
    all, and would otherwise dilute any played-rate measured against it.
    """
    positions = positions or SKILL_POSITIONS
    # Like weekly_stats(): an unstarted season has no injury reports yet.
    seasons = published_seasons(seasons, loader=lambda s: nfl.load_injuries(seasons=[s]))
    columns = [
        "season", "week", "team", "gsis_id", "full_name", "position",
        "report_status", "report_primary_injury", "practice_status",
    ]
    return (nfl.load_injuries(seasons=seasons)
            .filter(pl.col("position").is_in(positions))
            .select(columns).to_pandas())


def depth_chart_ranks(season: int, positions: list[str] | None = None) -> pd.DataFrame:
    """
    Current depth-chart rank per player, keyed by gsis_id.

    nflverse's depth-chart pipeline changed schema starting with the 2025 season:
    2024 and earlier publish weekly, week-aligned snapshots (season/week/depth_team);
    2025 onward publish a single rolling "current" snapshot instead (dt/pos_rank, no
    week dimension, refreshed roughly daily) - only that newer schema is handled
    here. That's a deliberate scope limit, not an oversight: the one caller
    (projections.py's no-history prior) only ever needs "who is starting right now"
    for the week being projected, never a specific past week.
    """
    positions = positions or SKILL_POSITIONS
    dc = nfl.load_depth_charts(seasons=[season])
    if "pos_rank" not in dc.columns:
        raise RuntimeError(
            f"nfl.load_depth_charts(seasons=[{season}]) returned the pre-2025 weekly "
            "schema (no pos_rank column) - depth_chart_ranks() only supports the "
            "newer rolling-snapshot schema. See this function's docstring."
        )
    latest = dc.filter(pl.col("dt") == dc["dt"].max())
    return (latest.filter(pl.col("pos_abb").is_in(positions))
            .select(["gsis_id", "pos_abb", "pos_rank"])
            .rename({"gsis_id": "player_id", "pos_abb": "position", "pos_rank": "rank"})
            .unique(subset=["player_id"], keep="first")
            .to_pandas())
