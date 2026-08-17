"""Shared fixtures. Data is fetched once per session and reused."""

import pytest

import nfl_source as src
import projections as P

FIXTURE_SEASONS = [2024]
TWO_SEASON_FIXTURE = [2024, 2025]


@pytest.fixture(scope="session")
def sample_stats():
    return src.weekly_stats(FIXTURE_SEASONS)


@pytest.fixture(scope="session")
def sample_schedule():
    return src.schedule(FIXTURE_SEASONS)


@pytest.fixture(scope="session")
def built(sample_stats, sample_schedule):
    return P.build(sample_stats, sample_schedule)


@pytest.fixture(scope="session")
def two_season_stats():
    return src.weekly_stats(TWO_SEASON_FIXTURE)


@pytest.fixture(scope="session")
def two_season_schedule():
    return src.schedule(TWO_SEASON_FIXTURE)


@pytest.fixture(scope="session")
def two_season_built(two_season_stats, two_season_schedule):
    return P.build(two_season_stats, two_season_schedule)
