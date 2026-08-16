"""Shared fixtures. Data is fetched once per session and reused."""

import pytest

import nfl_source as src
import projections as P

FIXTURE_SEASONS = [2024]


@pytest.fixture(scope="session")
def sample_stats():
    return src.weekly_stats(FIXTURE_SEASONS)


@pytest.fixture(scope="session")
def sample_schedule():
    return src.schedule(FIXTURE_SEASONS)


@pytest.fixture(scope="session")
def built(sample_stats, sample_schedule):
    return P.build(sample_stats, sample_schedule)
