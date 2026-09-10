"""
Tests for the projection engine.

The most important test here is test_no_future_leakage. A backtest that leaks future
data reports accuracy the live system will never reproduce, which is a worse failure
than being inaccurate — it is being inaccurate while looking correct.
"""

import numpy as np
import pandas as pd
import pytest

import nfl_source as src
import projections as P

# Core projected stats, paired with their raw-stat counterpart in weekly_stats output.
CORE_STATS = [
    ("pass_yd", "passing_yards"),
    ("rush_yd", "rushing_yards"),
    ("rec_yd", "receiving_yards"),
    ("receptions", "receptions"),
]


def test_shrinkage_pulls_noisy_signals_to_the_prior():
    """
    TD rate measured 0.09 reliability, so a hot player's own rate should move the
    estimate only slightly. The old engine shrank TDs 15% - backwards.
    """
    observed = pd.Series([0.50])          # wildly above league rate
    prior = 0.045
    out = P._shrink(observed, prior, P.MEASURED_RELIABILITY["td_rate"]).iloc[0]
    assert out < 0.10, f"TD rate insufficiently shrunk: {out}"
    # usage, by contrast, should be trusted nearly at face value
    usage = P._shrink(pd.Series([10.0]), 5.0, P.MEASURED_RELIABILITY["usage"]).iloc[0]
    assert usage > 9.0


def test_shrinkage_handles_missing_observations():
    out = P._shrink(pd.Series([np.nan]), 4.0, 0.9).iloc[0]
    assert out == pytest.approx(4.0), "a player with no history should sit at the prior"


def test_no_future_leakage(sample_stats, sample_schedule):
    """
    Corrupting a late week must not change projections for earlier weeks.

    This is the structural guard against a backtest that grades itself on data the
    live system would not have had.
    """
    base = P.build(sample_stats, sample_schedule)
    late = sample_stats.week.max()

    corrupted = sample_stats.copy()
    mask = corrupted.week == late
    for col in ("targets", "receiving_yards", "receiving_tds", "carries", "rushing_yards"):
        corrupted.loc[mask, col] = corrupted.loc[mask, col] * 100 + 500

    after = P.build(corrupted, sample_schedule)
    key = ["player_id", "season", "week"]
    for proj_col, _ in CORE_STATS:
        earlier = base[base.week < late].set_index(key)[proj_col]
        after_earlier = after[after.week < late].set_index(key)[proj_col]
        joined = pd.concat([earlier, after_earlier], axis=1, keys=["before", "after"]).dropna()
        assert len(joined) > 100, "not enough overlapping rows to make the test meaningful"
        assert np.allclose(joined.before, joined.after), (
            f"future data changed a past {proj_col} projection"
        )


def test_pool_is_not_truncated(built):
    """The failure this project was rebuilt around: a pool with no wide receivers."""
    week = built[built.week == 10]
    assert len(week) >= 200, f"only {len(week)} players projected"
    assert (week.position == "WR").sum() >= 60, "receivers missing from the pool"


def test_star_receivers_are_projected(built):
    week = built[built.week == 10]
    names = set(week.player_display_name)
    for star in ["Ja'Marr Chase", "CeeDee Lamb", "Amon-Ra St. Brown"]:
        assert star in names, f"{star} absent from projections"


def test_projections_are_not_systematically_biased(built, sample_stats):
    """The old engine reportedly ran 2-4x high on fantasy-relevant yardage. Check
    each core stat directly rather than through a composite score."""
    actuals = sample_stats[["player_id", "season", "week"] + [a for _, a in CORE_STATS]]
    merged = built.merge(actuals, on=["player_id", "season", "week"], how="left",
                         suffixes=("", "_obs"))
    for proj_col, actual_col in CORE_STATS:
        obs = merged[actual_col + "_obs"] if actual_col == proj_col else merged[actual_col]
        bias = (merged[proj_col] - obs).mean()
        # Yardage stats tolerate a couple of yards of drift; receptions is on a much
        # smaller scale so its tolerance is tighter.
        limit = 0.5 if proj_col == "receptions" else 3.0
        assert abs(bias) < limit, f"{proj_col} bias of {bias:+.2f} exceeds {limit}"


def test_beats_naive_baseline(built, sample_stats):
    """backtest.py grades this properly across every stat; here we only assert the
    model earns its place on the two stats that matter most for the model's edge
    thesis - usage-driven volume (receptions) and its related yardage (rec_yd)."""
    stats_df = sample_stats.sort_values(["player_id", "season", "week"]).copy()
    for proj_col, actual_col in [("receptions", "receptions"), ("rec_yd", "receiving_yards")]:
        baseline_col = f"_bl_{proj_col}"
        stats_df[baseline_col] = (
            stats_df.groupby(["player_id", "season"], sort=False)[actual_col]
            .transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean())
        )
        m = built.merge(stats_df[["player_id", "season", "week", baseline_col, actual_col]],
                        on=["player_id", "season", "week"], how="left",
                        suffixes=("", "_obs")).dropna(subset=[baseline_col])
        obs = m[actual_col + "_obs"] if actual_col == proj_col else m[actual_col]

        model_mae = np.abs(m[proj_col] - obs).mean()
        base_mae = np.abs(m[baseline_col] - obs).mean()
        assert model_mae < base_mae, (
            f"{proj_col}: model MAE {model_mae:.3f} worse than baseline {base_mae:.3f}"
        )


def test_early_season_weeks_use_prior_season_history(two_season_built):
    """
    Weeks 1-2 of a season have zero SAME-season prior games. Without carrying a
    player's history across the season boundary, MIN_GAMES filters them out entirely
    - the exact defect that made 2025 week 1 return 0 projections before this was
    fixed. Confirms it stays fixed.
    """
    week1 = two_season_built[(two_season_built.season == 2025) & (two_season_built.week == 1)]
    assert len(week1) > 200, f"week 1 returned only {len(week1)} rows - cross-season history is broken"

    week2 = two_season_built[(two_season_built.season == 2025) & (two_season_built.week == 2)]
    assert len(week2) > 200, f"week 2 returned only {len(week2)} rows"


def test_confidence_is_bounded_by_measured_reliability(two_season_built):
    """
    Confidence must never exceed the split-half reliability ceiling for its tier,
    regardless of sample size - a projection cannot be more trustworthy than the
    underlying stat was measured to be.
    """
    assert two_season_built.confidence_yardage.max() <= P.MEASURED_RELIABILITY["efficiency"] + 1e-9
    assert two_season_built.confidence_touchdown.max() <= P.MEASURED_RELIABILITY["td_rate"] + 1e-9


def test_touchdown_confidence_never_reaches_high(two_season_built):
    """
    TD/INT reliability measured 0.09 - below any reasonable 'high' bar. If touchdown
    confidence ever shows 'high', the label thresholds and the measured reliability
    have drifted out of sync.
    """
    assert (two_season_built.confidence_touchdown_label != "high").all()


def test_confidence_ramps_with_games_played(two_season_built):
    """MIN_GAMES=2 means games_played < 2 never appears in build()'s output at all -
    compare the earliest included bucket against a full-sample one instead."""
    d = two_season_built
    early = d[d.games_played == P.MIN_GAMES].confidence_yardage.mean()
    full = d[d.games_played >= P.CONFIDENCE_FULL_SAMPLE_GAMES].confidence_yardage.mean()
    assert full > early, "confidence should be higher with more trailing games"


def test_kickoff_windows_separates_distinct_game_days():
    """2026 week 1 spans Wednesday, Thursday, Sunday, and Monday - a fixed weekly
    capture schedule cannot correctly time all of them."""
    windows = P.kickoff_windows(2026, 1)
    assert windows.game_date.nunique() >= 3, "expected multiple distinct game days"
    assert (windows.suggest_capture_by < windows.earliest_kickoff).all()


def test_upcoming_week_is_projectable():
    """
    The actual production use case: projecting a week that has not been played.
    build() is retrospective and returns nothing for a future week on its own, so
    project() constructs placeholder rows from the roster + schedule. Before this
    existed, P.project(2026, 1) raised a raw 404 and then returned zero rows.
    """
    proj = P.project(2026, 1)
    assert len(proj) > 300, f"only {len(proj)} players projected for an upcoming week"
    assert proj.implied_total.notna().all(), "Vegas totals should attach to every row"


def test_every_scheduled_team_gets_projections():
    """
    Guards the ARI/AZ class of bug: nflverse's 2026 roster file abbreviates Arizona
    'AZ' while every schedule uses 'ARI', which silently dropped one entire team's
    players with no error. nfl_source.TEAM_ALIASES normalizes it; this confirms no
    team is ever silently missing.
    """
    import nfl_source as src
    proj = P.project(2026, 1)
    sched = src.schedule([2026])
    week1 = sched[(sched.season == 2026) & (sched.week == 1)]
    scheduled = set(week1.home_team) | set(week1.away_team)
    assert scheduled - set(proj.team) == set(), "a scheduled team has no projected players"


def test_projections_are_never_negative(two_season_built):
    """Real single-game yardage can be negative; an expectation for a bet cannot."""
    cols = ["pass_yd", "rush_yd", "rec_yd", "receptions", "pass_td", "rush_td", "rec_td"]
    assert (two_season_built[cols] >= 0).all().all(), "negative projection present"


def test_roster_team_abbreviations_match_the_schedule():
    """Catches a new upstream abbreviation drift before it silently drops a team."""
    import nfl_source as src
    roster_teams = set(src.rosters([2026], active_only=False).team)
    sched = src.schedule([2026])
    sched_teams = set(sched[sched.season == 2026].home_team) | set(sched[sched.season == 2026].away_team)
    assert sched_teams - roster_teams == set(), (
        f"schedule teams absent from roster feed: {sorted(sched_teams - roster_teams)} "
        "- add to nfl_source.TEAM_ALIASES"
    )


def test_project_raises_on_stale_data_for_an_incomplete_season(monkeypatch):
    """
    A season that's started but not yet fully complete - i.e. genuinely live right
    now - must still block on stale data. Simulated by wiping 2025's scores to NaN
    (2025 itself is long over in reality, so this is the only way to construct a
    "started but not complete" case without waiting for a real one).
    """
    import datetime as dt
    import nfl_source as src
    old = dt.datetime.now() - dt.timedelta(days=30)
    monkeypatch.setattr(src, "freshness", lambda tag="stats_player": old)

    incomplete = src.schedule([2024, 2025]).copy()
    incomplete.loc[incomplete.season == 2025, "home_score"] = np.nan
    monkeypatch.setattr(src, "schedule", lambda seasons: incomplete)

    with pytest.raises(RuntimeError, match="Refusing to project on stale data"):
        P.project(2025, 5)


def test_project_skips_freshness_check_for_a_completed_season(monkeypatch):
    """
    2025 is fully over - every game has a final score, so nothing will ever publish
    for it again. A stale live feed (which only ever reflects the current, possibly
    still-in-progress season) must not block a retrospective query against it.
    """
    import datetime as dt
    import nfl_source as src
    old = dt.datetime.now() - dt.timedelta(days=30)
    monkeypatch.setattr(src, "freshness", lambda tag="stats_player": old)
    proj = P.project(2025, 5)
    assert not proj.empty


def test_project_skips_freshness_check_for_an_unstarted_season(monkeypatch):
    """
    A season with nothing published yet has nothing current to go stale, so the check
    must not fire - this is the ordinary "project next week in advance" case.

    Simulated by hiding 2025's stats, following the sibling tests above, rather than
    pointing at whichever real season happens to be unstarted today. This test used
    to target 2026 Week 1 for real and silently changed meaning the moment that week
    was actually played - a calendar-dependent test asserts a different thing every
    time the calendar moves.
    """
    import datetime as dt
    import nfl_source as src
    old = dt.datetime.now() - dt.timedelta(days=30)
    monkeypatch.setattr(src, "freshness", lambda tag="stats_player": old)

    stats = src.weekly_stats([2024, 2025])
    monkeypatch.setattr(src, "weekly_stats", lambda seasons: stats[stats.season != 2025])

    proj = P.project(2025, 1)
    assert len(proj) > 300


def test_apply_injury_status_excludes_out_and_discounts_questionable(monkeypatch):
    import nfl_source as src
    week_rows = pd.DataFrame({
        "player_id": ["A", "B", "C", "D"],
        "e_targets": [10.0, 8.0, 6.0, 4.0], "e_carries": [0.0] * 4, "e_attempts": [0.0] * 4,
        "pass_yd": [0.0] * 4, "rush_yd": [0.0] * 4,
        "rec_yd": [80.0, 60.0, 40.0, 20.0], "receptions": [6.0, 5.0, 4.0, 2.0],
        "pass_td": [0.0] * 4, "rush_td": [0.0] * 4,
        "rec_td": [0.5, 0.4, 0.3, 0.1], "interceptions": [0.0] * 4,
    })
    fake_injuries = pd.DataFrame({
        "season": [2099, 2099, 2099], "week": [1, 1, 1],
        "gsis_id": ["A", "B", "C"], "report_status": ["Out", "Doubtful", "Questionable"],
    })
    monkeypatch.setattr(src, "injuries", lambda seasons: fake_injuries)

    out = P._apply_injury_status(week_rows, 2099, 1)

    assert set(out.player_id) == {"C", "D"}, "Out/Doubtful should be dropped"
    c_row = out[out.player_id == "C"].iloc[0]
    assert c_row.rec_yd == pytest.approx(40.0 * P.INJURY_DISCOUNT["Questionable"])
    d_row = out[out.player_id == "D"].iloc[0]
    assert d_row.rec_yd == pytest.approx(20.0), "player with no injury tag should be unaffected"


def test_project_excludes_out_players_for_a_started_season():
    """Real check against real injury data: no player ruled Out for a played week
    should appear in that week's projection."""
    import nfl_source as src
    proj = P.project(2025, 5)
    week5 = src.injuries([2025])
    out_ids = set(week5[(week5.week == 5) & (week5.report_status == "Out")].gsis_id)
    assert not (set(proj.player_id) & out_ids), "an Out-designated player appeared in the projection"


def test_apply_rookie_prior_rescales_only_no_history_rows_with_a_depth_chart_entry(monkeypatch):
    import nfl_source as src
    week_rows = pd.DataFrame({
        "player_id": ["VET", "ROOKIE_RANKED", "ROOKIE_UNRANKED"],
        "position": ["WR", "WR", "WR"],
        "games_played": [10, 0, 0],
        "e_targets": [8.0, 4.5, 4.5], "e_carries": [0.1, 0.1, 0.1], "e_attempts": [0.0, 0.0, 0.0],
        "pass_yd": [0.0, 0.0, 0.0], "rush_yd": [0.5, 0.3, 0.3],
        "rec_yd": [70.0, 35.0, 35.0], "receptions": [5.0, 3.0, 3.0],
        "pass_td": [0.0, 0.0, 0.0], "rush_td": [0.0, 0.0, 0.0],
        "rec_td": [0.4, 0.2, 0.2], "interceptions": [0.0, 0.0, 0.0],
    })
    fake_ranks = pd.DataFrame({"player_id": ["ROOKIE_RANKED"], "position": ["WR"], "rank": [1]})
    monkeypatch.setattr(src, "depth_chart_ranks", lambda season: fake_ranks)

    out = P._apply_rookie_prior(week_rows, 2099)

    assert set(out.player_id) == {"VET", "ROOKIE_RANKED"}, (
        "no-history player with no depth-chart entry should still be dropped"
    )
    vet = out[out.player_id == "VET"].iloc[0]
    assert vet.rec_yd == pytest.approx(70.0), "player with real history should be untouched"
    rookie = out[out.player_id == "ROOKIE_RANKED"].iloc[0]
    wr_tier1 = P.DEPTH_RANK_VOLUME_PRIOR.query("position == 'WR' and tier == 1").iloc[0]
    assert rookie.e_targets == pytest.approx(wr_tier1.e_targets)
    assert rookie.rec_yd == pytest.approx(35.0 * wr_tier1.e_targets / 4.5)


def test_project_covers_no_history_players_for_an_upcoming_week():
    """
    The actual production gap this closes: ~400 of ~900 rostered players (every
    rookie among them) previously vanished below MIN_GAMES with no projection at
    all. project(2026, 1) should now include a meaningful share of them, always at
    low confidence.
    """
    proj = P.project(2026, 1)
    no_history = proj[proj.games_played < P.MIN_GAMES]
    assert len(no_history) > 100, f"only {len(no_history)} no-history players covered"
    assert (no_history.confidence_yardage_label == "low").all()
    assert (no_history.e_targets + no_history.e_carries + no_history.e_attempts > 0).any()


@pytest.fixture
def fake_proj():
    return pd.DataFrame({
        "rec_yd": [80.0, 80.0, 0.0],
        "e_targets": [8.0, 8.0, 0.05],   # third row below DISPERSION_VOLUME_FLOOR
    })


def test_exceed_probability_rejects_unsupported_stat(fake_proj):
    with pytest.raises(ValueError, match="No calibrated dispersion"):
        P.exceed_probability(fake_proj, "pass_td", 1.5)


def test_exceed_probability_is_monotonic_in_threshold(fake_proj):
    low = P.exceed_probability(fake_proj, "rec_yd", 40.0)
    high = P.exceed_probability(fake_proj, "rec_yd", 120.0)
    assert (low.iloc[:2] > high.iloc[:2]).all(), "a higher threshold must be less likely"
    assert 0.4 < low.iloc[0] < 1.0, "40 well below an 80 projection should be likely"
    assert 0.0 < high.iloc[0] < 0.4, "120 well above an 80 projection should be unlikely"


def test_exceed_probability_masks_rows_below_the_volume_floor(fake_proj):
    out = P.exceed_probability(fake_proj, "rec_yd", 40.0)
    assert pd.isna(out.iloc[2]), "no target volume - nobody would query this threshold"


def test_exceed_probability_skip_volume_check_bypasses_the_mask(fake_proj):
    out = P.exceed_probability(fake_proj, "rec_yd", 40.0, skip_volume_check=True)
    assert not pd.isna(out.iloc[2]), "explicit opt-out should skip the volume-floor mask"


def test_exceed_probability_accepts_a_per_row_threshold(fake_proj):
    thresholds = pd.Series([40.0, 120.0, 40.0])
    out = P.exceed_probability(fake_proj, "rec_yd", thresholds)
    assert out.iloc[0] > out.iloc[1], "same projection, different threshold per row"


def test_a_partially_played_week_still_projects_its_remaining_games():
    """
    An NFL week spans Wednesday to Monday, so "has this week been played" is not a
    single fact. Testing it as one boolean meant the first game's stats publishing
    stripped placeholders from every remaining game in the same week: 2026 Week 1
    collapsed from 493 players to the 23 who played Wednesday, and the Thursday,
    Sunday and Monday reports each returned "no games" instead of erroring - the
    silent-empty failure shape this codebase keeps having to design against.

    Real schedule data, synthetic stats: only a real schedule has the Wed/Thu/Sun/Mon
    spread that makes this fail, and only synthetic stats can put one game in the
    past while the rest are still upcoming.
    """
    schedule_df = src.schedule([2026])
    week1 = schedule_df[(schedule_df.season == 2026) & (schedule_df.week == 1)]
    opener = week1.sort_values("gameday").iloc[0]
    played = pd.DataFrame({
        "player_id": ["played-1", "played-2"],
        "team": [opener.home_team, opener.away_team],
        "season": [2026, 2026],
        "week": [1, 1],
    })

    out = P._placeholder_rows(2026, 1, played, schedule_df)

    teams = set(out.team)
    assert opener.home_team not in teams, "a played game must not be re-projected"
    assert opener.away_team not in teams
    expected = set(week1.home_team) | set(week1.away_team)
    expected -= {opener.home_team, opener.away_team}
    assert teams == expected, (
        "every team whose game has not kicked off yet must still get placeholders"
    )


def test_a_fully_played_week_produces_no_placeholders():
    """The other side of the same rule: once every game has stats, build()'s real
    rows are the projection and synthetic ones would double-count."""
    schedule_df = src.schedule([2026])
    week1 = schedule_df[(schedule_df.season == 2026) & (schedule_df.week == 1)]
    all_teams = sorted(set(week1.home_team) | set(week1.away_team))
    played = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(len(all_teams))],
        "team": all_teams,
        "season": 2026,
        "week": 1,
    })
    assert P._placeholder_rows(2026, 1, played, schedule_df).empty
