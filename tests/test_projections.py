"""
Tests for the projection engine.

The most important test here is test_no_future_leakage. A backtest that leaks future
data reports accuracy the live system will never reproduce, which is a worse failure
than being inaccurate — it is being inaccurate while looking correct.
"""

import numpy as np
import pandas as pd
import pytest

import projections as P

STAT_COLUMNS = [
    "passing_yards", "passing_tds", "passing_interceptions", "passing_2pt_conversions",
    "rushing_yards", "rushing_tds", "rushing_2pt_conversions", "rushing_fumbles_lost",
    "receptions", "receiving_yards", "receiving_tds", "receiving_2pt_conversions",
    "receiving_fumbles_lost", "sack_fumbles_lost", "special_teams_tds",
]


def _blank(**kwargs) -> pd.DataFrame:
    row = {c: 0.0 for c in STAT_COLUMNS}
    row.update(kwargs)
    return pd.DataFrame([row])


def test_dk_scoring_matches_hand_calculation():
    """8 rec + 100 yds + 1 TD = 8 + 10 + 6, plus the 100-yard bonus = 27."""
    line = _blank(receptions=8, receiving_yards=100, receiving_tds=1)
    assert P.dk_points(line).iloc[0] == pytest.approx(27.0)


def test_dk_yardage_bonus_is_a_step_not_a_slope():
    """99 yards earns no bonus; 100 earns exactly 3."""
    just_under = P.dk_points(_blank(rushing_yards=99)).iloc[0]
    exactly = P.dk_points(_blank(rushing_yards=100)).iloc[0]
    assert exactly - just_under == pytest.approx(3.0 + 0.1)


def test_dk_negative_scoring():
    line = _blank(passing_interceptions=2, rushing_fumbles_lost=1)
    assert P.dk_points(line).iloc[0] == pytest.approx(-3.0)


def test_bonus_probability_credits_near_misses():
    """
    A player projected below the threshold still clears it sometimes. Treating the
    bonus as a step function on the mean would bias every projection low.
    """
    mean = pd.Series([85.0])
    p = P._exceed_probability(mean, 100.0, k=P.STAT_DISPERSION["rec_yd"]).iloc[0]
    assert 0.05 < p < 0.6, f"implausible exceed probability {p}"


def test_bonus_probability_increases_with_mean():
    k = P.STAT_DISPERSION["rec_yd"]
    lo = P._exceed_probability(pd.Series([50.0]), 100.0, k).iloc[0]
    hi = P._exceed_probability(pd.Series([120.0]), 100.0, k).iloc[0]
    assert hi > lo


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
    earlier = base[base.week < late].set_index(["player_id", "season", "week"])["dk_points"]
    after_earlier = after[after.week < late].set_index(["player_id", "season", "week"])["dk_points"]
    joined = pd.concat([earlier, after_earlier], axis=1, keys=["before", "after"]).dropna()

    assert len(joined) > 100, "not enough overlapping rows to make the test meaningful"
    assert np.allclose(joined.before, joined.after), "future data changed a past projection"


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


def test_projections_are_not_systematically_biased(built):
    """The old engine reportedly ran 2-4x high. Bias should be within a point."""
    bias = (built.dk_points - built.actual_dk_points).mean()
    assert abs(bias) < 1.0, f"projection bias of {bias:+.2f} DK points"


def test_beats_naive_baseline(built, sample_stats):
    """Phase 4 grades this properly; here we only assert the model earns its place."""
    stats_df = sample_stats.sort_values(["player_id", "season", "week"]).copy()
    stats_df["actual_dk"] = P.dk_points(stats_df)
    baseline = (stats_df.groupby(["player_id", "season"], sort=False)["actual_dk"]
                .transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean()))
    stats_df["baseline"] = baseline
    m = built.merge(stats_df[["player_id", "season", "week", "baseline"]],
                    on=["player_id", "season", "week"], how="left").dropna(subset=["baseline"])

    model_mae = np.abs(m.dk_points - m.actual_dk_points).mean()
    base_mae = np.abs(m.baseline - m.actual_dk_points).mean()
    assert model_mae < base_mae, f"model MAE {model_mae:.3f} worse than baseline {base_mae:.3f}"


def test_exceedance_probabilities_are_calibrated(built, sample_stats):
    """
    Modelled P(rec yards >= 100) must match the realized rate, in aggregate and in
    the high-probability bucket where the threshold actually bites.

    A flat CV over-stated the top bucket by 4 percentage points, because dispersion
    falls as the projection rises. Prop betting is calibration-sensitive: a
    systematic over-statement here manufactures edges on overs that do not exist.
    """
    actuals = sample_stats[["player_id", "season", "week", "receiving_yards"]]
    d = built.merge(actuals, on=["player_id", "season", "week"], how="left").dropna(
        subset=["rec_yd", "receiving_yards"]
    )
    d["p"] = P._exceed_probability(d.rec_yd, 100.0, P.STAT_DISPERSION["rec_yd"])
    d["hit"] = (d.receiving_yards >= 100).astype(float)

    assert abs(d.p.mean() - d.hit.mean()) < 0.02, (
        f"aggregate miscalibration: predicted {d.p.mean():.4f} vs realized {d.hit.mean():.4f}"
    )

    top = d.nlargest(len(d) // 5, "p")
    assert abs(top.p.mean() - top.hit.mean()) < 0.05, (
        f"top-bucket miscalibration: predicted {top.p.mean():.4f} vs realized {top.hit.mean():.4f}"
    )


def test_dispersion_falls_as_projection_rises():
    """cv = k / sqrt(mean): an 8-yard projection is proportionally far more volatile."""
    k = P.STAT_DISPERSION["rec_yd"]
    small = P._exceed_probability(pd.Series([8.0]), 100.0, k).iloc[0]
    large = P._exceed_probability(pd.Series([80.0]), 100.0, k).iloc[0]
    assert small < 0.01, "low projections should almost never clear 100"
    assert large > 0.15, "high projections should clear 100 reasonably often"
