"""
NFL player projections: usage x efficiency, with reliability-weighted shrinkage.

Design follows from measurement rather than intuition. Split-half reliability on
2023-24 (see MEASURED_RELIABILITY below) showed usage metrics are near-deterministic
while efficiency is mostly noise:

    offense_pct      0.978        yards/carry     0.485
    carries          0.960        yards/target    0.325
    target_share     0.959        TD/target       0.092
    air_yards_share  0.938        TD/carry        0.091

That gap drives every decision here:

1. Volume is projected from the player's own trailing usage. It is reliable enough
   that decomposing into (team volume x player share) measured no better (-0.14%,
   noise), so we skip the extra machinery.
2. Efficiency is shrunk toward a positional baseline in proportion to its
   reliability - Kelley's formula, where the weight on a player's own observed rate
   IS its reliability.
3. Touchdowns are projected from opportunity, not from a player's own TD history,
   which is ~91% noise. The Vegas implied team total scales them, since Vegas
   predicts team scoring (r=0.409) and fully subsumes a team's own scoring history.

Deliberately NOT included, because measurement said they do not earn their place:

- Opponent / defense-vs-position adjustment. Correlation with the residual was
  +0.041, and applying it multiplicatively made MAE 1.59% WORSE.
- Aggressive time decay. The decay curve is nearly flat; 0.7 (the old engine's
  value) was up to 2.2% worse than optimal, and even no decay at all was within 1%.
- A team-volume model. Vegas predicts team points but NOT team volume (r=0.081):
  NFL play counts are near-constant across teams.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import nfl_source as src

# DraftKings NFL classic scoring.
DK_PASS_YD, DK_PASS_TD, DK_INT = 0.04, 4.0, -1.0
DK_RUSH_YD, DK_RUSH_TD = 0.1, 6.0
DK_REC, DK_REC_YD, DK_REC_TD = 1.0, 0.1, 6.0
DK_FUMBLE_LOST, DK_TWO_PT, DK_ST_TD = -1.0, 2.0, 6.0
DK_BONUS = 3.0
DK_BONUS_PASS_YD, DK_BONUS_RUSH_YD, DK_BONUS_REC_YD = 300.0, 100.0, 100.0

# Split-half reliability, Spearman-Brown corrected, measured on 2023-24.
# Used directly as the shrinkage weight: weight_on_player = reliability.
MEASURED_RELIABILITY = {
    "usage": 0.95,       # targets, carries, snap share
    "efficiency": 0.40,  # yards per target / per carry
    "td_rate": 0.09,     # TDs per opportunity - almost entirely noise
}

DECAY = 0.9  # mild recency weighting; the curve is flat, so this is not a sensitive knob
MIN_GAMES = 2

# Calibration constants measured once on 2023-24 and frozen rather than fitted per run.
# Fitting at run time would compute them over the whole frame, including weeks after the
# one being projected, which leaks future data into a backtest. Frozen constants are what
# the live system would actually have had.
#
# Dispersion is NOT constant: relative spread shrinks as the projection grows, the way
# it does for any volume-driven count. Measured on 2023-24, cv * sqrt(mean) is stable
# (4.6-5.1 across projection deciles), so dispersion is modelled as cv = k / sqrt(mean).
# A flat CV badly over-states tail probability for exactly the high-projection players
# who sit near the bonus thresholds, which is where it matters.
STAT_DISPERSION = {"pass_yd": 5.452, "rush_yd": 4.722, "rec_yd": 4.820}
CV_BOUNDS = (0.35, 2.0)
LEAGUE_IMPLIED_TOTAL = 21.97


def dk_points(df: pd.DataFrame) -> pd.Series:
    """
    DraftKings fantasy points for a realized stat line.

    Bonuses are applied as step functions here because these are actual results.
    For projections use `expected_dk_points`, where the bonus must be an expectation
    over the distribution rather than a threshold test on the mean.
    """
    pts = (
        df["passing_yards"] * DK_PASS_YD
        + df["passing_tds"] * DK_PASS_TD
        + df["passing_interceptions"] * DK_INT
        + df["rushing_yards"] * DK_RUSH_YD
        + df["rushing_tds"] * DK_RUSH_TD
        + df["receptions"] * DK_REC
        + df["receiving_yards"] * DK_REC_YD
        + df["receiving_tds"] * DK_REC_TD
        + df["special_teams_tds"] * DK_ST_TD
        + (df["rushing_2pt_conversions"] + df["receiving_2pt_conversions"]
           + df["passing_2pt_conversions"]) * DK_TWO_PT
        + (df["rushing_fumbles_lost"] + df["receiving_fumbles_lost"]
           + df["sack_fumbles_lost"]) * DK_FUMBLE_LOST
    )
    pts += DK_BONUS * (df["passing_yards"] >= DK_BONUS_PASS_YD)
    pts += DK_BONUS * (df["rushing_yards"] >= DK_BONUS_RUSH_YD)
    pts += DK_BONUS * (df["receiving_yards"] >= DK_BONUS_REC_YD)
    return pts


def _exceed_probability(mean: pd.Series, threshold: float, k: float) -> pd.Series:
    """
    P(X >= threshold) for a gamma-distributed stat with the given mean.

    DK's yardage bonuses are step functions, so expected points depend on the
    probability of clearing them, not on whether the mean clears them. A player
    projected for 85 rushing yards still earns the bonus a meaningful share of the
    time; ignoring that biases every projection low.

    Gamma is used because yardage is non-negative and right-skewed. It is fitted by
    method of moments, far cheaper and more stable than simulating.

    Dispersion follows cv = k / sqrt(mean) rather than a constant: a player projected
    for 8 yards is proportionally far more volatile than one projected for 65, and
    holding CV fixed overstates the tail precisely where the bonus thresholds bite.
    """
    mean = mean.clip(lower=0.01)
    cv = (k / np.sqrt(mean)).clip(*CV_BOUNDS)
    shape = 1.0 / (cv ** 2)
    scale = mean / shape
    return pd.Series(stats.gamma.sf(threshold, a=shape, scale=scale), index=mean.index)


def expected_dk_points(proj: pd.DataFrame, k: dict[str, float] | None = None) -> pd.Series:
    """Expected DK points for a projection, integrating the bonus thresholds."""
    k = k or STAT_DISPERSION
    pts = (
        proj["pass_yd"] * DK_PASS_YD
        + proj["pass_td"] * DK_PASS_TD
        + proj["interceptions"] * DK_INT
        + proj["rush_yd"] * DK_RUSH_YD
        + proj["rush_td"] * DK_RUSH_TD
        + proj["receptions"] * DK_REC
        + proj["rec_yd"] * DK_REC_YD
        + proj["rec_td"] * DK_REC_TD
    )
    pts += DK_BONUS * _exceed_probability(proj["pass_yd"], DK_BONUS_PASS_YD, k["pass_yd"])
    pts += DK_BONUS * _exceed_probability(proj["rush_yd"], DK_BONUS_RUSH_YD, k["rush_yd"])
    pts += DK_BONUS * _exceed_probability(proj["rec_yd"], DK_BONUS_REC_YD, k["rec_yd"])
    return pts


def _shrink(observed: pd.Series, prior: pd.Series | float, reliability: float) -> pd.Series:
    """
    Kelley's formula: weight the player's own observation by its reliability.

    A metric that is 91% noise should move the projection 9% of the way from the
    population baseline, not 85% as the old engine's hardcoded factors implied.
    """
    return reliability * observed.fillna(prior) + (1 - reliability) * prior


def _trailing(df: pd.DataFrame, col: str, decay: float = DECAY) -> pd.Series:
    """Recency-weighted mean of a player's prior games this season (excludes current)."""

    def weighted(values: pd.Series) -> pd.Series:
        out = np.full(len(values), np.nan)
        arr = values.to_numpy(dtype=float)
        for i in range(1, len(arr)):
            hist = arr[:i][::-1]
            mask = ~np.isnan(hist)
            if mask.sum() == 0:
                continue
            w = decay ** np.arange(len(hist))[mask]
            out[i] = np.average(hist[mask], weights=w)
        return pd.Series(out, index=values.index)

    return df.groupby(["player_id", "season"], sort=False)[col].transform(weighted)


def _trailing_rate(df: pd.DataFrame, num: str, den: str) -> tuple[pd.Series, pd.Series]:
    """Cumulative rate (num/den) over prior games, plus the denominator as sample size."""
    g = df.groupby(["player_id", "season"], sort=False)
    cum_num = g[num].transform(lambda x: x.shift(1).cumsum())
    cum_den = g[den].transform(lambda x: x.shift(1).cumsum())
    return cum_num / cum_den.replace(0, np.nan), cum_den


def _prior_rate(df: pd.DataFrame, num: str, den: str) -> pd.Series:
    """
    Positional baseline rate (e.g. league yards-per-target for WRs) using only weeks
    that precede each row.

    Computing this over the full sample would leak future information into a
    backtest and quietly inflate measured accuracy. The effect per row is small, but
    it is the kind of error that makes a model look better than it is, so it is
    excluded structurally rather than argued about.
    """
    agg = (df.groupby(["position", "season", "week"], as_index=False)[[num, den]].sum()
             .sort_values(["position", "season", "week"]))
    g = agg.groupby("position", sort=False)
    agg["_num"] = g[num].transform(lambda x: x.shift(1).cumsum())
    agg["_den"] = g[den].transform(lambda x: x.shift(1).cumsum())
    agg["_rate"] = agg["_num"] / agg["_den"].replace(0, np.nan)
    merged = df.merge(agg[["position", "season", "week", "_rate"]],
                      on=["position", "season", "week"], how="left")
    # Earliest weeks have no prior; fall back to that position's overall rate.
    overall = df.groupby("position").apply(
        lambda g_: g_[num].sum() / max(g_[den].sum(), 1), include_groups=False
    )
    return merged["_rate"].fillna(df["position"].map(overall)).to_numpy()


def _prior_mean(df: pd.DataFrame, col: str) -> np.ndarray:
    """Positional mean of `col` using only weeks preceding each row. See _prior_rate."""
    agg = (df.groupby(["position", "season", "week"], as_index=False)[col].mean()
             .sort_values(["position", "season", "week"]))
    agg["_mean"] = agg.groupby("position", sort=False)[col].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    merged = df.merge(agg[["position", "season", "week", "_mean"]],
                      on=["position", "season", "week"], how="left")
    overall = df.groupby("position")[col].mean()
    return merged["_mean"].fillna(df["position"].map(overall)).to_numpy()


def _implied_totals(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Vegas implied points per team-game.

    spread_line is from the home team's perspective (positive = home favored).
    """
    home = pd.DataFrame({
        "season": schedule.season, "week": schedule.week, "team": schedule.home_team,
        "implied_total": schedule.total_line / 2 + schedule.spread_line / 2,
    })
    away = pd.DataFrame({
        "season": schedule.season, "week": schedule.week, "team": schedule.away_team,
        "implied_total": schedule.total_line / 2 - schedule.spread_line / 2,
    })
    return pd.concat([home, away], ignore_index=True).dropna(subset=["implied_total"])


def build(stats_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-player-week projections for every week with prior history.

    Returns one row per player-week containing projected volume, yards, TDs, and
    expected DK points. Rows are projections *for* that week using only data from
    earlier weeks, so the output is directly comparable to actuals for backtesting.
    """
    df = stats_df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # --- volume: the reliable half. Own trailing usage, lightly shrunk. ---
    r_use = MEASURED_RELIABILITY["usage"]
    for col, out in (("targets", "e_targets"), ("carries", "e_carries"),
                     ("attempts", "e_attempts")):
        trail = _trailing(df, col)
        prior = pd.Series(_prior_mean(df, col), index=df.index)
        df[out] = _shrink(trail, prior, r_use)

    # --- efficiency: the noisy half. Shrunk hard toward positional baseline. ---
    r_eff = MEASURED_RELIABILITY["efficiency"]
    for num, den, out in (("receiving_yards", "targets", "e_ypt"),
                          ("rushing_yards", "carries", "e_ypc"),
                          ("passing_yards", "attempts", "e_ypa"),
                          ("receptions", "targets", "e_catch_rate")):
        rate, _ = _trailing_rate(df, num, den)
        prior = pd.Series(_prior_rate(df, num, den), index=df.index)
        df[out] = _shrink(rate, prior, r_eff)

    # --- touchdowns: projected from opportunity, not from the player's TD history ---
    r_td = MEASURED_RELIABILITY["td_rate"]
    for num, den, out in (("receiving_tds", "targets", "e_td_per_target"),
                          ("rushing_tds", "carries", "e_td_per_carry"),
                          ("passing_tds", "attempts", "e_td_per_attempt")):
        rate, _ = _trailing_rate(df, num, den)
        prior = pd.Series(_prior_rate(df, num, den), index=df.index)
        df[out] = _shrink(rate, prior, r_td)

    df["pass_yd"] = df["e_attempts"] * df["e_ypa"]
    df["rush_yd"] = df["e_carries"] * df["e_ypc"]
    df["rec_yd"] = df["e_targets"] * df["e_ypt"]
    df["receptions"] = df["e_targets"] * df["e_catch_rate"]
    df["pass_td"] = df["e_attempts"] * df["e_td_per_attempt"]
    df["rush_td"] = df["e_carries"] * df["e_td_per_carry"]
    df["rec_td"] = df["e_targets"] * df["e_td_per_target"]

    int_rate, _ = _trailing_rate(df, "passing_interceptions", "attempts")
    int_prior = pd.Series(_prior_rate(df, "passing_interceptions", "attempts"), index=df.index)
    df["interceptions"] = df["e_attempts"] * _shrink(int_rate, int_prior, r_td)

    # --- Vegas: scales scoring only. It predicts team points, not team volume. ---
    implied = _implied_totals(schedule_df)
    df = df.merge(implied, on=["season", "week", "team"], how="left")
    scale = (df["implied_total"] / LEAGUE_IMPLIED_TOTAL).fillna(1.0)
    for col in ("pass_td", "rush_td", "rec_td"):
        df[col] *= scale

    df["games_played"] = df.groupby(["player_id", "season"], sort=False).cumcount()
    df["dk_points"] = expected_dk_points(df)
    df["actual_dk_points"] = dk_points(df)

    keep = [
        "player_id", "player_display_name", "position", "team", "opponent_team",
        "season", "week", "games_played", "implied_total",
        "e_targets", "e_carries", "e_attempts",
        "pass_yd", "rush_yd", "rec_yd", "receptions",
        "pass_td", "rush_td", "rec_td", "interceptions",
        "dk_points", "actual_dk_points",
    ]
    return df[df.games_played >= MIN_GAMES][keep].reset_index(drop=True)


def project(season: int, week: int, seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Projections for a single upcoming week.

    Args:
        season: Season to project.
        week: Week to project. Only earlier weeks inform the projection.
        seasons: Seasons to load for history. Defaults to the target season.
    """
    seasons = seasons or [season]
    stats_df = src.weekly_stats(seasons)
    schedule_df = src.schedule(seasons)
    built = build(stats_df, schedule_df)
    return (built[(built.season == season) & (built.week == week)]
            .sort_values("dk_points", ascending=False)
            .reset_index(drop=True))
