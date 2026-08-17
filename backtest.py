"""
Projection validation.

Grades the engine on the metrics that govern real decisions, not just MAE:

- bias      : systematic over/under-projection. The old engine's core failure.
- spearman  : rank correlation. This is what fantasy lineup decisions actually turn
              on - you need the right ordering, not calibrated absolute values.
- MAE       : average error, reported against a naive baseline so the number means
              something. "MAE 3.9" is not interpretable on its own.

Scored per raw stat (passing yards, rushing yards, receiving yards, receptions, TDs,
interceptions) rather than a single composite score - the project cares about the
individual stat lines themselves, which is also what prop bets are priced on.

Holdout discipline: every constant in projections.py was measured on 2023-24, so
2025 is untouched. Tuning-set results are expected to flatter the model; the holdout
number is the real one, and the gap between them is the overfitting estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import nfl_source as src
import projections as P

TUNING_SEASONS = [2023, 2024]
HOLDOUT_SEASON = 2025

# (label, projected column in build()'s output, actual column in weekly_stats)
STATS = [
    ("pass_yd", "pass_yd", "passing_yards"),
    ("pass_td", "pass_td", "passing_tds"),
    ("interceptions", "interceptions", "passing_interceptions"),
    ("rush_yd", "rush_yd", "rushing_yards"),
    ("rush_td", "rush_td", "rushing_tds"),
    ("rec_yd", "rec_yd", "receiving_yards"),
    ("rec_td", "rec_td", "receiving_tds"),
    ("receptions", "receptions", "receptions"),
]


def _baselines(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Naive last-4-week comparator per stat, plus the actual values themselves.

    Grouped by player only, not player+season, matching projections.py's build():
    the model now carries a player's trailing history across the season boundary
    (weeks 1-2 would otherwise have zero same-season history and be dropped by
    MIN_GAMES). If this baseline stayed season-scoped, it would have no fair number
    for weeks 1-2 at all, and scorecard()'s dropna would silently exclude exactly the
    weeks the model is now able to project - hiding the model's real early-season
    performance right when validating it matters most.

    Output columns are prefixed `obs_`/`bl_` rather than reusing the stat's own name
    (e.g. `obs_receptions`, not `receptions`), because build()'s output already has a
    *projected* `receptions` column - reusing the bare name would silently compare a
    column against itself after the merge instead of raising an error. That exact
    collision produced a first-draft bug here (receptions scored as a perfect
    prediction, MAE 0.0) before the columns were renamed to be unambiguous.
    """
    df = stats_df.sort_values(["player_id", "season", "week"]).copy()
    out_cols = ["player_id", "season", "week"]
    for label, _, actual_col in STATS:
        obs_col, baseline_col = f"obs_{label}", f"bl_{label}"
        g = df.groupby("player_id", sort=False)[actual_col]
        df[baseline_col] = g.transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean())
        df[obs_col] = df[actual_col]
        out_cols += [obs_col, baseline_col]
    return df[out_cols]


def evaluate(seasons: list[int]) -> pd.DataFrame:
    """
    Score the engine and its baseline over the given seasons, for every stat.

    Loads one extra season before the earliest requested one purely as trailing
    context - scored rows are still filtered to `seasons` only. Without this, weeks
    1-2 of the earliest scored season would have no history (same reason build() now
    carries a player's data across the season boundary) and silently score nothing,
    hiding the exact weeks that most need validating before a real season starts.
    """
    load_seasons = [min(seasons) - 1] + list(seasons)
    stats_df = src.weekly_stats(load_seasons)
    schedule_df = src.schedule(load_seasons)
    built = P.build(stats_df, schedule_df)
    built = built[built.season.isin(seasons)]
    return built.merge(_baselines(stats_df), on=["player_id", "season", "week"], how="left")


def _score(pred: pd.Series, actual: pd.Series) -> dict:
    err = pred - actual
    return {
        "n": len(pred),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "spearman": float(pred.corr(actual, method="spearman")),
    }


def scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """Model vs. last-4-week baseline, one row per stat."""
    rows = []
    for label, proj_col, _ in STATS:
        obs_col, baseline_col = f"obs_{label}", f"bl_{label}"
        d = df.dropna(subset=[baseline_col])
        d = d[d[baseline_col] > 0]   # excludes players for whom this stat isn't part of their role
        if d.empty:
            continue
        rows.append({"stat": label, "model": "engine", "n": len(d),
                    **_score(d[proj_col], d[obs_col])})
        rows.append({"stat": label, "model": "last-4 baseline", "n": len(d),
                    **_score(d[baseline_col], d[obs_col])})
    return pd.DataFrame(rows)


def run(seasons: list[int], label: str) -> pd.DataFrame:
    """Evaluate and print a scorecard for one set of seasons."""
    df = evaluate(seasons)
    card = scorecard(df)
    print(f"\n{'=' * 70}\n{label}  (seasons={seasons})\n{'=' * 70}")
    print(f"{'stat':16s} {'n':>6s} {'MAE':>8s} {'bias':>8s} {'spearman':>9s}   vs baseline")

    for stat, _, _ in STATS:
        sub = card[card.stat == stat]
        if sub.empty:
            continue
        m = sub[sub.model == "engine"].iloc[0]
        b = sub[sub.model == "last-4 baseline"].iloc[0]
        improvement = 100 * (b.mae - m.mae) / b.mae if b.mae else float("nan")
        print(f"  {stat:14s} {int(m.n):6d} {m.mae:8.3f} {m.bias:+8.3f} {m.spearman:9.3f}"
              f"   {improvement:+6.2f}%")
    return df


if __name__ == "__main__":
    run(TUNING_SEASONS, "TUNING SET (constants were fitted here)")
    run([HOLDOUT_SEASON], "HOLDOUT (never used for any decision)")
