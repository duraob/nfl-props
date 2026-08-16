"""
Projection validation.

Grades the engine on the metrics that govern real decisions, not just MAE:

- bias      : systematic over/under-projection. The old engine's core failure.
- spearman  : rank correlation. This is what fantasy lineup decisions actually turn
              on - you need the right ordering, not calibrated absolute values.
- MAE       : average error, reported against a naive baseline so the number means
              something. "MAE 3.9" is not interpretable on its own.
- calibration: do the modelled probabilities match realized frequencies? This is
              what prop betting turns on, and a model can rank well while being
              badly calibrated.

Holdout discipline: every constant in projections.py was measured on 2022-24, so
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


def _baselines(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Naive comparators. Without these, an MAE figure has no meaning."""
    df = stats_df.sort_values(["player_id", "season", "week"]).copy()
    df["actual_dk"] = P.dk_points(df)
    g = df.groupby(["player_id", "season"], sort=False)["actual_dk"]
    df["bl_last4"] = g.transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean())
    df["bl_season"] = g.transform(lambda x: x.shift(1).expanding(min_periods=2).mean())
    # Realized yardage travels alongside the baselines so calibration can be checked.
    return df[["player_id", "season", "week", "bl_last4", "bl_season",
               "receiving_yards", "rushing_yards", "passing_yards"]]


def evaluate(seasons: list[int]) -> pd.DataFrame:
    """Score the engine and its baselines over the given seasons."""
    stats_df = src.weekly_stats(seasons)
    schedule_df = src.schedule(seasons)
    built = P.build(stats_df, schedule_df)
    merged = built.merge(_baselines(stats_df), on=["player_id", "season", "week"], how="left")
    return merged.dropna(subset=["bl_last4", "bl_season"])


def _score(pred: pd.Series, actual: pd.Series) -> dict:
    err = pred - actual
    return {
        "n": len(pred),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "spearman": float(pred.corr(actual, method="spearman")),
        "pearson": float(pred.corr(actual)),
    }


def scorecard(df: pd.DataFrame, by_position: bool = True) -> pd.DataFrame:
    """Model vs baselines, overall and per position."""
    rows = []
    groups = [("ALL", df)]
    if by_position:
        groups += [(p, d) for p, d in df.groupby("position")]

    for label, d in groups:
        for name, col in (("model", "dk_points"),
                          ("last-4 baseline", "bl_last4"),
                          ("season baseline", "bl_season")):
            rows.append({"group": label, "model": name, **_score(d[col], d.actual_dk_points)})
    return pd.DataFrame(rows)


def calibration(df: pd.DataFrame, stat: str = "rec_yd", threshold: float = 100.0,
                bins: int = 5) -> pd.DataFrame:
    """
    Do modelled probabilities match reality?

    Buckets rows by predicted P(stat >= threshold) and compares to the realized rate.
    A well-calibrated model puts ~30% of the 30%-bucket over the line. This validates
    the same distribution the DK bonus math relies on, and is the prerequisite for
    trusting any prop-betting edge later.
    """
    actual_col = {"rec_yd": "receiving_yards", "rush_yd": "rushing_yards",
                  "pass_yd": "passing_yards"}[stat]
    d = df.dropna(subset=[stat, actual_col]).copy()
    d["p_pred"] = P._exceed_probability(d[stat], threshold, P.STAT_DISPERSION[stat])
    d["hit"] = (d[actual_col] >= threshold).astype(float)

    d["bucket"] = pd.qcut(d.p_pred, bins, duplicates="drop")
    out = d.groupby("bucket", observed=True).agg(
        n=("hit", "size"), predicted=("p_pred", "mean"), realized=("hit", "mean")
    ).reset_index(drop=True)
    out["error"] = out.realized - out.predicted
    return out


def run(seasons: list[int], label: str) -> pd.DataFrame:
    """Evaluate and print a scorecard for one set of seasons."""
    df = evaluate(seasons)
    card = scorecard(df)
    print(f"\n{'=' * 66}\n{label}  (seasons={seasons}, n={len(df)})\n{'=' * 66}")
    overall = card[card.group == "ALL"]
    print(f"{'':18s} {'MAE':>7s} {'bias':>8s} {'spearman':>9s}")
    for _, r in overall.iterrows():
        print(f"  {r.model:16s} {r.mae:7.3f} {r.bias:+8.3f} {r.spearman:9.3f}")

    m = overall[overall.model == "model"].iloc[0]
    b = overall[overall.model == "last-4 baseline"].iloc[0]
    print(f"\n  MAE improvement vs last-4 : {100 * (b.mae - m.mae) / b.mae:+.2f}%")
    print(f"  Spearman improvement      : {100 * (m.spearman - b.spearman) / b.spearman:+.2f}%")

    print("\n  by position (model MAE vs last-4 baseline):")
    for pos in ("QB", "RB", "WR", "TE"):
        sub = card[card.group == pos]
        if sub.empty:
            continue
        mm = sub[sub.model == "model"].iloc[0]
        bb = sub[sub.model == "last-4 baseline"].iloc[0]
        print(f"    {pos}: {mm.mae:6.3f} vs {bb.mae:6.3f}  ({100 * (bb.mae - mm.mae) / bb.mae:+5.1f}%)"
              f"   spearman {mm.spearman:.3f}")
    return df


if __name__ == "__main__":
    tune = run(TUNING_SEASONS, "TUNING SET (constants were fitted here)")
    hold = run([HOLDOUT_SEASON], "HOLDOUT (never used for any decision)")

    print(f"\n{'=' * 66}\nCALIBRATION on holdout - P(receiving yards >= 100)\n{'=' * 66}")
    print(calibration(hold).to_string(index=False))
