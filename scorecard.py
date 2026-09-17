"""
Weekly scorecard, pushed to Telegram: bias, rank correlation, calibration, and
CLV-to-date.

backtest.py answers "does the model work" once, against 2023-25. This answers it
every week, against the actual season as it happens - the genuinely clean test
CLAUDE.md keeps pointing to, since 2025 has been observed multiple times and 2023-24
were fitted on directly.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

import backtest as B
import ledger as L
import nfl_source as src
import projections as P
import settle as S

# label -> raw weekly_stats column, taken from backtest.py's STATS so both scorers
# agree on what "actual" means for a given stat.
_ACTUAL_COLUMN = {label: raw for label, _, raw in B.STATS}


def week_accuracy(season: int, week: int) -> pd.DataFrame:
    """
    Bias/rank-correlation for one week's logged predictions vs. real outcomes,
    scored the way backtest.py scores - including its role filter and its baseline.

    Both of those were missing here originally, and the result was not comparable to
    the holdout table it was formatted to look like. weekly_stats has a row per
    player who *played*, with zeros in every stat outside his role, and this graded
    every logged player on all eight stats: 90% of the rows scored for pass_yd in
    Week 1 were non-quarterbacks with an actual of 0. Those rows drag bias toward
    zero and inflate spearman, because ranking quarterbacks above receivers in
    passing yards is free. Restricted to players who actually threw, Week 1's
    pass_yd spearman was 0.02, not the 0.46 the unfiltered card reported.

    The filter is `bl_{stat} > 0` - a positive trailing average, i.e. this stat is
    part of the player's role - exactly backtest.scorecard()'s, and deliberately not
    "actual > 0", which would condition on the outcome being graded.
    """
    if not L.PREDICTIONS.exists():
        return pd.DataFrame()
    preds = pd.read_csv(L.PREDICTIONS, parse_dates=["ts_utc"])
    preds = preds[(preds.season == season) & (preds.week == week)]
    if preds.empty:
        return pd.DataFrame()
    preds = preds.sort_values("ts_utc").drop_duplicates(subset=["player_id", "stat"], keep="last")

    # Prior season loaded too: the trailing baseline carries across the season
    # boundary (see backtest._baselines), so weeks 1-2 have a fair comparator at all.
    baselines = B._baselines(src.weekly_stats([season - 1, season]))
    baselines = baselines[(baselines.season == season) & (baselines.week == week)]

    rows = []
    for label, _, _ in B.STATS:
        obs_col, baseline_col = f"obs_{label}", f"bl_{label}"
        graded = preds[preds.stat == label].merge(
            baselines[["player_id", obs_col, baseline_col]], on="player_id", how="inner"
        ).dropna(subset=[obs_col, baseline_col])
        graded = graded[graded[baseline_col] > 0]
        if len(graded) < 5:   # too few graded rows for a meaningful spearman
            continue
        model = B._score(graded.projection, graded[obs_col])
        base = B._score(graded[baseline_col], graded[obs_col])
        rows.append({"stat": label, **model,
                     "baseline_mae": base["mae"], "baseline_spearman": base["spearman"]})
    return pd.DataFrame(rows)


def week_coverage(season: int, week: int) -> dict:
    """How many projected players the week actually graded.

    Printed because the old card showed an identical n for all eight stats, which
    was the visible symptom of the missing role filter - differing n is now the
    at-a-glance signal that it is still applied.
    """
    if not L.PREDICTIONS.exists():
        return {"projected": 0, "played": 0}
    preds = pd.read_csv(L.PREDICTIONS)
    preds = preds[(preds.season == season) & (preds.week == week)]
    if preds.empty:
        return {"projected": 0, "played": 0}
    played = src.weekly_stats([season])
    played = set(played[played.week == week].player_id)
    projected = set(preds.player_id)
    return {"projected": len(projected), "played": len(projected & played)}


def _settled_to_date(season: int) -> pd.DataFrame:
    """Every bet placed this season, graded - concatenated across whichever weeks
    have both a bet and a played game. Weeks with nothing to settle are skipped
    rather than raising (settle.py raises only when bets.csv doesn't exist at all)."""
    if not S.BETS.exists():
        return pd.DataFrame()
    bets = pd.read_csv(S.BETS)
    bets = bets[bets.season == season]
    if bets.empty:
        return pd.DataFrame()
    rows = [S.settle(season, w) for w in sorted(bets.week.unique())]
    rows = [r for r in rows if not r.empty]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def clv_to_date(season: int) -> dict:
    """Aggregate win rate and closing-line value across every settled bet this
    season. Both are None (not zero) when there isn't data yet - a scorecard that
    silently printed 0% would misread as 'the model is losing' instead of 'nothing
    has happened yet.'"""
    settled = _settled_to_date(season)
    if settled.empty:
        return {"n_bets": 0, "n_graded": 0, "n_wins": 0, "win_rate": None,
                "n_with_close": 0, "n_stale_close": 0, "beat_close_rate": None}
    graded = settled.dropna(subset=["result"])
    graded = graded[graded.result != "push"]
    with_close = settled.dropna(subset=["closing_price"])
    # A closing price captured before the bet is the snapshot the bet was made from,
    # so its CLV is 0.00 by construction - counted separately rather than averaged
    # in, where it would read as "no edge" instead of "not measured". See settle.py.
    fresh = with_close[~with_close.close_is_stale.astype(bool)]
    return {
        "n_bets": len(settled),
        "n_graded": len(graded),
        "n_wins": int((graded.result == "win").sum()),
        "win_rate": float((graded.result == "win").mean()) if len(graded) else None,
        "n_with_close": len(with_close),
        "n_stale_close": len(with_close) - len(fresh),
        "beat_close_rate": float(fresh.beat_close.mean()) if len(fresh) else None,
    }


def calibration_to_date(season: int, n_bins: int = 5) -> pd.DataFrame:
    """
    Reliability check across every settled bet: of the bets where the model said
    roughly X% likely to clear, did ~X% actually clear?

    Recomputes model_probability from each settled bet's logged projection and
    recorded line (skip_volume_check=True: a bet is itself proof the player had
    real volume in that stat, so the check settle.py's thin records can't perform
    is redundant here, not skipped for convenience).
    """
    settled = _settled_to_date(season)
    if settled.empty:
        return pd.DataFrame()
    settled = settled.dropna(subset=["projection", "actual", "line"])
    settled = settled[settled.stat.isin(P.STAT_DISPERSION)]
    if settled.empty:
        return pd.DataFrame()

    rows = []
    for stat, group in settled.groupby("stat"):
        proj_df = pd.DataFrame({stat: group.projection.to_numpy()}, index=group.index)
        model_p = P.exceed_probability(proj_df, stat, group.line.to_numpy(), skip_volume_check=True)
        cleared = (group.actual >= group.line).astype(float)
        rows.append(pd.DataFrame({"model_probability": model_p, "cleared": cleared}))
    combined = pd.concat(rows, ignore_index=True)

    bins = pd.cut(combined.model_probability, n_bins, include_lowest=True)
    table = combined.groupby(bins, observed=True).agg(
        n=("cleared", "size"), predicted=("model_probability", "mean"),
        actual_rate=("cleared", "mean"),
    ).reset_index(names="bin")
    return table[table.n > 0]


def _label(stat: str) -> str:
    """Telegram renders with parse_mode=Markdown, where `_` opens italics - so
    `pass_td` arrived on the phone as *passtd* in italics and silently swallowed the
    formatting of everything after it. Spaces read better in a message anyway."""
    return stat.replace("_", " ")


def build_scorecard(season: int, week: int) -> str:
    """Formats the above into a short Telegram message."""
    lines = [f"*Week {week} scorecard*", ""]

    accuracy = week_accuracy(season, week)
    if accuracy.empty:
        lines.append("No graded predictions for this week yet.")
    else:
        lines.append("*Accuracy* - model vs. what actually happened")
        lines.append("bias: + means projected too high. rank: 1.0 = perfect order,")
        lines.append("0 = coin flip. base: how much better than just averaging")
        lines.append("that player's last 4 games.")
        for _, row in accuracy.iterrows():
            better = 100 * (row.baseline_mae - row.mae) / row.baseline_mae if row.baseline_mae else float("nan")
            lines.append(f"  {_label(row.stat):14s} bias {row.bias:+6.1f}  rank {row.spearman:+5.2f}"
                         f"  base {better:+.0f}%  (n={int(row.n)})")
        coverage = week_coverage(season, week)
        lines.append(f"  {coverage['played']} of {coverage['projected']} projected players played.")
        lines.append("  Only players for whom a stat is part of their role are")
        lines.append("  graded on it, so n differs per stat.")
    lines.append("")

    clv = clv_to_date(season)
    lines.append(f"*Bets* - {clv['n_bets']} recorded this season")
    if clv["win_rate"] is not None:
        lines.append(f"  win rate: {clv['win_rate']:.0%} ({clv['n_wins']} of {clv['n_graded']}, pushes excluded)")
    if clv["beat_close_rate"] is not None:
        lines.append(f"  beat closing price: {clv['beat_close_rate']:.0%} ({clv['n_with_close'] - clv['n_stale_close']} priced)")
    elif clv["n_stale_close"]:
        # Never print a 0% here. Every Week 1 closing price was the same snapshot the
        # bet was placed from, so a CLV of 0.00 would be an artifact of the capture
        # schedule reported as a result.
        lines.append(f"  closing-line value: not measurable - all {clv['n_stale_close']} closing")
        lines.append("  prices were captured before the bet was placed. Needs a")
        lines.append("  sweep nearer kickoff, not a model change.")
    if clv["win_rate"] is None and clv["beat_close_rate"] is None and not clv["n_stale_close"]:
        lines.append("  nothing settled yet.")
    lines.append("")

    calibration = calibration_to_date(season)
    if calibration.empty:
        lines.append("*Calibration*: not enough settled bets yet.")
    else:
        lines.append("*Calibration* - when the model said X% likely, how often did")
        lines.append("it happen? Bets only, so this is a small, self-selected sample.")
        for _, row in calibration.iterrows():
            lines.append(f"  {row.predicted:.0%} predicted -> {row.actual_rate:.0%} actual (n={int(row.n)})")

    return "\n".join(lines)


def most_recently_completed_week(now: dt.datetime | None = None) -> tuple[int, int] | None:
    """
    (season, week) for the most recent week whose games have all finished - the
    natural "how did we do last week" target for a Tuesday scorecard push.

    Deliberately separate from schedule_captures.current_week(), which answers a
    different question ("should I still be capturing for this week") with a 1-day
    grace period tuned for that - reusing it here would make the scorecard's target
    week depend on exactly how many hours past Monday night's kickoff cron happens
    to run, rather than being simply "the last fully-played week."
    """
    now = now or dt.datetime.now(dt.UTC)
    season = now.year if now.month >= 3 else now.year - 1
    try:
        sched = src.schedule([season])
    except Exception:
        return None
    sched = sched.dropna(subset=["gameday"]).copy()
    if sched.empty:
        return None
    sched["gameday"] = pd.to_datetime(sched["gameday"]).dt.tz_localize(now.tzinfo)
    latest_by_week = sched.groupby("week")["gameday"].max()
    finished = latest_by_week[latest_by_week < now]
    return (season, int(finished.idxmax())) if not finished.empty else None


if __name__ == "__main__":
    import sys

    import telegram_notify as T

    if len(sys.argv) > 2 and sys.argv[1].isdigit():
        season, week = int(sys.argv[1]), int(sys.argv[2])
    else:
        week_info = most_recently_completed_week()
        if week_info is None:
            print("No completed NFL week yet - nothing to score.")
            sys.exit(0)
        season, week = week_info

    text = build_scorecard(season, week)
    if "--push" in sys.argv:
        T.send_message(text)
    else:
        print(text)
