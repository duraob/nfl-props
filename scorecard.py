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
    """Bias/rank-correlation for one week's logged predictions vs. real outcomes -
    same scoring as backtest.py (B._score), on live ledger data instead of a
    retrospective build() run."""
    if not L.PREDICTIONS.exists():
        return pd.DataFrame()
    preds = pd.read_csv(L.PREDICTIONS, parse_dates=["ts_utc"])
    preds = preds[(preds.season == season) & (preds.week == week)]
    if preds.empty:
        return pd.DataFrame()
    preds = preds.sort_values("ts_utc").drop_duplicates(subset=["player_id", "stat"], keep="last")

    stats_df = src.weekly_stats([season])
    stats_df = stats_df[stats_df.week == week]

    rows = []
    for stat, group in preds.groupby("stat"):
        raw_col = _ACTUAL_COLUMN.get(stat)
        if raw_col is None:
            continue
        merged = group.merge(stats_df[["player_id", raw_col]], on="player_id", how="inner")
        merged = merged.dropna(subset=[raw_col])
        if len(merged) < 5:   # too few graded rows for a meaningful spearman
            continue
        rows.append({"stat": stat, **B._score(merged.projection, merged[raw_col])})
    return pd.DataFrame(rows)


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
        return {"n_bets": 0, "win_rate": None, "n_with_close": 0, "beat_close_rate": None}
    graded = settled.dropna(subset=["result"])
    graded = graded[graded.result != "push"]
    with_close = settled.dropna(subset=["closing_price"])
    return {
        "n_bets": len(settled),
        "win_rate": float((graded.result == "win").mean()) if len(graded) else None,
        "n_with_close": len(with_close),
        "beat_close_rate": float(with_close.beat_close.mean()) if len(with_close) else None,
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


def build_scorecard(season: int, week: int) -> str:
    """Formats the above into a short Telegram message."""
    lines = [f"*Week {week} scorecard*", ""]

    accuracy = week_accuracy(season, week)
    if accuracy.empty:
        lines.append("No graded predictions for this week yet.")
    else:
        lines.append("*This week's accuracy* (vs. real outcomes)")
        for _, row in accuracy.iterrows():
            lines.append(f"  {row.stat}: bias {row.bias:+.1f}, rank corr {row.spearman:.2f} (n={int(row.n)})")
    lines.append("")

    clv = clv_to_date(season)
    lines.append(f"*CLV to date* ({clv['n_bets']} bet(s) recorded)")
    if clv["win_rate"] is not None:
        lines.append(f"  win rate: {clv['win_rate']:.0%}")
    if clv["beat_close_rate"] is not None:
        lines.append(f"  beat closing price: {clv['beat_close_rate']:.0%} ({clv['n_with_close']} priced)")
    if clv["win_rate"] is None and clv["beat_close_rate"] is None:
        lines.append("  nothing settled yet.")
    lines.append("")

    calibration = calibration_to_date(season)
    if calibration.empty:
        lines.append("*Calibration*: not enough settled bets yet.")
    else:
        lines.append("*Calibration* (predicted vs. actual clear rate, by bin)")
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
