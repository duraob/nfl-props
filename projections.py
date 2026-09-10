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

# Bump on any change to build()'s model logic (shrinkage, priors, injury/rookie
# handling) so a change in ledger.py's logged predictions can be attributed to the
# change rather than confused with ordinary variance. Not bumped for pure
# refactors that don't change output.
MODEL_VERSION = "1.0"

# Split-half reliability, Spearman-Brown corrected, measured on 2023-24.
# Used directly as the shrinkage weight: weight_on_player = reliability.
MEASURED_RELIABILITY = {
    "usage": 0.95,       # targets, carries, snap share
    "efficiency": 0.40,  # yards per target / per carry
    "td_rate": 0.09,     # TDs per opportunity - almost entirely noise
}

DECAY = 0.9  # mild recency weighting; the curve is flat, so this is not a sensitive knob
MIN_GAMES = 2

# Measured once on 2023-24 and frozen rather than fitted per run. Fitting at run time
# would compute it over the whole frame, including weeks after the one being projected,
# which leaks future data into a backtest.
LEAGUE_IMPLIED_TOTAL = 21.97

# Measured on 2023-25: for every skill-position player-week flagged on the injury
# report, build()'s retrospective projection vs. actual production (0 for player-weeks
# with no stats row at all - i.e. did not play). This nets "didn't play" and "played
# but limited" into one expected-value multiplier rather than measuring either alone.
# Out (976 tagged, 0.0% played) and Doubtful (134 tagged, 0.7% played) are
# indistinguishable from each other and are excluded outright below rather than given
# a near-zero multiplier that implies false precision on a handful of players.
INJURY_DISCOUNT = {"Questionable": 0.514}  # 1205 tagged, 51.0% played
EXCLUDED_INJURY_STATUSES = {"Out", "Doubtful"}
INJURY_DISCOUNTED_COLUMNS = [
    "e_targets", "e_carries", "e_attempts",
    "pass_yd", "rush_yd", "rec_yd", "receptions",
    "pass_td", "rush_td", "rec_td", "interceptions",
]

# Measured on 2023-24 depth charts (the only seasons with week-aligned history - see
# nfl_source.depth_chart_ranks). depth_team there ranks players WITHIN one formation
# slot (three different WRs can each be "1", for the X/Z/Slot spots), so each
# player's best slot rank was first reduced to one overall depth order per team
# before averaging actual targets/carries/attempts by (position, that rank capped at
# 3rd). The descending pattern by tier (e.g. TE: 3.40 / 1.80 / 0.62 targets) confirms
# depth-chart rank is a real volume signal - a rookie's role, not just his stat line.
DEPTH_RANK_VOLUME_PRIOR = pd.DataFrame([
    {"position": "QB", "tier": 1, "e_targets": 0.0, "e_carries": 3.1, "e_attempts": 24.8},
    {"position": "QB", "tier": 2, "e_targets": 0.0, "e_carries": 0.6, "e_attempts": 3.7},
    {"position": "QB", "tier": 3, "e_targets": 0.0, "e_carries": 0.4, "e_attempts": 2.0},
    {"position": "RB", "tier": 1, "e_targets": 2.3, "e_carries": 9.5, "e_attempts": 0.0},
    {"position": "RB", "tier": 2, "e_targets": 1.5, "e_carries": 5.2, "e_attempts": 0.0},
    {"position": "RB", "tier": 3, "e_targets": 0.9, "e_carries": 3.1, "e_attempts": 0.0},
    {"position": "WR", "tier": 1, "e_targets": 4.3, "e_carries": 0.1, "e_attempts": 0.0},
    {"position": "WR", "tier": 2, "e_targets": 4.2, "e_carries": 0.1, "e_attempts": 0.0},
    {"position": "WR", "tier": 3, "e_targets": 2.3, "e_carries": 0.1, "e_attempts": 0.0},
    {"position": "TE", "tier": 1, "e_targets": 3.4, "e_carries": 0.0, "e_attempts": 0.0},
    {"position": "TE", "tier": 2, "e_targets": 1.8, "e_carries": 0.0, "e_attempts": 0.0},
    {"position": "TE", "tier": 3, "e_targets": 0.6, "e_carries": 0.0, "e_attempts": 0.0},
])


def _shrink(observed: pd.Series, prior: pd.Series | float, reliability: float) -> pd.Series:
    """
    Kelley's formula: weight the player's own observation by its reliability.

    A metric that is 91% noise should move the projection 9% of the way from the
    population baseline, not 85% as the old engine's hardcoded factors implied.
    """
    return reliability * observed.fillna(prior) + (1 - reliability) * prior


def _trailing(df: pd.DataFrame, col: str, decay: float = DECAY) -> pd.Series:
    """
    Recency-weighted mean of a player's prior games (excludes current).

    Grouped by player only, not player+season: resetting at the season boundary
    would mean weeks 1-2 of every season have zero trailing history and get filtered
    out entirely by MIN_GAMES, producing no projections right when they matter most.
    Last season's final games become this season's history instead, decayed the same
    as any other prior game - a practical necessity for early-season projections,
    though it does not know when a player's role changed teams or offenses.
    """

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

    return df.groupby("player_id", sort=False)[col].transform(weighted)


def _trailing_rate(df: pd.DataFrame, num: str, den: str) -> tuple[pd.Series, pd.Series]:
    """Cumulative rate (num/den) over prior games, plus the denominator as sample size.
    Spans season boundaries - see _trailing."""
    g = df.groupby("player_id", sort=False)
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


def _team_game_info(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Per-team-game Vegas implied total plus kickoff day/time.

    spread_line is from the home team's perspective (positive = home favored). Kickoff
    fields matter because the NFL week is not one event: a 2026 Week 1 slate has games
    on Wednesday, Thursday, Sunday, and Monday, each needing its own capture/decision
    timing rather than one blanket "midweek" treatment.
    """
    cols = ["gameday", "gametime"]
    home = pd.DataFrame({
        "season": schedule.season, "week": schedule.week, "team": schedule.home_team,
        "implied_total": schedule.total_line / 2 + schedule.spread_line / 2,
        **{c: schedule[c] for c in cols},
    })
    away = pd.DataFrame({
        "season": schedule.season, "week": schedule.week, "team": schedule.away_team,
        "implied_total": schedule.total_line / 2 - schedule.spread_line / 2,
        **{c: schedule[c] for c in cols},
    })
    return pd.concat([home, away], ignore_index=True).dropna(subset=["implied_total"])


# Confidence tiers, capped by the reliability measurements above rather than reaching
# 1.0: a yardage/reception projection cannot exceed ~0.40 confidence no matter the
# sample size, because that is the ceiling measured for efficiency stats. TD/INT
# projections cap at ~0.09 - always "low" under CONFIDENCE_LABELS below, which is the
# honest answer, not a bug to fix.
CONFIDENCE_FULL_SAMPLE_GAMES = 8  # games of trailing history where confidence saturates
CONFIDENCE_TIERS = {
    # (projected columns): reliability ceiling for that tier
    ("pass_yd", "rush_yd", "rec_yd", "receptions"): MEASURED_RELIABILITY["efficiency"],
    ("pass_td", "rush_td", "rec_td", "interceptions"): MEASURED_RELIABILITY["td_rate"],
}
CONFIDENCE_LABELS = [(0.30, "high"), (0.12, "medium"), (0.0, "low")]
# Derived from CONFIDENCE_TIERS - single source of truth for "which confidence column
# grades this stat," used by build() below and by ledger.py/market_odds.py.
STAT_CONFIDENCE_COLUMN = {
    stat: ("confidence_yardage" if "pass_yd" in cols else "confidence_touchdown")
    for cols in CONFIDENCE_TIERS for stat in cols
}


def _confidence_label(value: float) -> str:
    for cutoff, label in CONFIDENCE_LABELS:
        if value >= cutoff:
            return label
    return "low"


def build(stats_df: pd.DataFrame, schedule_df: pd.DataFrame, min_games: int = MIN_GAMES) -> pd.DataFrame:
    """
    Build per-player-week projections for every week with prior history.

    Returns one row per player-week containing projected volume, yards, and TDs.
    Rows are projections *for* that week using only data from earlier weeks, so the
    output is directly comparable to actuals for backtesting.

    min_games: rows below this many prior games are dropped, since they hold nothing
        but a flat positional-average prior (see _prior_mean/_prior_rate) rather than
        anything measured about the player. Defaults to MIN_GAMES; project() passes 0
        so it can apply a better, depth-chart-based prior to some of those rows
        itself instead of losing them outright - see projections.py's no-history
        handling.
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

    # Real single-game yardage can go negative (a QB's receiving yards on a lateral),
    # so trailing rates occasionally are, and shrinkage can carry that through to a
    # small negative projection. As an *expectation* for a bet or a lineup that is
    # meaningless, and it looks like a defect in any report. Clamp at zero.
    for column in ("pass_yd", "rush_yd", "rec_yd", "receptions",
                   "pass_td", "rush_td", "rec_td"):
        df[column] = df[column].clip(lower=0)

    int_rate, _ = _trailing_rate(df, "passing_interceptions", "attempts")
    int_prior = pd.Series(_prior_rate(df, "passing_interceptions", "attempts"), index=df.index)
    df["interceptions"] = df["e_attempts"] * _shrink(int_rate, int_prior, r_td)

    # --- Vegas + kickoff timing: both come from the same per-team-game frame. ---
    game_info = _team_game_info(schedule_df)
    df = df.merge(game_info, on=["season", "week", "team"], how="left")
    scale = (df["implied_total"] / LEAGUE_IMPLIED_TOTAL).fillna(1.0)
    for col in ("pass_td", "rush_td", "rec_td"):
        df[col] *= scale

    # Cross-season count, matching _trailing's grouping: a player with 16 games of
    # history from last season should not look identical to a rookie in week 1.
    df["games_played"] = df.groupby("player_id", sort=False).cumcount()

    # --- confidence: capped by measured reliability, ramping in with sample size. ---
    games_factor = (df["games_played"] / CONFIDENCE_FULL_SAMPLE_GAMES).clip(upper=1.0)
    for cols, ceiling in CONFIDENCE_TIERS.items():
        conf_col = STAT_CONFIDENCE_COLUMN[cols[0]]
        df[conf_col] = ceiling * games_factor
    df["confidence_yardage_label"] = df["confidence_yardage"].apply(_confidence_label)
    df["confidence_touchdown_label"] = df["confidence_touchdown"].apply(_confidence_label)

    keep = [
        "player_id", "player_display_name", "position", "team", "opponent_team",
        "season", "week", "gameday", "gametime", "games_played", "implied_total",
        "e_targets", "e_carries", "e_attempts",
        "pass_yd", "rush_yd", "rec_yd", "receptions",
        "pass_td", "rush_td", "rec_td", "interceptions",
        "confidence_yardage", "confidence_touchdown",
        "confidence_yardage_label", "confidence_touchdown_label",
    ]
    return df[df.games_played >= min_games][keep].reset_index(drop=True)


def _placeholder_rows(season: int, week: int, stats_df: pd.DataFrame,
                      schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthetic stat rows for a week that has not been played yet.

    build() is retrospective: it emits one projection per player-week that already
    exists in the stats data. A future week has no such rows, so without this it
    returns nothing for exactly the week you want to bet on. This constructs the
    missing rows from the published roster (who is on which team) plus the schedule
    (who they play), with every raw stat left NaN so nothing mistakes a not-yet-played
    game for a zero-production one.

    The trailing calculations in build() all exclude the current row, so these
    placeholders receive projections derived purely from real prior games.
    """
    roster = src.rosters([season])
    week_games = schedule_df[(schedule_df.season == season) & (schedule_df.week == week)]

    # An NFL week is not one event - 2026 Week 1 runs Wednesday to Monday. Testing
    # "has this week been played" as a single boolean means the first game's stats
    # publishing strips placeholders from every *remaining* game in the same week:
    # Week 1 collapsed from 493 players to the 23 who played Wednesday, and the
    # Thursday, Sunday and Monday reports all returned "no games" rather than
    # erroring. Exclude only the teams whose own game already has stats.
    played = stats_df[(stats_df.season == season) & (stats_df.week == week)]
    played_teams = set(played.team.dropna().unique())
    week_games = week_games[~(week_games.home_team.isin(played_teams)
                              | week_games.away_team.isin(played_teams))]

    if roster.empty or week_games.empty:
        return pd.DataFrame()

    # One row per team-game, so a player maps to his opponent and kickoff.
    home = week_games.rename(columns={"home_team": "team", "away_team": "opponent_team"})
    away = week_games.rename(columns={"away_team": "team", "home_team": "opponent_team"})
    matchups = pd.concat([home, away], ignore_index=True)[
        ["team", "opponent_team", "gameday", "gametime"]
    ]

    rows = roster.merge(matchups, on="team", how="inner")
    if rows.empty:
        return pd.DataFrame()

    # Every scheduled team must contribute players. A team silently producing zero
    # projections is how the 2026 ARI/AZ abbreviation mismatch first hid itself -
    # see nfl_source.TEAM_ALIASES. Surface any future mismatch loudly instead.
    scheduled_teams = set(matchups.team)
    covered_teams = set(rows.team)
    if missing_teams := scheduled_teams - covered_teams:
        raise RuntimeError(
            f"No rostered players matched these scheduled teams: {sorted(missing_teams)}. "
            "This usually means a team-abbreviation mismatch between the roster and "
            "schedule feeds - check nfl_source.TEAM_ALIASES."
        )

    placeholders = pd.DataFrame({
        "player_id": rows.gsis_id,
        "player_display_name": rows.full_name,
        "position": rows.position,
        "team": rows.team,
        "opponent_team": rows.opponent_team,
        "season": season,
        "week": week,
    })
    # Raw stat columns must exist for build()'s aggregations, but stay NaN: this game
    # has not happened, and a 0 would read as "played and produced nothing".
    for column in src.STAT_COLUMNS:
        if column not in placeholders.columns:
            placeholders[column] = np.nan
    return placeholders


def _apply_injury_status(week_rows: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """
    Drop players ruled Out/Doubtful for this week and discount Questionable players
    toward INJURY_DISCOUNT. Only called once the season has started - see
    project()'s season_started gate; injury reports do not exist before then.
    """
    status = src.injuries([season])
    status = status[(status.season == season) & (status.week == week)][
        ["gsis_id", "report_status"]
    ]
    week_rows = week_rows.merge(status, left_on="player_id", right_on="gsis_id", how="left")
    week_rows = week_rows[~week_rows.report_status.isin(EXCLUDED_INJURY_STATUSES)]
    multiplier = week_rows["report_status"].map(INJURY_DISCOUNT).fillna(1.0)
    week_rows[INJURY_DISCOUNTED_COLUMNS] = week_rows[INJURY_DISCOUNTED_COLUMNS].mul(
        multiplier, axis=0
    )
    return week_rows.drop(columns=["gsis_id", "report_status"])


def _apply_rookie_prior(week_rows: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Rescue no-history players (games_played < MIN_GAMES - every rookie, among
    others) from build()'s ordinary MIN_GAMES filter, provided a current depth-chart
    entry exists to project them from.

    Called with build()'s min_games=0, so week_rows still holds these rows - each
    currently equal to the flat positional-average prior (see _prior_mean/
    _prior_rate), identical for a Week 1 starter and a fourth-string body. Where a
    depth-chart rank exists, this rescales volume (and everything derived from it)
    toward DEPTH_RANK_VOLUME_PRIOR, which differentiates by role. Anyone still below
    MIN_GAMES with no depth-chart entry either is dropped, same as build()'s ordinary
    filter - there is no signal to project them from.
    """
    below_min = week_rows.games_played < MIN_GAMES
    if not below_min.any():
        return week_rows

    ranks = src.depth_chart_ranks(season)
    week_rows = week_rows.merge(ranks, on=["player_id", "position"], how="left")
    week_rows["tier"] = week_rows["rank"].clip(upper=3)
    week_rows = week_rows[(week_rows.games_played >= MIN_GAMES) | week_rows["rank"].notna()]
    week_rows = week_rows.merge(DEPTH_RANK_VOLUME_PRIOR, on=["position", "tier"],
                                how="left", suffixes=("", "_prior"))

    apply_rows = (week_rows.games_played < MIN_GAMES) & week_rows["rank"].notna()
    for volume_col, stat_cols in (
        ("e_targets", ["rec_yd", "receptions", "rec_td"]),
        ("e_carries", ["rush_yd", "rush_td"]),
        ("e_attempts", ["pass_yd", "pass_td", "interceptions"]),
    ):
        prior_col = f"{volume_col}_prior"
        scale = (week_rows[prior_col] / week_rows[volume_col].replace(0, np.nan)).fillna(1.0)
        week_rows.loc[apply_rows, stat_cols] = week_rows.loc[apply_rows, stat_cols].mul(
            scale[apply_rows], axis=0
        )
        week_rows.loc[apply_rows, volume_col] = week_rows.loc[apply_rows, prior_col]

    drop_cols = ["rank", "tier"] + [f"e_{c}_prior" for c in ("targets", "carries", "attempts")]
    return week_rows.drop(columns=drop_cols).reset_index(drop=True)


def project(season: int, week: int, seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Projections for a single week, played or not.

    If the week has already been played, its real rows are projected (this is what
    backtesting uses). If it has not - the normal case when betting - placeholder
    rows are built from the published roster and schedule so the upcoming week can
    be projected at all. See _placeholder_rows.

    Sorted by total expected touches (targets + carries + attempts) descending, as a
    neutral "who's featured this week" ordering that falls directly out of the volume
    projections already computed - no scoring convention required. Note this ranks
    QBs above everyone by construction (~30 attempts vs ~8-15 touches); report.py
    ranks within position instead.

    Args:
        season: Season to project.
        week: Week to project. Only earlier weeks inform the projection.
        seasons: Seasons to load for history. Defaults to [season - 1, season] - a
            single-season default would silently return zero rows for weeks 1-2,
            which have no same-season history yet (build() carries a player's data
            across the season boundary, but only if that data was actually loaded).

    No-history players (every rookie among others) are projected from a depth-chart
    role prior instead of being silently dropped - see _apply_rookie_prior - always
    at low confidence, since a depth-chart slot is a guess about role, not evidence
    of production.

    Excludes players ruled Out/Doubtful (discounting Questionable toward
    INJURY_DISCOUNT) once `season` has actually started publishing stats - an
    unstarted season (the normal case when projecting next week's slate in advance)
    has no injury reports yet, so the check would do nothing useful.

    Also raises if nflverse's data is stale (nfl_source.require_fresh()), but only
    for a season that's both started AND not yet complete - i.e. the one actually
    live right now. A fully completed season can never go "more stale" (nothing new
    will ever publish for it), so a retrospective query like project(2025, 5) run
    well after that season ended must not fail just because nflverse's live feed
    happens to be quiet, which is most of the offseason.
    """
    seasons = seasons or [season - 1, season]
    stats_df = src.weekly_stats(seasons)
    schedule_df = src.schedule(seasons)

    # Injury reports only exist once nflverse has actual data for this season - an
    # unstarted season (the normal case projecting next week ahead of time) has none
    # yet, so that check is skipped rather than raising for no reason.
    season_started = season in stats_df.season.unique()

    # Freshness only matters for a season that could still change - i.e. the live,
    # in-progress one. A fully completed season (every game already has a final
    # score) can never become "more stale": nothing new will ever publish for it, so
    # checking nflverse's live feed for it just fails a retrospective query (e.g.
    # project(2025, 5) run any time after the 2025 season ended) for no reason
    # whenever that live feed happens to be quiet - which is most of the offseason.
    # Determined from schedule data already loaded, not today's calendar date -
    # dates are their own landmine (see schedule_captures.py's Eastern-time fix).
    season_schedule = schedule_df[schedule_df.season == season]
    season_complete = not season_schedule.empty and season_schedule.home_score.notna().all()
    if season_started and not season_complete:
        src.require_fresh()

    # Always attempted: _placeholder_rows self-filters to the games in this week that
    # have not been played yet, and returns empty once they all have.
    placeholders = _placeholder_rows(season, week, stats_df, schedule_df)
    if not placeholders.empty:
        stats_df = pd.concat([stats_df, placeholders], ignore_index=True)

    built = build(stats_df, schedule_df, min_games=0)
    week_rows = built[(built.season == season) & (built.week == week)].copy()
    week_rows = _apply_rookie_prior(week_rows, season)
    if season_started:
        week_rows = _apply_injury_status(week_rows, season, week)

    touches = week_rows.e_targets + week_rows.e_carries + week_rows.e_attempts
    return week_rows.assign(e_touches=touches).sort_values(
        "e_touches", ascending=False
    ).reset_index(drop=True)


def kickoff_windows(season: int, week: int, seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Distinct kickoff days for a week, each with a suggested capture/decision time.

    An NFL week is not one event - 2026 Week 1 alone has games on Wednesday, Thursday,
    Sunday, and Monday. A single "midweek + Sunday" capture schedule silently misses
    the Wednesday/Thursday closing line (already played out by the time a Sunday
    sweep runs) and would try to act on Sunday/Monday games too early. This enumerates
    the real windows so capture and betting decisions can be timed per game, not per
    week.

    "suggest_by" is 3 hours before the earliest kickoff in that window, matching
    dk_capture.py's default within_hours=6 with margin to actually place a bet.
    """
    seasons = seasons or [season]
    schedule_df = src.schedule(seasons)
    week_games = schedule_df[(schedule_df.season == season) & (schedule_df.week == week)].copy()
    week_games["kickoff"] = pd.to_datetime(
        week_games.gameday.astype(str) + " " + week_games.gametime.astype(str)
    )
    windows = (week_games.groupby(week_games.kickoff.dt.date)
               .agg(games=("game_id", "count"), earliest_kickoff=("kickoff", "min"),
                   latest_kickoff=("kickoff", "max"))
               .reset_index(names="game_date"))
    windows["suggest_capture_by"] = windows["earliest_kickoff"] - pd.Timedelta(hours=3)
    return windows.sort_values("earliest_kickoff").reset_index(drop=True)


# Gamma dispersion per stat: cv = k / sqrt(mean), so relative spread narrows as the
# projection grows - a player projected for 8 yards is proportionally far more
# volatile than one projected for 65, and holding CV fixed overstates the tail
# exactly where a threshold is likely to bite. Method-of-moments, measured on 2023-24
# (Var(actual) / mean(projection) is ~stable across projection deciles once each
# stat is restricted to players with meaningful volume in it - DISPERSION_VOLUME_FLOOR
# below - since an unthrown QB's ~0 rec_yd projection is real but irrelevant noise
# for calibrating a threshold nobody would ever query for that player).
# Checked (not fit) on 2025: predicted clear-rate runs a few points low in several
# probability bins, inherited from the base model's own small point-estimate bias
# (see backtest.py) rather than a defect in the dispersion fit itself. 2026 in-season
# is the next genuinely clean check - this has not been retuned on 2025.
STAT_DISPERSION = {"pass_yd": 6.624, "rush_yd": 4.895, "rec_yd": 4.797, "receptions": 1.171}
DISPERSION_CV_BOUNDS = (0.35, 2.0)
DISPERSION_VOLUME_FLOOR = {
    "pass_yd": ("e_attempts", 5), "rush_yd": ("e_carries", 2),
    "rec_yd": ("e_targets", 2), "receptions": ("e_targets", 2),
}


def exceed_probability(proj: pd.DataFrame, stat: str, threshold,
                       skip_volume_check: bool = False) -> pd.Series:
    """
    P(stat >= threshold) for each row of a project()/build() result.

    Both Kalshi and DraftKings price props as thresholds ("50+ receiving yards"),
    but build()/project() only output a point estimate. This converts one to the
    other with a gamma distribution (non-negative, right-skewed - a natural fit for
    yardage and receptions), fit by method of moments from the projected mean and
    STAT_DISPERSION. `threshold` may be a single number (applied to every row) or a
    per-row array/Series aligned to `proj`'s index (e.g. each player's own matched
    betting line).

    Only pass_yd/rush_yd/rec_yd/receptions are supported - the stats with both a
    real point projection and an actual prop market captured (see odds_capture.py's
    SERIES / dk_capture.py's MARKETS). Not extended to touchdowns/interceptions:
    their reliability ceiling is 0.09 (see MEASURED_RELIABILITY) - a "probability"
    computed from a nearly-noise point estimate would be false precision on top of
    false precision, and no weekly market for them is captured either.

    Rows below DISPERSION_VOLUME_FLOOR (e.g. a WR's near-zero pass_yd projection)
    return NaN rather than a number: nobody would query a threshold for that
    player/stat, and the dispersion constant was never fit on that population.
    `skip_volume_check` bypasses this for callers that only have the point
    projection itself (no volume columns) but are already implicitly pre-filtered
    to real volume - e.g. scorecard.py recomputing calibration from a settled bet,
    where the player was worth betting on by construction.
    """
    if stat not in STAT_DISPERSION:
        raise ValueError(f"No calibrated dispersion for {stat!r} - only "
                         f"{sorted(STAT_DISPERSION)} are supported.")
    mean = proj[stat].clip(lower=0.01)
    k = STAT_DISPERSION[stat]
    cv = (k / np.sqrt(mean)).clip(*DISPERSION_CV_BOUNDS)
    shape = 1.0 / cv ** 2
    scale = mean / shape
    result = pd.Series(stats.gamma.sf(threshold, a=shape, scale=scale), index=proj.index)
    if skip_volume_check:
        return result

    volume_col, floor = DISPERSION_VOLUME_FLOOR[stat]
    return result.where(proj[volume_col] > floor)
