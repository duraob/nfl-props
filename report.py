"""
Formats projections into a short, per-kickoff-window Telegram message.

The design constraint is "I do not have the ammo to take every single bet" - a dump
of 300 projected players is not useful. This filters down two ways:

1. By kickoff window: a report for Thursday's game(s) should not be buried in a wall
   of Sunday players you can't act on for three more days. `projections.gameday`
   (attached in build()) does this filtering directly - no separate team/schedule
   join needed here.
2. By confidence: only "high" confidence yardage projections are listed by default.
   Touchdown/interception confidence never reaches "high" under the measured
   reliability ceiling (0.09) - deliberately, since those numbers are mostly noise.
   Showing them here would just be re-introducing the low-signal stats as a wall of
   text instead of a wall of code.

Confidence is NOT an edge calculation - a "high confidence" projection means we
trust the number itself, not that it beats whatever line a book or Kalshi is
offering. Where a captured market line exists for a shown player's headline stat,
this also shows edge (model probability minus the market's implied probability) -
see market_odds.py. The two answer different questions and are never collapsed into
one number: confidence can be high with no edge (an accurately-priced player), and
edge can be nonzero with low confidence (a noisy number that happens to look off-
market, which is a weaker signal, not a stronger one). Most players will show no
edge simply because no line has been captured for them yet - that's the common case
today, not a bug.
"""

from __future__ import annotations

import pandas as pd

import market_odds as M
import projections as P

STAT_LABELS = {"pass_yd": "PASS YDS", "rush_yd": "RUSH YDS",
               "rec_yd": "REC YDS", "receptions": "RECEPTIONS"}

# Every bound below traces to a specific way this system produced a confident wrong
# answer, not to taste. Changing one means claiming that failure mode is gone.
#
# NEAR_PROJECTION - only rungs where the model's own probability is mid-range. Far
#   out on either tail the answer is dominated by exceed_probability's dispersion fit
#   rather than by the projection, and that fit is the least validated part of the
#   model (see Phase 8 in CLAUDE.md: it runs a few points low in several 2025 bins).
#   A real 219-yard QB projection showed +5% "edge" at a 350-yard rung purely because
#   the gamma tail is too fat - the same defect reads as -17% down at 150.
# EDGE_AT_ASK - measured against the ask, never the mid: you do not transact at the
#   mid, and mid-based edge flatters every quote. The upper bound is the important
#   half. Past ~20 points on a liquid book the disagreement is information you are
#   missing - an inactive, a depth-chart drop, a trade - not a mispricing. A WR4 on a
#   new team showed +37% at every rung while the market priced him near zero; the
#   market was right.
# MAX_SPREAD / MIN_DEPTH - a quote of 0.01/0.72 has a mid of 0.365 that means
#   nothing, and mid-based edge is largest exactly where the book is emptiest.
# A cap on the sheet, for two reasons that happen to want the same number. Telegram
# hard-limits a message at 4096 characters and a 33-bet sheet ran 5206 - split
# mid-block, it is unreadable exactly when it is longest. The bankroll argument is
# the stronger one though: at the 1-2% sizing this model's edge justifies, 33 bets is
# a third of the roll on a single slate, and Sunday's 13 games screen ~12x the volume
# of a Thursday. A sheet you cannot responsibly back in full is not a short list.
MAX_BETS = 10

NEAR_PROJECTION = (0.25, 0.75)
EDGE_AT_ASK = (0.05, 0.20)
MAX_SPREAD = 0.10
MIN_DEPTH = 500


def _screen_bets(proj: pd.DataFrame, season: int) -> tuple[pd.DataFrame, int]:
    """
    (qualifying bets, number of priced lines considered).

    One row per (player, stat) - the best-priced rung - rather than one per rung: a
    player quoted at 30/50/60/70 yards is one decision, not four, and listing every
    rung reads as four independent edges when they share a single projection.

    `agreeing_rungs` counts how many rungs for that same (player, stat) also cleared
    the screen. It is a free calibration signal that falls out of the ladder: a real
    disagreement with the market persists across neighbouring thresholds, while an
    artifact of the dispersion fit flips sign as you walk up them.
    """
    lines = M.market_lines(season)
    edges = M.compute_edge(proj, lines)
    if edges.empty:
        return edges, 0

    edges = edges.assign(edge_at_ask=edges.model_probability - edges.yes_ask)
    low, high = NEAR_PROJECTION
    edge_low, edge_high = EDGE_AT_ASK
    passing = edges[
        edges.model_probability.between(low, high)
        & edges.edge_at_ask.between(edge_low, edge_high)
        & (edges.spread <= MAX_SPREAD)
        & (edges.depth >= MIN_DEPTH)
    ].copy()
    if passing.empty:
        return passing, len(edges)

    counts = passing.groupby(["player_id", "stat"]).size().rename("agreeing_rungs")
    best = passing.sort_values("edge_at_ask", ascending=False).drop_duplicates(
        subset=["player_id", "stat"], keep="first"
    )
    best = best.merge(counts, on=["player_id", "stat"])
    meta = proj[["player_id", "player_display_name", "position", "team",
                 "opponent_team"]].drop_duplicates("player_id")
    best = best.merge(meta, on="player_id", how="left")
    return best.sort_values("edge_at_ask", ascending=False), len(edges)


def _format_bet(index: int, row: pd.Series) -> str:
    """One bet as a block: the action first, the evidence under it."""
    label = STAT_LABELS.get(row.stat, row.stat)
    depth = f"${row.depth / 1000:.1f}k" if row.depth >= 1000 else f"${row.depth:.0f}"
    out = [
        f"*{index}. {row.player_display_name}* — {row.team} {row.position} vs {row.opponent_team}",
        f"   *{row.line:g}+ {label}* · buy at {row.yes_ask * 100:.0f}c",
        f"   model {row.model_probability:.0%} vs mkt {row.implied_probability:.0%} "
        f"→ +{row.edge_at_ask * 100:.0f} pts",
        f"   proj {row.projection:.1f} · spread {row.spread * 100:.0f}c · {depth} deep",
    ]
    if row.agreeing_rungs > 1:
        out.append(f"   {row.agreeing_rungs} nearby lines agree")
    return "\n".join(out)


def build_report(season: int, week: int, game_date: str | None = None,
                 seasons: list[int] | None = None) -> str:
    """
    The bet sheet for one kickoff day: only wagers that clear _screen_bets.

    This is deliberately not a projection listing. The full projection frame is still
    recorded on every run by ledger.log_predictions() - every player, every stat,
    unfiltered - which is what settle.py and scorecard.py grade against actuals. That
    record is the learning loop; this message is the decision, and the two want
    opposite things. A message listing every high-confidence player buried three real
    bets among nineteen rows of noise, including a TE projected for 1.1 receiving
    yards.

    Args:
        season: Season to project.
        week: Week to project.
        game_date: Restrict to one kickoff day (e.g. "2026-09-10"). Use
            projections.kickoff_windows(season, week) for a week's real dates.
        seasons: History to load. Defaults to [season - 1, season] so early-season
            weeks have real trailing data - see projections.py's cross-season note.

    DraftKings lines never clear the screen today: the API publishes no resting size,
    so `depth` is null and MIN_DEPTH excludes them. That is intended rather than
    incidental - DraftKings holds ~4.5% against Kalshi's ~1% (see "Which venue to
    bet"), and a quote whose book cannot be inspected has no place on a sheet whose
    entire job is inspecting books. They remain in market_odds.market_lines() for
    settle.py's closing-line work.
    """
    seasons = seasons or [season - 1, season]
    proj = P.project(season, week, seasons=seasons)
    if proj.empty:
        return f"No projections available for {season} week {week}."

    if game_date is not None:
        proj = proj[proj.gameday.astype(str) == str(game_date)]
        if proj.empty:
            return f"No games on {game_date} in {season} week {week}."

    bets, considered = _screen_bets(proj, season)
    build_report.last_bets = bets.head(0)   # replaced below once the cap is applied
    header_date = f" · {game_date}" if game_date else ""
    lines = [f"*WEEK {week}{header_date}*"]

    if bets.empty:
        # Said out loud rather than sending nothing: silence is what a broken cron
        # looks like, and the two must never be confusable from the phone.
        lines += ["", f"No bets clear the screen ({considered} priced lines checked)."]
        return "\n".join(lines)

    shown = bets.head(MAX_BETS)
    build_report.last_bets = shown   # slots must match what was actually sent
    summary = f"{len(shown)} bet{'s' if len(shown) > 1 else ''} from {considered} lines screened"
    if len(bets) > len(shown):
        summary += f" (top {MAX_BETS} of {len(bets)} qualifying)"
    lines += [summary, ""]
    for index, (_, row) in enumerate(shown.iterrows(), start=1):
        lines += [_format_bet(index, row), ""]
    lines += ["_Edge is measured at the ask, so it is what you would actually pay. "
              "Verify the live book before acting - these are last-capture prices._",
              "", "_Record a wager with_ `/bet <number> <stake>`"]
    return "\n".join(lines)
