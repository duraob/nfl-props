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

HEADLINE_STATS = {
    "QB": [("pass_yd", "pass yds"), ("pass_td", "pass TD")],
    "RB": [("rush_yd", "rush yds"), ("receptions", "rec")],
    "WR": [("rec_yd", "rec yds"), ("receptions", "rec")],
    "TE": [("rec_yd", "rec yds"), ("receptions", "rec")],
}

MAX_PER_POSITION = 6


def _edge_lookup(proj: pd.DataFrame, season: int) -> dict[tuple[str, str], pd.Series]:
    """(player_id, stat) -> best available edge row, preferring Kalshi over
    DraftKings where both have a captured line - see "Which venue to bet" in
    CLAUDE.md."""
    lines = M.market_lines(season)
    edges = M.compute_edge(proj, lines)
    if edges.empty:
        return {}
    edges = edges.assign(_venue_rank=edges.venue.map({"kalshi": 0, "draftkings": 1}))
    edges = edges.sort_values("_venue_rank").drop_duplicates(
        subset=["player_id", "stat"], keep="first"
    )
    return {(row.player_id, row.stat): row for _, row in edges.iterrows()}


def _format_player_line(row: pd.Series, edges: dict[tuple[str, str], pd.Series]) -> str:
    stats = HEADLINE_STATS.get(row.position, [("rec_yd", "rec yds")])
    parts = []
    for col, label in stats:
        part = f"{label} {row[col]:.1f}"
        edge_row = edges.get((row.player_id, col))
        if edge_row is not None:
            part += f" (edge {edge_row.edge:+.0%} vs {edge_row.venue} {edge_row.line:g})"
        parts.append(part)
    return (f"*{row.player_display_name}* ({row.team} vs {row.opponent_team}) — "
            f"{', '.join(parts)} [{row.confidence_yardage_label}]")


def build_report(season: int, week: int, game_date: str | None = None,
                 seasons: list[int] | None = None,
                 min_confidence: str = "high") -> str:
    """
    A short, filtered projection report, optionally for a single kickoff day.

    Args:
        season: Season to project.
        week: Week to project.
        game_date: Restrict to one kickoff day (e.g. "2026-09-10" for a Thursday
            slate). Use projections.kickoff_windows(season, week) to find the real
            dates for a given week. None includes the whole week.
        seasons: History to load. Defaults to [season - 1, season] so early-season
            weeks have real trailing data - see projections.py's cross-season note.
        min_confidence: "high" (default) or "medium". "low" is not offered - that
            tier is dominated by touchdown/interception projections, which this
            report is not meant to surface.
    """
    seasons = seasons or [season - 1, season]
    proj = P.project(season, week, seasons=seasons)
    if proj.empty:
        return f"No projections available for {season} week {week}."

    if game_date is not None:
        proj = proj[proj.gameday.astype(str) == str(game_date)]
        if proj.empty:
            return f"No games on {game_date} in {season} week {week}."

    tiers = ["high"] if min_confidence == "high" else ["high", "medium"]
    filtered = proj[proj.position.isin(HEADLINE_STATS) &
                    proj.confidence_yardage_label.isin(tiers)]
    edges = _edge_lookup(proj, season)

    header_date = f" — {game_date}" if game_date else ""
    lines = [f"*Week {week} projections{header_date}*", ""]
    if filtered.empty:
        lines.append("No players meet the confidence bar for this slate.")
    else:
        # Ranked within each position, not globally: pass attempts (~30+) always
        # outrank targets or carries (~8-15) in raw touch count, so a single global
        # sort by e_touches silently produces an all-QB list. See module note.
        for position in ("QB", "RB", "WR", "TE"):
            group = filtered[filtered.position == position].sort_values(
                "e_touches", ascending=False
            ).head(MAX_PER_POSITION)
            if group.empty:
                continue
            lines.append(f"*{position}*")
            for _, row in group.iterrows():
                lines.append(_format_player_line(row, edges))
            lines.append("")
    lines += ["", "_Confidence reflects trust in the number itself. Edge (shown only "
                   "where a market line has been captured) estimates whether it beats "
                   "that price - verify against the live line before acting, not the "
                   "last capture._"]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    week = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    game_date = sys.argv[3] if len(sys.argv) > 3 else None
    print(build_report(season, week, game_date))
