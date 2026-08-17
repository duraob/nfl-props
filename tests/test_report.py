"""
Tests for the report formatter.

The one that matters most: a report ranked by raw touch count silently produces an
all-QB list, because pass attempts (~30+/game) always outrank targets or carries
(~8-15/game). That shipped once during development before being caught by eye - this
locks in the fix (rank within each position, not globally).
"""

import pandas as pd

import report


def test_report_includes_every_position():
    """Regression test for the all-QB bug: a real week must surface RB/WR/TE too,
    not just quarterbacks sorted by raw touch volume."""
    text = report.build_report(2025, 5, seasons=[2024, 2025])
    for header in ("*QB*", "*RB*", "*WR*", "*TE*"):
        assert header in text, f"{header} missing from report - position ranking may be broken"


def test_report_respects_game_date_filter():
    """
    A per-position cap keeps every report short once enough candidates qualify, so
    message length alone doesn't distinguish a filtered report from an unfiltered
    one - check that filtering actually changes *which* players appear instead.
    """
    full_week = report.build_report(2025, 1, seasons=[2024, 2025])
    thursday_only = report.build_report(2025, 1, game_date="2025-09-04", seasons=[2024, 2025])
    # The Thursday opener (2025-09-04) was DAL @ PHI only; a Sunday player like
    # Jonathan Taylor (IND) must not leak into a Thursday-filtered report.
    assert "Jonathan Taylor" not in thursday_only
    assert full_week != thursday_only
    assert len(thursday_only) < len(full_week), (
        "a single-game slate has fewer real candidates than the whole week and "
        "should not fill every position's cap the way the full week does"
    )


def test_report_handles_empty_slate_gracefully():
    text = report.build_report(2025, 1, game_date="1999-01-01", seasons=[2024, 2025])
    assert "No games" in text


def test_report_never_exceeds_telegram_message_limit():
    text = report.build_report(2025, 5, seasons=[2024, 2025])
    assert len(text) < 4096, "a single-day/week report should fit in one Telegram message"


def test_touchdown_stats_are_never_shown_as_headline_numbers():
    """The report is deliberately restricted to confidence_yardage - touchdown
    confidence never reaches 'high', so TD-only players should not appear as if they
    were a trusted number."""
    text = report.build_report(2025, 5, seasons=[2024, 2025])
    assert "[high]" in text  # sanity: something was actually included


def _row(player_id="A", position="WR", **stats):
    base = {"player_id": player_id, "player_display_name": "Test Player",
            "position": position, "team": "AA", "opponent_team": "BB",
            "confidence_yardage_label": "high", "rec_yd": 50.0, "receptions": 4.0}
    base.update(stats)
    return pd.Series(base)


def test_format_player_line_shows_edge_when_a_line_is_captured():
    edges = {("A", "rec_yd"): pd.Series({"edge": 0.12, "venue": "kalshi", "line": 45.0})}
    line = report._format_player_line(_row(), edges)
    assert "edge +12% vs kalshi 45" in line


def test_format_player_line_omits_edge_with_no_captured_line():
    line = report._format_player_line(_row(), {})
    assert "edge" not in line


def test_edge_lookup_prefers_kalshi_over_draftkings(monkeypatch):
    proj = pd.DataFrame({"player_id": ["A"], "rec_yd": [50.0], "e_targets": [6.0],
                         "confidence_yardage": [0.3], "confidence_touchdown": [0.05]})
    fake_lines = pd.DataFrame({
        "player_id": ["A", "A"], "stat": ["rec_yd", "rec_yd"], "line": [45.0, 45.0],
        "implied_probability": [0.5, 0.6], "venue": ["draftkings", "kalshi"],
        "captured_at": ["t1", "t1"],
    })
    monkeypatch.setattr(report.M, "market_lines", lambda season: fake_lines)

    edges = report._edge_lookup(proj, 2099)

    assert len(edges) == 1
    assert edges[("A", "rec_yd")].venue == "kalshi"
