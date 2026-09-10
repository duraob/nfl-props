"""
Tests for the bet-sheet report.

The screen is the whole product here, so most of these pin one rejection rule each,
and each rule exists because this system produced a confident wrong answer without
it - see report.py's constants for the incidents. The formatter is tested only for
the parts a reader acts on: the threshold, the side, and the price.
"""

import pandas as pd

import report


def _proj(**over):
    base = {"player_id": ["A"], "player_display_name": ["Test Player"],
            "position": ["WR"], "team": ["AA"], "opponent_team": ["BB"],
            "rec_yd": [50.0], "receptions": [4.0], "e_targets": [6.0],
            "confidence_yardage": [0.35], "confidence_touchdown": [0.05]}
    base.update(over)
    return pd.DataFrame(base)


def _lines(line=45.0, implied=0.38, ask=0.40, spread=0.03, depth=5000.0, stat="rec_yd",
           venue="kalshi"):
    return pd.DataFrame({
        "player_id": ["A"], "stat": [stat], "line": [line],
        "implied_probability": [implied], "yes_ask": [ask], "spread": [spread],
        "depth": [depth], "venue": [venue], "captured_at": ["t1"],
    })


def _screen(monkeypatch, lines, proj=None):
    monkeypatch.setattr(report.M, "market_lines", lambda season: lines)
    return report._screen_bets(proj if proj is not None else _proj(), 2099)


def test_a_qualifying_line_becomes_a_bet(monkeypatch):
    bets, considered = _screen(monkeypatch, _lines())
    assert considered == 1
    assert len(bets) == 1
    assert bets.iloc[0].edge_at_ask > 0


def test_a_tail_rung_is_rejected_however_large_its_edge(monkeypatch):
    """
    The Drake Maye case. A 219-yard projection showed +5% "edge" at a 350-yard rung
    purely because the gamma tail is too fat - the same defect reads as -17% at 150.
    Edge far from the projection measures the dispersion fit, not the market.
    """
    bets, considered = _screen(monkeypatch, _lines(line=350.0, implied=0.05, ask=0.06))
    assert considered == 1, "the line was priced and considered"
    assert bets.empty, "a rung the model puts outside 25-75% must never be offered"


def test_an_implausibly_large_edge_is_rejected(monkeypatch):
    """
    The Mack Hollins case. A WR4 on a new team showed +37% at every rung while the
    market priced him near zero. Past the cap the market knows something the model
    does not - an inactive, a depth-chart drop - and the model is the wrong one.
    """
    bets, _ = _screen(monkeypatch, _lines(implied=0.10, ask=0.11))
    assert bets.empty, "a 20+ point edge is a red flag, not the best bet on the board"


def test_a_wide_book_is_rejected(monkeypatch):
    """Bucky Irving quoted 0.01/0.72: the mid is arithmetic, not a price."""
    bets, _ = _screen(monkeypatch, _lines(spread=0.71, ask=0.30))
    assert bets.empty


def test_a_shallow_book_is_rejected(monkeypatch):
    bets, _ = _screen(monkeypatch, _lines(depth=50.0))
    assert bets.empty


def test_draftkings_never_clears_the_screen(monkeypatch):
    """DraftKings publishes no resting size, so depth is null and MIN_DEPTH drops it -
    intended, not incidental. See build_report's docstring."""
    dk = _lines(venue="draftkings", depth=float("nan"))
    bets, considered = _screen(monkeypatch, dk)
    assert considered == 1
    assert bets.empty


def test_one_bet_per_player_stat_with_the_agreeing_rung_count(monkeypatch):
    """
    A player quoted at four rungs is one decision, not four. The count of rungs that
    also cleared is kept, because a real disagreement persists across neighbouring
    thresholds while a dispersion artifact flips sign as you walk up them.
    """
    ladder = pd.concat([
        _lines(line=40.0, implied=0.44, ask=0.46),   # model 0.534 -> +0.074
        _lines(line=45.0, implied=0.38, ask=0.40),   # model 0.469 -> +0.069
        _lines(line=50.0, implied=0.28, ask=0.30),   # model 0.410 -> +0.110, best
    ], ignore_index=True)
    bets, _ = _screen(monkeypatch, ladder)
    assert len(bets) == 1, "one row per (player, stat)"
    assert bets.iloc[0].agreeing_rungs == 3
    assert bets.iloc[0].line == 50.0, (
        "the surviving rung must be the best-priced one - the original bug printed "
        "whichever row drop_duplicates happened to keep, which is how a 219-yard "
        "projection got reported against a 350-yard line"
    )


def test_bet_block_states_the_action_the_threshold_and_the_price(monkeypatch):
    bets, _ = _screen(monkeypatch, _lines())
    text = report._format_bet(1, bets.iloc[0])
    assert "45+ REC YDS" in text, "the threshold and stat must be unmissable"
    assert "buy at 40c" in text, "the price shown is the ask actually paid"
    assert "Test Player" in text


def test_an_empty_slate_says_how_many_lines_were_checked(monkeypatch):
    """Silence is what a broken cron looks like. A screened-and-empty slate has to be
    distinguishable from no message at all, from a phone."""
    monkeypatch.setattr(report.P, "project", lambda season, week, seasons=None: _proj().assign(
        gameday="2099-09-10", season=season, week=week))
    monkeypatch.setattr(report.M, "market_lines", lambda season: _lines(depth=1.0))
    text = report.build_report(2099, 1)
    assert "No bets clear the screen" in text
    assert "1 priced lines checked" in text


def test_report_handles_an_empty_slate_gracefully():
    assert "No games" in report.build_report(2025, 1, game_date="1999-01-01",
                                             seasons=[2024, 2025])


def test_report_never_exceeds_telegram_message_limit():
    """
    Telegram hard-limits a message at 4096 characters, and send_message splits on
    that boundary - which lands mid-block precisely when the sheet is longest. An
    uncapped screen produced 33 bets and 5206 characters against real captured lines.
    """
    text = report.build_report(2025, 5, seasons=[2024, 2025])
    assert len(text) < 4096, "a bet sheet must fit in one Telegram message"


def test_the_sheet_is_capped_and_says_so(monkeypatch):
    """
    The cap is a bankroll rule before it is a formatting one: at the 1-2% sizing this
    model's edge justifies, backing every qualifying bet on a 13-game Sunday would
    commit a third of the roll to one slate. Saying "top 10 of N" matters too - a
    silent cap reads as "only 10 qualified".
    """
    ladder = pd.concat([_lines(line=45.0 + i * 0.01) for i in range(report.MAX_BETS + 5)],
                       ignore_index=True)
    ladder["player_id"] = [f"P{i}" for i in range(len(ladder))]
    proj = pd.concat([_proj(player_id=[f"P{i}"]) for i in range(len(ladder))],
                     ignore_index=True)
    proj["gameday"] = "2099-09-10"
    monkeypatch.setattr(report.P, "project",
                        lambda season, week, seasons=None: proj.assign(season=season, week=week))
    monkeypatch.setattr(report.M, "market_lines", lambda season: ladder)

    text = report.build_report(2099, 1)

    assert text.count("· buy at") == report.MAX_BETS
    assert f"top {report.MAX_BETS} of {len(ladder)} qualifying" in text
    assert len(report.build_report.last_bets) == report.MAX_BETS, (
        "/bet slot numbers must resolve against what was actually sent"
    )
