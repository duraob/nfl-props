"""
Tests for odds capture.

These guard two things that cost real money or real data if they break: the credit
budget, and the Kalshi orderbook parse. Neither is exercised by the projection tests.

Network is stubbed throughout - capture correctness should not depend on live markets,
and the test suite must not burn API credits.
"""

import datetime as dt

import pytest

import dk_capture as D
import odds_capture as K


# --------------------------------------------------------------------------
# Kalshi orderbook parsing
# --------------------------------------------------------------------------

class _Response:
    def __init__(self, payload, ok=True):
        self._payload, self.ok = payload, ok

    def json(self):
        return self._payload


def test_kalshi_derives_ask_from_the_no_side(monkeypatch):
    """
    Kalshi returns bids on BOTH sides and never an ask. A NO bid at p is a YES ask
    at 1-p. Getting this wrong silently produces a nonsense spread.
    """
    book = {"orderbook_fp": {
        "yes_dollars": [["0.51", "333"], ["0.56", "2574"]],
        "no_dollars": [["0.41", "6376"], ["0.43", "2084"]],
    }}
    monkeypatch.setattr(K.requests, "get", lambda *a, **k: _Response(book))
    out = K.fetch_orderbook("ANY")
    assert out["yes_bid"] == 0.56           # best (highest) yes bid
    assert out["yes_ask"] == pytest.approx(0.57)   # 1 - best no bid (0.43)
    assert out["spread"] == pytest.approx(0.01)
    assert out["mid"] == pytest.approx(0.565)


def test_kalshi_handles_one_sided_book(monkeypatch):
    """Many thin markets have no bid at all; that must not raise."""
    book = {"orderbook_fp": {"yes_dollars": [], "no_dollars": [["0.30", "100"]]}}
    monkeypatch.setattr(K.requests, "get", lambda *a, **k: _Response(book))
    out = K.fetch_orderbook("ANY")
    assert out["yes_bid"] is None
    assert out["yes_ask"] == pytest.approx(0.70)
    assert out["spread"] is None


def test_kalshi_handles_empty_book(monkeypatch):
    monkeypatch.setattr(K.requests, "get", lambda *a, **k: _Response({}))
    out = K.fetch_orderbook("ANY")
    assert out["yes_bid"] is None and out["yes_ask"] is None


# --------------------------------------------------------------------------
# The Odds API budget guard
# --------------------------------------------------------------------------

def _events(n: int) -> list[dict]:
    soon = dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)
    return [{"id": f"e{i}", "commence_time": soon.isoformat().replace("+00:00", "Z")}
            for i in range(n)]


def test_budget_guard_refuses_oversized_sweep(monkeypatch):
    """
    /events returns the whole season (272 games in August). Sweeping it unfiltered
    at 3 markets costs 816 credits against a 500/month free tier. The guard must
    refuse rather than overrun.
    """
    monkeypatch.setattr(D, "upcoming_events", lambda h: _events(272))
    monkeypatch.setattr(D, "remaining_credits", lambda: 500)
    with pytest.raises(RuntimeError, match="Refusing to spend"):
        D.capture(within_hours=8760)


def test_budget_guard_respects_the_reserve(monkeypatch):
    """A sweep that fits the quota but eats the reserve must still be refused."""
    monkeypatch.setattr(D, "upcoming_events", lambda h: _events(16))
    monkeypatch.setattr(D, "remaining_credits", lambda: 60)   # 48 needed, 50 reserved
    with pytest.raises(RuntimeError, match="Refusing to spend"):
        D.capture(within_hours=6)


def test_affordable_sweep_is_allowed(monkeypatch):
    monkeypatch.setattr(D, "upcoming_events", lambda h: _events(16))
    monkeypatch.setattr(D, "remaining_credits", lambda: 500)
    monkeypatch.setattr(D, "event_odds", lambda eid, mk: ([], 0))
    D.capture(within_hours=6)   # 48 credits of 500: fine, and returns None with no props


def test_empty_window_spends_nothing(monkeypatch):
    monkeypatch.setattr(D, "upcoming_events", lambda h: [])
    monkeypatch.setattr(D, "remaining_credits", lambda: 500)

    def _boom(*a, **k):
        raise AssertionError("must not fetch odds when no events are in the window")

    monkeypatch.setattr(D, "event_odds", _boom)
    assert D.capture(within_hours=6) is None


def test_dry_run_never_spends(monkeypatch):
    monkeypatch.setattr(D, "upcoming_events", lambda h: _events(16))
    monkeypatch.setattr(D, "remaining_credits", lambda: 500)

    def _boom(*a, **k):
        raise AssertionError("dry run fetched odds")

    monkeypatch.setattr(D, "event_odds", _boom)
    assert D.capture(within_hours=6, dry_run=True) is None
