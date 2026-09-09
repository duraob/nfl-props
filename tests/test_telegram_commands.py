"""
Tests for inbound Telegram commands.

The one that matters most is the chat allowlist: a bot username is discoverable, and
bets.csv is append-only, so a stranger's message becomes a permanent row that is
never rewritten. Parsing is tested for the right-to-left branch, because multi-word
player names are exactly where a left-to-right parser would silently mis-split.
"""

import datetime as dt

import pandas as pd
import pytest

import telegram_commands as TC

TS = "2026-09-13T14:00:00+00:00"


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(TC.L, "BETS", tmp_path / "bets.csv")
    monkeypatch.setattr(TC.L, "RECOMMENDATIONS", tmp_path / "recommendations.csv")
    monkeypatch.setattr(TC.S, "current_week", lambda now=None: (2026, 1))
    return tmp_path


def _sheet(**over):
    base = {"ts_utc": "t1", "season": 2026, "week": 1, "game_date": "2026-09-13",
            "slot": 1, "player_id": "00-0033280", "stat": "receptions",
            "venue": "kalshi", "line": 3.0, "yes_ask": 0.41,
            "model_probability": 0.50, "implied_probability": 0.40,
            "edge_at_ask": 0.09, "spread": 0.01, "depth": 13924.0}
    base.update(over)
    return pd.DataFrame([base])


def test_bet_by_slot_records_the_sheets_own_price(ledger):
    _sheet().to_csv(TC.L.RECOMMENDATIONS, index=False)
    reply = TC.handle("/bet 1 25", TS)
    assert "Recorded" in reply, reply
    row = pd.read_csv(TC.L.BETS).iloc[0]
    assert row.player_id == "00-0033280"
    assert row.line == 3.0
    assert row.price == 0.41
    assert row.stake == 25.0
    assert row.ts_utc == TS, "the message's own timestamp, not collection time"


def test_an_unknown_slot_is_refused_rather_than_guessed(ledger):
    _sheet().to_csv(TC.L.RECOMMENDATIONS, index=False)
    reply = TC.handle("/bet 7 25", TS)
    assert "no entry 7" in reply
    assert not TC.L.BETS.exists(), "nothing may reach an append-only file on a bad ref"


def test_bet_with_no_sheet_yet_is_refused(ledger):
    assert "no bet sheet" in TC.handle("/bet 1 25", TS)


def test_freeform_parses_a_multi_word_name_right_to_left():
    parsed = TC._parse_bet("Amon-Ra St. Brown rec 5 0.55 30".split())
    assert parsed["player_name"] == "Amon-Ra St. Brown"
    assert parsed["stat"] == "receptions"
    assert parsed["line"] == 5.0
    assert parsed["price"] == 0.55
    assert parsed["stake"] == 30.0


def test_freeform_rejects_an_unknown_stat():
    with pytest.raises(ValueError, match="unknown stat"):
        TC._parse_bet("Some Player touchdowns 1 0.5 10".split())


def test_a_dollar_sign_on_the_stake_is_tolerated():
    assert TC._parse_bet(["2", "$25"])["stake"] == 25.0


def test_an_unmatched_player_name_is_refused(ledger):
    reply = TC.handle("/bet Totally Fictional Player rec 3 0.4 10", TS)
    assert "no roster match" in reply
    assert not TC.L.BETS.exists()


def test_only_the_allowed_chat_is_honoured(ledger, monkeypatch):
    """A stranger who finds the bot must not be able to write to the ledger."""
    monkeypatch.setattr(TC, "_token_and_chat", lambda: ("tok", "111"))
    monkeypatch.setattr(TC, "_write_offset", lambda update_id: None)
    _sheet().to_csv(TC.L.RECOMMENDATIONS, index=False)
    sent = []
    monkeypatch.setattr(TC.T, "send_message", lambda text: sent.append(text))
    monkeypatch.setattr(TC, "fetch_updates", lambda token: [{
        "update_id": 1,
        "message": {"chat": {"id": 999}, "text": "/bet 1 25",
                    "date": int(dt.datetime(2026, 9, 13, 14, tzinfo=dt.UTC).timestamp())},
    }])

    TC.run()

    assert not TC.L.BETS.exists(), "an unauthorised chat must never reach the ledger"
    assert sent == [], "and must not get a reply either"


def test_the_allowed_chat_is_processed(ledger, monkeypatch):
    monkeypatch.setattr(TC, "_token_and_chat", lambda: ("tok", "111"))
    monkeypatch.setattr(TC, "_write_offset", lambda update_id: None)
    _sheet().to_csv(TC.L.RECOMMENDATIONS, index=False)
    sent = []
    monkeypatch.setattr(TC.T, "send_message", lambda text: sent.append(text))
    monkeypatch.setattr(TC, "fetch_updates", lambda token: [{
        "update_id": 1,
        "message": {"chat": {"id": 111}, "text": "/bet 1 25",
                    "date": int(dt.datetime(2026, 9, 13, 14, tzinfo=dt.UTC).timestamp())},
    }])

    TC.run()

    assert len(pd.read_csv(TC.L.BETS)) == 1
    assert "Recorded" in sent[0]


def test_offset_persists_so_cron_never_replays_a_wager(tmp_path, monkeypatch):
    """Without a persisted offset every cron run would replay the backlog and record
    the same bet again, permanently, in an append-only file."""
    monkeypatch.setattr(TC, "OFFSET_FILE", tmp_path / "offset")
    assert TC._read_offset() is None
    TC._write_offset(42)
    assert TC._read_offset() == 42
