"""
Tests for telegram_notify.py.

All network calls are mocked - this suite must never actually message the bot.
"""

import pytest

import telegram_notify as T


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_missing_token_raises(monkeypatch):
    monkeypatch.setattr(T, "dotenv_values", lambda path: {"TELEGRAM_CHAT_ID": "123"})
    with pytest.raises(RuntimeError, match="TELEGRAM_API"):
        T.send_message("hello")


def test_missing_chat_id_raises(monkeypatch):
    monkeypatch.setattr(T, "dotenv_values", lambda path: {"TELEGRAM_API": "tok"})
    with pytest.raises(RuntimeError, match="TELEGRAM_CHAT_ID"):
        T.send_message("hello")


def test_short_message_sends_once(monkeypatch):
    monkeypatch.setattr(T, "dotenv_values", lambda path: {"TELEGRAM_API": "tok", "TELEGRAM_CHAT_ID": "1"})
    calls = []
    monkeypatch.setattr(T.requests, "post", lambda url, data, timeout: calls.append(data) or _Response({"ok": True}))
    T.send_message("short message")
    assert len(calls) == 1
    assert calls[0]["text"] == "short message"


def test_long_message_is_chunked_not_truncated(monkeypatch):
    """
    A cut-off bet list is worse than two messages. Confirms a message over Telegram's
    4096-char limit is split into multiple sends rather than silently truncated.
    """
    monkeypatch.setattr(T, "dotenv_values", lambda path: {"TELEGRAM_API": "tok", "TELEGRAM_CHAT_ID": "1"})
    calls = []
    monkeypatch.setattr(T.requests, "post", lambda url, data, timeout: calls.append(data) or _Response({"ok": True}))

    long_text = "x" * 5000
    T.send_message(long_text)

    assert len(calls) == 2, "a 5000-char message should split into 2 sends of <=4096 each"
    rejoined = "".join(c["text"] for c in calls)
    assert rejoined == long_text, "chunking must not drop or reorder any content"
    assert all(len(c["text"]) <= T.MAX_MESSAGE_LENGTH for c in calls)
