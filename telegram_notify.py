"""
Telegram push delivery.

Deliberately minimal: one function, one HTTP call. Push only, no interactive
polling/commands - decisions in this project happen at specific kickoff-adjacent
moments (see projections.kickoff_windows), so a message arriving at the right time
matters more than a chat interface to query on demand. That can be added later
behind the same bot token if it turns out to be worth the extra surface area.

TELEGRAM_API is the bot token from @BotFather (shape: "<bot_id>:<secret>").
TELEGRAM_BOT_TOKEN in .env is NOT a valid token by itself - it's just the numeric id
prefix, left over from a partial paste. TELEGRAM_CHAT_ID is discovered once via
/getUpdates after messaging the bot directly, then reused.
"""

from __future__ import annotations

import requests
from dotenv import dotenv_values

BASE = "https://api.telegram.org/bot{token}"
MAX_MESSAGE_LENGTH = 4096  # Telegram's hard limit per message


def _config() -> tuple[str, str]:
    values = dotenv_values(".env")
    token = values.get("TELEGRAM_API")
    chat_id = values.get("TELEGRAM_CHAT_ID")
    if not token:
        raise RuntimeError("TELEGRAM_API missing from .env (bot token from @BotFather)")
    if not chat_id:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID missing from .env. Message your bot once, then run "
            "`python -c \"import telegram_notify as t; print(t.discover_chat_id())\"` "
            "to find it."
        )
    return token, chat_id


def discover_chat_id() -> int | None:
    """Look up the chat_id of whoever last messaged the bot. Needs no chat_id itself."""
    values = dotenv_values(".env")
    token = values.get("TELEGRAM_API")
    response = requests.get(f"{BASE.format(token=token)}/getUpdates", timeout=15)
    response.raise_for_status()
    updates = response.json().get("result", [])
    if not updates:
        return None
    return updates[-1]["message"]["chat"]["id"]


def send_message(text: str) -> dict:
    """
    Push a message to the configured chat.

    Splits on MAX_MESSAGE_LENGTH rather than truncating - a cut-off bet list is worse
    than two messages.
    """
    token, chat_id = _config()
    url = f"{BASE.format(token=token)}/sendMessage"
    chunks = [text[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(text), MAX_MESSAGE_LENGTH)] or [""]

    last = {}
    for chunk in chunks:
        response = requests.post(url, data={
            "chat_id": chat_id, "text": chunk, "parse_mode": "Markdown",
        }, timeout=15)
        response.raise_for_status()
        last = response.json()
    return last


if __name__ == "__main__":
    result = send_message("Telegram delivery is wired up and working.")
    print("sent:", result.get("ok"))
