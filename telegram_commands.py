"""
Inbound Telegram commands for the bet ledger.

telegram_notify.py is push-only by design (see its docstring). This is the "later"
that note anticipated, and it stays deliberately small: recording a wager is the one
thing that genuinely cannot happen from the droplet, because only you know what you
actually filled at.

Polled from cron rather than run as a daemon. getUpdates is one cheap HTTP call, cron
already exists on this box, and a long-running process would be a third thing to
supervise for no gain. Every ~2 minutes is fast enough - you are recording a bet you
already placed, not racing a market.

Two decisions worth keeping:

- **Only ALLOWED_CHAT is honoured.** A bot's username is discoverable and anyone can
  message it. Without this check a stranger writes rows into an append-only ledger
  that is never rewritten, and the corruption would be permanent and silent.
- **No /undo.** bets.csv is append-only for the same reason predictions.csv is: a
  ledger that can retract its own history cannot be trusted to grade anything. Fix a
  typo in the file by hand, where it leaves a visible edit.

Commands:
    /bet <n> <stake>                       back sheet entry n
    /bet <player> <stat> <line> <price> <stake>
    /bets                                  this week's recorded wagers
    /help
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

import ledger as L
import schedule_captures as S
import telegram_notify as T

OFFSET_FILE = Path("data/scheduler_state/telegram_offset")
STAT_ALIASES = {
    "rec": "receptions", "receptions": "receptions", "catches": "receptions",
    "recyd": "rec_yd", "rec_yd": "rec_yd", "recyds": "rec_yd",
    "rushyd": "rush_yd", "rush_yd": "rush_yd", "rushyds": "rush_yd",
    "passyd": "pass_yd", "pass_yd": "pass_yd", "passyds": "pass_yd",
}
HELP = (
    "*Recording wagers*\n"
    "`/bet 2 25` — back entry 2 from the latest sheet, $25\n"
    "`/bet Colby Parkinson rec 3 0.41 25` — freeform\n"
    "`/bets` — this week's recorded wagers\n\n"
    "_Freeform is read right-to-left (stake, price, line, stat), so multi-word "
    "names need no quoting. Price is what you filled at, not the screen's ask._"
)


def _token_and_chat() -> tuple[str, str]:
    values = dotenv_values(".env")
    token, chat = values.get("TELEGRAM_API"), values.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        raise RuntimeError("TELEGRAM_API and TELEGRAM_CHAT_ID must both be set in .env")
    return token, chat


def _read_offset() -> int | None:
    if not OFFSET_FILE.exists():
        return None
    text = OFFSET_FILE.read_text().strip()
    return int(text) if text else None


def _write_offset(update_id: int) -> None:
    OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    OFFSET_FILE.write_text(str(update_id))


def fetch_updates(token: str) -> list[dict]:
    """
    Pending messages since the last one handled.

    The offset is persisted rather than held in memory because cron re-invokes this
    stateless every time; without it every run would replay the whole backlog and
    record each wager again. Telegram also drops anything already acknowledged by
    offset, so this doubles as the server-side delete.
    """
    params = {"timeout": 0}
    offset = _read_offset()
    if offset is not None:
        params["offset"] = offset + 1
    response = requests.get(f"{T.BASE.format(token=token)}/getUpdates",
                            params=params, timeout=20)
    response.raise_for_status()
    return response.json().get("result", [])


def _resolve_stat(word: str) -> str:
    key = word.lower().replace(" ", "").replace("-", "")
    if key not in STAT_ALIASES:
        raise ValueError(f"unknown stat {word!r} - use rec, rec_yd, rush_yd or pass_yd")
    return STAT_ALIASES[key]


def _parse_bet(args: list[str]) -> dict:
    """
    Either `<slot> <stake>` or `<player words...> <stat> <line> <price> <stake>`.

    The freeform branch is read right-to-left: the four trailing fields are fixed in
    number, so whatever precedes them is the player's name however many words it
    runs to. Parsing left-to-right would need quoting, and quoting on a phone
    keyboard is how a wrong row ends up in an append-only file.
    """
    if len(args) == 2 and args[0].isdigit():
        return {"slot": int(args[0]), "stake": float(args[1].lstrip("$"))}
    if len(args) < 5:
        raise ValueError("need `/bet <n> <stake>` or "
                         "`/bet <player> <stat> <line> <price> <stake>`")
    stake, price, line = float(args[-1].lstrip("$")), float(args[-2]), float(args[-3])
    return {"player_name": " ".join(args[:-4]), "stat": _resolve_stat(args[-4]),
            "line": line, "price": price, "stake": stake}


def _bet_from_slot(slot: int, stake: float, season: int, week: int) -> tuple[dict, str]:
    sheet = L.latest_recommendations()
    if sheet.empty:
        raise ValueError("no bet sheet has been sent yet")
    match = sheet[sheet.slot == slot]
    if match.empty:
        raise ValueError(f"no entry {slot} on the latest sheet (1-{int(sheet.slot.max())})")
    row = match.iloc[0]
    return ({"season": int(row.season), "week": int(row.week), "player_id": row.player_id,
             "stat": row.stat, "venue": row.venue, "side": "over",
             "line": float(row.line), "price": float(row.yes_ask), "stake": stake},
            f"{row.line:g}+ {row.stat}")


def _bet_from_words(parsed: dict, season: int, week: int) -> tuple[dict, str]:
    import nfl_source as src
    names = src.rosters([season]).set_index("full_name").gsis_id
    if parsed["player_name"] not in names.index:
        raise ValueError(f"no roster match for {parsed['player_name']!r} - check spelling")
    return ({"season": season, "week": week,
             "player_id": names[parsed["player_name"]], "stat": parsed["stat"],
             "venue": "kalshi", "side": "over", "line": parsed["line"],
             "price": parsed["price"], "stake": parsed["stake"]},
            f"{parsed['line']:g}+ {parsed['stat']}")


def handle(text: str, message_ts: str) -> str:
    """One command in, one reply out. Never raises - the reply carries the error."""
    parts = text.strip().split()
    if not parts:
        return ""
    command, args = parts[0].lower().split("@")[0], parts[1:]

    if command in ("/help", "/start"):
        return HELP
    if command == "/bets":
        if not L.BETS.exists():
            return "No wagers recorded yet."
        bets = pd.read_csv(L.BETS).tail(10)
        rows = [f"{b.stat} {b.line:g}+ @ {b.price:g} — ${b.stake:g}"
                for b in bets.itertuples()]
        return "*Recent wagers*\n" + "\n".join(rows)
    if command != "/bet":
        return ""

    try:
        week_info = S.current_week()
        if week_info is None:
            return "No current NFL week - nothing to record against."
        season, week = week_info
        parsed = _parse_bet(args)
        if "slot" in parsed:
            fields, described = _bet_from_slot(parsed["slot"], parsed["stake"], season, week)
        else:
            fields, described = _bet_from_words(parsed, season, week)
        L.record_bet(**fields, ts_utc=message_ts)
        # Sync straight away rather than waiting for the next capture window.
        # sync_captured_data() is otherwise only reached from inside a kickoff
        # window, so a wager placed after a week's last window sat on this machine
        # alone for days - two of Week 1's seven did exactly that. Best-effort: the
        # bet is already durably on local disk, and failing to push it must never
        # turn into "could not record that" on the phone.
        try:
            S.sync_captured_data()
        except Exception as exc:
            print(f"bet recorded but git sync failed: {type(exc).__name__}: {exc}")
        return (f"Recorded: {described} @ {fields['price']:g} "
                f"for ${fields['stake']:g} ({fields['venue']}).")
    except Exception as exc:
        return f"Could not record that: {exc}"


def run() -> None:
    """One polling pass. Safe to call from cron as often as you like."""
    token, allowed_chat = _token_and_chat()
    for update in fetch_updates(token):
        _write_offset(update["update_id"])
        message = update.get("message") or {}
        text = message.get("text", "")
        if str(message.get("chat", {}).get("id")) != str(allowed_chat):
            print(f"ignored message from unauthorised chat {message.get('chat', {}).get('id')}")
            continue
        if not text.startswith("/"):
            continue
        ts = dt.datetime.fromtimestamp(message["date"], dt.UTC).isoformat(timespec="seconds")
        reply = handle(text, ts)
        if reply:
            T.send_message(reply)
            print(f"handled {text.split()[0]}")


if __name__ == "__main__":
    run()
