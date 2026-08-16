"""
Kalshi odds capture: append-only, no analysis.

This module deliberately does nothing clever. Its only job is to never lose data.
Betting lines cannot be reconstructed after the fact, so every week without capture
is a permanently missing week of closing-line-value evidence - the one signal that
tells you whether an edge is real before hundreds of settled bets accumulate.

Kalshi specifics learned from the live API:

- The /markets endpoint returns null for yes_bid/yes_ask/volume. Prices live only in
  /markets/{ticker}/orderbook. Reading the list endpoint alone silently yields nothing.
- The orderbook returns bids on BOTH sides, never asks. A NO bid at p is mechanically
  a YES ask at 1-p, which is how the spread is derived below.
- Market data needs no authentication. KALSHI_API (an API key id) is only required to
  trade, and signing additionally needs the matching RSA private key.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
HISTORY = Path("data/odds_history/kalshi.csv")

# Game markets are liquid but efficient; season player props are wide but thin.
# Both are captured because which one carries an edge is an empirical question that
# needs history to answer.
SERIES = [
    # Weekly per-game player props. These are the target: the model's measured edge is
    # in player volume, and Kalshi prices them as threshold binaries ("50+ receiving
    # yards"), which is exactly what _exceed_probability already emits - no translation.
    # They sit at zero open markets between slates and reopen near gameday, so capture
    # has to run on a schedule rather than being judged by a one-off look.
    "KXNFLRECYDS",
    "KXNFLRSHYDS",
    "KXNFLPASSYDS",
    "KXNFLREC",             # receptions - series exists, never yet observed with markets
    "KXNFLRSHATT",          # rushing attempts - same
    "KXNFLPASSATT",
    "KXNFLPASSCOMP",
    # Game markets: liquid but efficient. Captured as a fair-value reference, not to bet.
    "KXNFLGAME",
    "KXNFLSPREAD",
    "KXNFLTOTAL",
    # Season-long player props: wide and thin, but a usage model projects them well.
    "KXNFLSEASONPASSYDS",
    "KXNFLSEASONRECTD",
    "KXNFLSEASONRSHTD",
    "KXNFLSEASONPASSTDS",
]

REQUEST_PAUSE = 0.12  # ~8 req/s, comfortably inside Kalshi's read limits


def fetch_markets(series_ticker: str, status: str = "open") -> list[dict]:
    """All markets for a series, following pagination."""
    out, cursor = [], None
    while True:
        params = {"series_ticker": series_ticker, "status": status, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        response = requests.get(f"{BASE}/markets", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        out.extend(payload.get("markets", []))
        cursor = payload.get("cursor")
        if not cursor:
            return out
        time.sleep(REQUEST_PAUSE)


def fetch_orderbook(ticker: str, depth: int = 10) -> dict | None:
    """Best prices and resting size for one market, or None if the book is empty."""
    response = requests.get(f"{BASE}/markets/{ticker}/orderbook",
                            params={"depth": depth}, timeout=30)
    if not response.ok:
        return None
    book = (response.json() or {}).get("orderbook_fp") or {}
    yes = [(float(p), float(q)) for p, q in (book.get("yes_dollars") or [])]
    no = [(float(p), float(q)) for p, q in (book.get("no_dollars") or [])]

    best_yes_bid = max((p for p, _ in yes), default=None)
    best_no_bid = max((p for p, _ in no), default=None)
    # A NO bid at p is a YES ask at 1-p. This is the only way to get an ask from Kalshi.
    yes_ask = round(1.0 - best_no_bid, 4) if best_no_bid is not None else None

    return {
        "yes_bid": best_yes_bid,
        "yes_ask": yes_ask,
        "spread": round(yes_ask - best_yes_bid, 4) if (yes_ask and best_yes_bid) else None,
        "mid": round((yes_ask + best_yes_bid) / 2, 4) if (yes_ask and best_yes_bid) else None,
        "yes_depth": round(sum(q for _, q in yes), 2),
        "no_depth": round(sum(q for _, q in no), 2),
    }


def snapshot(series: list[str] | None = None) -> pd.DataFrame:
    """One capture pass over every configured series."""
    series = series or SERIES
    captured_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    rows = []

    for series_ticker in series:
        try:
            markets = fetch_markets(series_ticker)
        except requests.RequestException as exc:
            print(f"  {series_ticker}: fetch failed ({type(exc).__name__}), skipping")
            continue

        for market in markets:
            ticker = market["ticker"]
            book = fetch_orderbook(ticker)
            time.sleep(REQUEST_PAUSE)
            if book is None:
                continue
            rows.append({
                "captured_at": captured_at,
                "series": series_ticker,
                "ticker": ticker,
                "title": market.get("title"),
                "close_time": market.get("close_time"),
                **book,
            })
        print(f"  {series_ticker:22s} {len(markets):4d} markets")

    return pd.DataFrame(rows)


def capture(series: list[str] | None = None) -> Path:
    """Take a snapshot and append it to the history file. Never rewrites past rows."""
    print(f"Kalshi capture starting {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC")
    df = snapshot(series)
    if df.empty:
        print("No rows captured; history left unchanged.")
        return HISTORY

    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HISTORY, mode="a", header=not HISTORY.exists(), index=False)
    priced = df.spread.notna().sum()
    print(f"\nAppended {len(df)} rows ({priced} with a two-sided book) -> {HISTORY}")
    return HISTORY


if __name__ == "__main__":
    capture()
