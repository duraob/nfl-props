"""
DraftKings closing-line capture via The Odds API.

Purpose is narrow: record the line DraftKings closed at, so closing-line value is
computable. You place bets by hand and record your own price; Kalshi supplies
unlimited free intra-week movement. The only thing neither of those gives you is
DK's own number at close, which is what this fetches.

Budget is the binding constraint and the module is built around it:

- The free tier is 500 credits/month. Player props cost one credit per market per
  event, so a careless sweep is expensive.
- /events returns the ENTIRE remaining season (272 events in August). Sweeping it
  unfiltered at 3 markets would cost 816 credits - the whole month's budget, twice
  over, in a single run. `within_hours` is therefore mandatory, not a convenience.
- Empty responses cost nothing, so polling before props are posted is free.

Every spend path checks remaining credits first and refuses rather than overrunning.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
HISTORY = Path("data/odds_history/draftkings.csv")

# Volume props first: usage is the reliable half of the model (0.94-0.98 split-half
# reliability) while yardage mixes in efficiency, which is mostly noise. These are
# where the measured edge lives, and each market costs one credit per event.
MARKETS = ["player_receptions", "player_rush_attempts", "player_pass_attempts"]

BOOKMAKER = "draftkings"
CREDIT_RESERVE = 50  # never spend the last of the month's quota


def _key() -> str:
    values = dotenv_values(".env")
    key = values.get("ODDS_API") or values.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Set ODDS_API in .env (free key from the-odds-api.com)")
    return key


def remaining_credits() -> int:
    """Credits left this month. The /sports endpoint is free to call."""
    response = requests.get(f"{BASE}/sports", params={"apiKey": _key()}, timeout=30)
    response.raise_for_status()
    return int(response.headers.get("x-requests-remaining", 0))


def upcoming_events(within_hours: int) -> list[dict]:
    """
    Events kicking off within the window. Costs 0 credits.

    The window is what keeps a sweep affordable: unfiltered this returns the whole
    season.
    """
    response = requests.get(f"{BASE}/sports/{SPORT}/events",
                            params={"apiKey": _key()}, timeout=30)
    response.raise_for_status()
    now = dt.datetime.now(dt.UTC)
    cutoff = now + dt.timedelta(hours=within_hours)
    events = []
    for event in response.json():
        start = dt.datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        if now <= start <= cutoff:
            events.append(event)
    return events


def event_odds(event_id: str, markets: list[str]) -> tuple[list[dict], int]:
    """Player props for one event. Returns (rows, credits_charged)."""
    response = requests.get(
        f"{BASE}/sports/{SPORT}/events/{event_id}/odds",
        params={"apiKey": _key(), "markets": ",".join(markets),
                "bookmakers": BOOKMAKER, "oddsFormat": "american"},
        timeout=30,
    )
    if not response.ok:
        return [], 0
    charged = int(response.headers.get("x-requests-last", 0))
    payload = response.json()
    captured_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")

    rows = []
    for book in payload.get("bookmakers", []):
        for market in book.get("markets", []):
            for outcome in market.get("outcomes", []):
                rows.append({
                    "captured_at": captured_at,
                    "event_id": event_id,
                    "commence_time": payload.get("commence_time"),
                    "home_team": payload.get("home_team"),
                    "away_team": payload.get("away_team"),
                    "bookmaker": book.get("key"),
                    "market": market.get("key"),
                    "player": outcome.get("description"),
                    "side": outcome.get("name"),      # Over / Under
                    "line": outcome.get("point"),
                    "price": outcome.get("price"),    # american odds
                })
    return rows, charged


def capture(within_hours: int = 6, markets: list[str] | None = None,
            dry_run: bool = False) -> Path | None:
    """
    Capture DK props for games kicking off soon.

    Args:
        within_hours: Only events starting inside this window. Default 6 targets the
            closing line; widen it to see earlier pricing, at proportional cost.
        markets: Prop markets to pull. Each costs 1 credit per event.
        dry_run: Report the cost and change nothing.
    """
    markets = markets or MARKETS
    events = upcoming_events(within_hours)
    cost = len(events) * len(markets)
    left = remaining_credits()

    print(f"events within {within_hours}h : {len(events)}")
    print(f"markets                   : {len(markets)} ({', '.join(markets)})")
    print(f"estimated cost            : {cost} credits")
    print(f"remaining this month      : {left}")

    if not events:
        print("\nNothing kicking off in the window; no credits spent.")
        return None
    if dry_run:
        print("\nDry run - nothing fetched.")
        return None
    if cost > left - CREDIT_RESERVE:
        raise RuntimeError(
            f"Refusing to spend {cost} credits: only {left} left and {CREDIT_RESERVE} "
            f"is reserved. Narrow within_hours or drop a market."
        )

    rows, spent = [], 0
    for event in events:
        event_rows, charged = event_odds(event["id"], markets)
        rows.extend(event_rows)
        spent += charged

    if not rows:
        print(f"\nNo props posted yet. Credits spent: {spent}")
        return None

    df = pd.DataFrame(rows)
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HISTORY, mode="a", header=not HISTORY.exists(), index=False)
    print(f"\nAppended {len(df)} rows for {df.player.nunique()} players -> {HISTORY}")
    print(f"Credits spent: {spent} | remaining: {remaining_credits()}")
    return HISTORY


if __name__ == "__main__":
    import sys

    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    capture(within_hours=hours, dry_run="--dry-run" in sys.argv)
