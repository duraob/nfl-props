# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding protocol (from `.cursorrules`)

- Write the absolute minimum code necessary to solve the problem. Method-based modules, not class-based, for new code.
- No sweeping changes, no unrelated edits. Remove redundant/obsolete methods rather than leaving them.
- **Do not use fallback methods or mock data.** If data is missing, fail loudly.
- Activate the virtualenv before running anything.
- If the user must configure something (API key, data file), say so clearly.
- After implementing, tell the user how to test it.

## Environment

Always activate the venv first — every command below assumes it:

```bash
source .venv/bin/activate          # note: forward slash, this is macOS
pip install -r requirements.txt
python -m pytest tests/ -q
```

Python is 3.14. This matters: **`nfl_data_py` cannot be installed** on 3.14 (it pins an
old pandas with no 3.14 wheel and fails to build). Use **`nflreadpy`**, the current
nflverse package, which works and is already pinned in `requirements.txt`.

`.env` at repo root (gitignored):

```
ODDS_API=...      # The Odds API — free tier: 500 req/month. See dk_capture.py.
KALSHI_API=...    # Kalshi API key id. Market data needs no auth; this is unused
                  # until/unless the project starts placing Kalshi trades.
```

### Tests

`python -m pytest tests/ -q` — all pass. `tests/test_data_source.py` and
`tests/test_nfl_source.py` guard the nflverse contract (should always pass; a failure
means upstream nflverse changed under us). `tests/test_projections.py` guards the
model, including `test_no_future_leakage`, which has already caught one real bug — see
its docstring before changing anything in `projections.py`'s prior-computation
functions. `tests/test_capture.py` guards the Kalshi orderbook parse and the Odds API
credit-budget guard with mocked network calls; it costs nothing to run.

## Running things

All commands run **from the repo root** (every path in the codebase is CWD-relative).

```bash
python -m pytest tests/ -q                      # full suite
python backtest.py                               # tuning + holdout scorecard (the gate)
python -c "import projections as P; print(P.kickoff_windows(2026, 1))"  # when to act
python odds_capture.py                            # Kalshi snapshot, append-only, free
python dk_capture.py 6 --dry-run                  # DK cost estimate, spends nothing
python dk_capture.py 6                            # DK closing lines, games within 6h
python -c "import report; print(report.build_report(2026, 1))"          # formatted report, edge included where captured
python -c "import market_odds as M; print(M.market_lines(2026))"        # normalized lines from both venues, direct
python -c "import projections as P, ledger as L; L.log_predictions(P.project(2026, 1))"  # log before kickoff
python settle.py 2026 1                           # grade a played week's bets
python schedule_captures.py                       # one poll; no-ops unless a window is due right now
python scorecard.py 2026 1                        # print last week's accuracy/CLV/calibration
python scorecard.py --push                        # same, for the most recently completed week, sent to Telegram
python telegram_commands.py                       # one inbound-command polling pass
```

```python
import projections as P
proj = P.project(season=2026, week=1)  # seasons defaults to [2025, 2026] automatically
```

**`schedule_captures.py` is now the single "run the week" entry point** for the
time-critical half of the pipeline — capture, prediction logging, and the report
push — intended to run every 15 minutes under cron on an always-on machine (see
"Automation" in the Roadmap below and `deploy/setup.sh`), rather than each piece
being run by hand at roughly the right time. Generating a projection or a report
directly, as above, still works exactly the same regardless of whether that's
deployed - the scheduler calls the same functions, it doesn't replace them.
**`RUN.md` is a current, accurate, plain-language guide** — written for a
non-technical reader, but kept in sync with the actual system rather than
describing the old pipeline.

## Architecture

```
nfl_source.py (nflverse)  ─┬─>  projections.py  ─┬─>  backtest.py (tuning + holdout gate)
                            │                      │
                            │                      ├─>  ledger.py (log_predictions)
                            │                      │            │
                            │                      └─>  market_odds.py (compute_edge)
                            │                                │  │
odds_capture.py (Kalshi)   ─┤                                │  └─>  report.py (_screen_bets)
dk_capture.py (DK closing) ─┤                                │            │
        │                   │                                │            ├─> telegram_notify.py
        │                   │                                │            └─> ledger.log_recommendations
        │                   │                                │                        │
        │                   │                                │      telegram_commands.py  (/bet <slot>)
        │                   │                                │                        │
        │                   │                                │            ledger.record_bet
        │                   │                                │                        │
        └─> data/odds_history/{kalshi,draftkings,predictions,recommendations,bets}.csv
                             │                                │        (all append-only)
                             └────────────────> settle.py <───┘  (error, result, CLV)
```

`projections.py` needs only `nfl_source.py`. `report.py` formats its output for
delivery; `telegram_notify.py` is a thin push layer under that. The two capture
scripts need only network access and write straight to history files — neither
depends on the projection engine. Two joins exist against captured odds now, both
via `market_odds.py`'s normalized shape: `settle.py` joins **after the fact**, to
grade a bet already placed (error, result, CLV — Phase 7); `report.py` joins
**before** a bet, via `market_odds.compute_edge()`, to show edge alongside
confidence for whichever shown players have a captured line (Phase 8). Most players
show no edge simply because most stat/player combinations have no line captured yet
— that's expected, not a bug; see "Confidence, not edge" below for what the
confidence filter alone still means where no line exists.

### `nfl_source.py` — the data layer

Single entry point for all NFL data, backed by nflverse. Replaced ~3,050 lines of
Selenium scraping. Functions: `weekly_stats()`, `schedule()`, `injuries()`, plus
`require_fresh()` / `clear_cache()`.

Three things worth knowing:

- **Freshness is verified, not assumed.** nflverse publishes `timestamp.json` per
  dataset; `require_fresh()` reads it and *raises* rather than projecting off stale
  numbers. Call it before any in-season run.
- **nflreadpy caches for 24h** (`cache_duration=86400`, memory mode). Fine midweek,
  wrong on Sunday morning when a day-old cache hides inactives. Call `clear_cache()`
  on the Sunday run.
- **The one non-trivial join is snap counts.** `player_stats` keys on gsis
  `player_id`; `snap_counts` keys on `pfr_player_id`. `load_players()` supplies the
  crosswalk, matching 99.9% of skill-position rows. If `offense_pct` ever comes back
  mostly null, that join is what broke — `tests/test_nfl_source.py` guards it.

`schedule()` carries `spread_line` and `total_line` — the market's historical closing
numbers, free, with 100% coverage. That is the fair-value benchmark for the team model
and required no odds API.

**Feeds publish on different timelines.** Schedules and rosters exist months before a
season starts; `weekly_stats()` and `injuries()` 404 until the first games are played.
`published_seasons()` filters unstarted seasons out (printing which it skipped, and
raising if *nothing* requested is available) so projecting an upcoming week works
instead of crashing — this is what makes `project(2026, 1)` possible in August.

**`TEAM_ALIASES` exists because nflverse is not perfectly self-consistent.** The 2026
roster file abbreviates Arizona `AZ` while every schedule (and the 2023-25 rosters)
use `ARI`. Unnormalized, Arizona silently vanished from the roster→schedule join —
an entire team with zero projections and no error, the same failure shape as the old
57-player-pool bug. Normalized in `rosters()`, with
`test_roster_team_abbreviations_match_the_schedule` to catch new drift and a hard
raise in `_placeholder_rows` if any scheduled team ends up with no players.

Polars is an implementation detail: nflreadpy returns polars, everything converts to
pandas at this boundary (hence `pyarrow`). The rest of the project is pandas-only.

### `projections.py` — the model

Replaces the legacy engine. Usage × efficiency with reliability-weighted shrinkage.
`build(stats, schedule)` returns per-player-week projections for the raw stats
themselves — `pass_yd`, `rush_yd`, `rec_yd`, `receptions`, `pass_td`, `rush_td`,
`rec_td`, `interceptions` — plus `gameday`/`gametime` and two confidence columns
(below). `project(season, week)` wraps it for a single upcoming week, sorted by
total expected touches (`e_touches`) **within position** — see `report.py` for why
that qualifier matters.

**History carries across the season boundary.** `_trailing()`, `_trailing_rate()`,
and `games_played` group by `player_id` alone, not `player_id` + `season`. Grouping
by season was the original design and it silently broke weeks 1-2 of *every*
season: with no same-season history yet, `MIN_GAMES` filtered every player out and
`build()` returned zero rows. Confirmed and fixed once real 2025 Week 1 data was
checked. `project()` now defaults `seasons` to `[season - 1, season]`, so calling it
plainly (`P.project(2026, 1)`) is safe. **`build()` has no such default** — it takes
`stats_df`/`schedule_df` directly, so a caller who loads only the target season (e.g.
`src.weekly_stats([2026])`) will reproduce the empty-weeks-1-2 bug even though the
grouping fix is in place, because there is no prior-season data in the frame to
carry forward. The tradeoff is real, not free either way: a player who changed teams
or role over the offseason still gets his prior team's trailing average, undiscounted
for the change. `tests/test_projections.py::test_early_season_weeks_use_prior_season_history`
guards the row-count side of this; there is no test for the role-change blind spot,
because there is no fix for it yet either.

**Confidence, not edge.** `confidence_yardage`/`confidence_touchdown` (plus
`_label` columns: high/medium/low) tell you how much to trust the *number itself* —
they are `MEASURED_RELIABILITY` ceiling × `min(1, games_played / CONFIDENCE_FULL_SAMPLE_GAMES)`.
Since the TD/INT reliability ceiling is 0.09, **touchdown confidence can never
reach "high"** — that is correct, not a bug, per the reliability table below.
Confidence alone still says nothing about whether a number beats a market price —
that comparison is `market_odds.compute_edge()` (Phase 8), reported in `report.py`
*alongside* confidence, never merged into one number, because they answer different
questions: confidence can be high with no edge (an accurately-priced player), and
edge can be nonzero with low confidence (a noisy number that happens to look
off-market — a weaker signal, not a stronger one). Edge only ever appears where a
market line has actually been captured for that player/stat, which today is most of
the time *not* the case (see "Which venue to bet" below) — confidence remains the
only triage available for everyone else.

**Projecting a week that has not been played.** `build()` is retrospective — it emits
one projection per player-week *already present in the stats data*, so on its own it
returns nothing for an upcoming week. `project()` handles this by appending
placeholder rows built from `src.rosters()` + the schedule (`_placeholder_rows`),
with raw stats left NaN so a not-yet-played game is never mistaken for a zero. Before
this existed, `P.project(2026, 1)` — the actual production call — raised a raw 404 and
then returned zero rows. Guarded by `test_upcoming_week_is_projectable`.

**No-history coverage.** Anyone below `MIN_GAMES` of prior-game history — every
rookie among them — no longer vanishes silently. `project()` rescues these rows with
a depth-chart-based volume prior (`_apply_rookie_prior`, `DEPTH_RANK_VOLUME_PRIOR`)
where a current depth-chart entry exists, always at low confidence; a player with
neither history nor a depth-chart entry is still dropped. See Phase 6b in the
Roadmap for the measurement behind it and the schema landmine it navigates.

**`kickoff_windows(season, week)`** enumerates the distinct game days for a week and
a suggested capture time (3h before the earliest kickoff in each) — a real 2026 Week
1 has games on Wednesday, Thursday, Sunday, *and* Monday, so "capture Wednesday and
Sunday" (the original plan) would miss the Wednesday/Thursday closing line entirely
and try to act on Sunday/Monday days too early. See "Capture cadence" below.

**There is no composite fantasy-point score.** A DraftKings-scoring layer
(`dk_points`/`expected_dk_points`, plus the gamma-distribution machinery needed to
integrate DK's nonlinear yardage bonuses) existed through Phase 4 and was removed
deliberately — the project's goal is the individual stat predictions themselves, not
a fantasy score. **`exceed_probability()` (Phase 8) reused the old gamma/dispersion
approach but not its numbers** — it was rebuilt as its own concern, recalibrated
across the full range of thresholds rather than just DK's three bonus levels, and
restricted to players with real volume in that stat (see `DISPERSION_VOLUME_FLOOR` —
an unthrown QB's ~0 rec_yd projection is real but irrelevant noise for calibrating a
threshold nobody would query for that player). See Phase 8 in the Roadmap for the
measured constants and known calibration caveats.

Every parameter is measured, not chosen. Split-half reliability (2023-24) is the
design driver:

| | reliability | shrinkage applied |
|---|---|---|
| usage (targets, carries, snaps) | 0.94–0.98 | trust ~95% |
| efficiency (yds/target, yds/carry) | 0.33–0.49 | trust ~40% |
| TD rate | **0.09** | trust ~9% |

Kelley's formula: the weight on a player's own observed value *is* its reliability.
The old engine shrank TDs 15% and yards 8% — backwards, since TD rate is the noisiest
signal.

Deliberately excluded, each because measurement rejected it:

- **Opponent / defense-vs-position adjustment.** r = +0.041 against the residual;
  applying it multiplicatively made MAE **1.59% worse**.
- **Aggressive time decay.** The curve is nearly flat. The old 0.7 was up to 2.2%
  worse than optimal; even no decay was within 1%. `DECAY = 0.9` is not a sensitive knob.
- **A team-volume model.** Vegas predicts team *points* (r=0.409, fully subsuming a
  team's own scoring history) but **not** team *volume* (r=0.081) — NFL play counts
  are near-constant across teams. So the implied total scales **TDs only**.

One thing to preserve when editing: **`LEAGUE_IMPLIED_TOTAL` is a frozen constant on
purpose.** Fitting it at run time would compute it over the whole frame including
future weeks, which leaks into backtests. `tests/test_projections.py::test_no_future_leakage`
catches this class of bug — it already caught one real instance during Phase 4, when
several `build()` priors were being computed over the whole frame instead of leak-free.

### Validation — `backtest.py`

`python backtest.py` prints a scorecard **per raw stat** (not a composite score) for
tuning and holdout, each against a last-4-week-average baseline. **Gate passed** — the
model beats the naive baseline on every stat, on the untouched holdout:

```
HOLDOUT 2025 (never used to fit anything)
  stat               n      MAE     bias  spearman   vs baseline
  pass_yd          529   63.554   -1.589     0.416    +3.04%
  pass_td          489    0.906   +0.068     0.332    +9.55%
  interceptions    428    0.654   +0.017     0.056    +7.17%
  rush_yd         2289   13.926   -1.037     0.775    +4.42%
  rush_td          869    0.462   -0.016     0.271   +16.73%
  rec_yd          3871   17.431   -0.367     0.595    +4.91%
  rec_td          1633    0.375   +0.004     0.264   +20.21%
  receptions      3923    1.318   +0.015     0.625    +2.95%
```

Tuning-set (2023-24) numbers are close to these — e.g. `rec_yd` MAE 17.897 vs the
holdout's 17.431 — which is the generalization check: a model fitted to 2023-24 that
did much better there than on 2025 would be overfit, and it isn't.

Things to carry forward:

- **Interceptions and TD counts are barely predictable at all** (spearman 0.05-0.33).
  Expected — TD/target and TD/carry measured 0.09 split-half reliability (see
  `projections.py` above). The model still beats the naive baseline on these, but by
  a smaller margin in absolute prediction quality than the improvement percentage
  alone suggests. Don't over-read a "+20%" on a stat the model is still bad at
  predicting in absolute terms.
- **QB stats (`pass_yd`, `pass_td`) are the weakest predicted category overall**,
  consistent with the volume-props edge thesis: passing efficiency and TDs are more
  noise-dominated than rushing/receiving volume.
- **The 2025 holdout has been observed more than once now** — first during the DK
  bonus-calibration work (since removed), again here after the stat-level rescoring.
  It is no longer a clean holdout in the strictest sense. Use 2026 in-season results
  as the next genuinely clean test rather than continuing to re-grade on 2025.

### `odds_capture.py` — Kalshi capture

Append-only snapshot of Kalshi NFL markets into `data/odds_history/kalshi.csv`.
Deliberately does no analysis: lines cannot be reconstructed after the fact, so the
only requirement is never losing data. See "Capture cadence" below for *when* — a
fixed twice-weekly schedule is wrong for a week with Wednesday/Thursday games.

Two API facts that will silently produce empty data if forgotten:

- **`/markets` returns `null` for every price field** (`yes_bid`, `yes_ask`, `volume`,
  `open_interest`). Prices exist **only** in `/markets/{ticker}/orderbook`.
- **The orderbook returns bids on both sides, never asks.** A NO bid at `p` is a YES
  ask at `1-p`. That is the only way to compute a spread.

Market data needs **no authentication**. `KALSHI_API` in `.env` is an API key *id*;
signing also requires the matching RSA private key, and that is only needed to trade.

Measured liquidity (2026-08-16, ~4 weeks before Week 1):

| series | median spread | median depth |
|---|---|---|
| `KXNFLGAME` | 0.01 | $17,125 |
| `KXNFLSPREAD` / `KXNFLTOTAL` | 0.05 | $90k-108k |
| `KXNFLSEASONPASSYDS` | 0.14 | $1,008 |
| `KXNFLSEASONRECTD` / `RSHTD` | 0.25-0.29 | $105-142 |

**Kalshi does run weekly, single-game player props** — `KXNFLRECYDS`, `KXNFLRSHYDS`,
`KXNFLPASSYDS`, confirmed via already-settled preseason markets (157/37/16 settled
markets respectively). Two related series, `KXNFLREC` (receptions) and
`KXNFLRSHATT` (rush attempts) — the two purest volume-prop series and the closest
match to the measured edge — exist but have never been observed with an actual
market. Unresolved until real regular-season weeks confirm or rule them out. These
weekly series sit at zero open markets between slates and reopen near each week's
games, which is exactly why they can look absent if checked at the wrong time — see
`SERIES` in `odds_capture.py` for the full list now being captured.

Season-prop spreads are bimodal, not uniformly wide: prominent player/threshold
combinations quote 1-2 cents while obscure ones sit at 25-30. Tight markets are also
the better-priced ones, so edge and executability trade off directly.

### `dk_capture.py` — DraftKings closing lines

Narrow purpose: record what DK closed at, so CLV is computable. Bets are placed by
hand (price recorded manually) and Kalshi supplies free intra-week movement, so this
only needs the closing sweep — see "Capture cadence" below for exactly when.

```bash
python dk_capture.py 6 --dry-run   # cost estimate, spends nothing
python dk_capture.py 6             # capture games kicking off within 6h
```

**Budget is the binding constraint.** Free tier is 500 credits/month; props cost
1 credit per market per event.

- `/events` returns the **entire remaining season** (272 games in August). Unfiltered
  at 3 markets that is **816 credits** — the whole month's budget, in one run.
  `within_hours` is mandatory, not a convenience.
- `/events` and empty responses cost **0 credits**, so polling before props post is free.
- `capture()` refuses rather than overrunning, keeping `CREDIT_RESERVE = 50` back.
  Guarded by `tests/test_capture.py`.

Sustainable cadence: 16 games x 3 markets x 1 sweep x 4.33 weeks = **208 credits/month**.

`.env` uses `ODDS_API` (the legacy `odds.py` expected `ODDS_API_KEY`).

### Capture cadence — driven by `kickoff_windows()`, not a fixed weekly schedule

The original plan called for capturing twice a week (Wednesday open, Sunday close).
**That is wrong**: real 2026 Week 1 has games on Wednesday, Thursday, Sunday, *and*
Monday. A Wednesday-open/Sunday-close schedule would never capture a real closing
line for the Wednesday or Thursday games (both already played out by the time
Sunday's sweep runs), and would try to act on Sunday/Monday games far too early.

Use `projections.kickoff_windows(season, week)` to get the real answer for any given
week — it returns one row per distinct game day with `suggest_capture_by` (3h before
that day's earliest kickoff, matching `dk_capture.py`'s `within_hours=6` default with
margin to actually place a bet):

```python
import projections as P
P.kickoff_windows(2026, 1)
#   game_date  games  earliest_kickoff  ...  suggest_capture_by
#   2026-09-09     1  2026-09-09 20:20  ...  2026-09-09 17:20   <- Wednesday opener
#   2026-09-10     1  2026-09-10 20:35  ...  2026-09-10 17:35   <- Thursday
#   2026-09-13    13  2026-09-13 13:00  ...  2026-09-13 10:00   <- Sunday slate
#   2026-09-14     1  2026-09-14 20:15  ...  2026-09-14 17:15   <- Monday
```

Run `odds_capture.py`/`dk_capture.py` once per row, near `suggest_capture_by`, not on
a fixed two-day-a-week schedule. `gametime` is assumed Eastern (early Sunday games
show `13:00`, the NFL's standard 1:00 PM ET slot) — unverified against an
authoritative source, but consistent across every game checked.

### `report.py` + `telegram_notify.py` + `telegram_commands.py` — delivery

`report.py` formats a **bet sheet**, not a projection listing: only wagers that clear
`_screen_bets()`. `telegram_notify.py` pushes it; `telegram_commands.py` receives
`/bet` back.

```bash
python -c "import report, telegram_notify as T; T.send_message(report.build_report(2026, 1, game_date='2026-09-10'))"
python telegram_commands.py    # one polling pass; cron runs this every 2 min
```

**The full projection frame is still recorded on every run** by
`ledger.log_predictions()` — every player, every stat, unfiltered — which is what
`settle.py`/`scorecard.py` grade against actuals. That record is the learning loop;
the message is the decision, and the two want opposite things. The previous format
listed every high-confidence player and buried three real bets among nineteen rows,
including a TE projected for 1.1 receiving yards.

**Every screen bound traces to a specific confidently-wrong output**, not to taste:

| bound | the failure it prevents |
|---|---|
| `NEAR_PROJECTION = (0.25, 0.75)` | A 219-yard QB projection showed **+5% edge at a 350-yard rung** — the gamma tail is too fat, and the same defect reads as **-17% at 150**. Edge far from the projection measures the dispersion fit, not the market. |
| `EDGE_AT_ASK = (0.05, 0.20)` | Lower: mid-based edge flatters every quote, since you never transact at the mid. **Upper is the important half** — a WR4 on a new team showed +37% at every rung while the market priced him near zero. Past ~20 points the market knows something (an inactive, a depth-chart drop) and the model is wrong. |
| `MAX_SPREAD = 0.10` / `MIN_DEPTH = 500` | A book quoting 0.01/0.72 has a "mid" of 0.365 that means nothing — and mid-based edge is **largest exactly where the book is emptiest**. |

`agreeing_rungs` falls out of the ladder for free: a real disagreement persists across
neighbouring thresholds, while a dispersion artifact flips sign as you walk up them.

**One bet per (player, stat), best-priced rung.** The old `_edge_lookup` sorted only
by venue and then `drop_duplicates`'d, so *which* rung got printed was incidental —
that is how a 219-yard projection got reported against a 350-yard line.

**DraftKings never clears the screen today.** The API publishes no resting size, so
`depth` is null and `MIN_DEPTH` excludes it. Intended rather than incidental — DK
holds ~4.5% against Kalshi's ~1%, and a quote whose book cannot be inspected has no
place on a sheet whose whole job is inspecting books. DK stays in `market_lines()`
for `settle.py`'s closing-line work.

**An empty slate says so out loud** ("No bets clear the screen, N priced lines
checked"). Silence is what a broken cron looks like, and the two must never be
confusable from a phone — a malformed `flock` line once no-op'd the scheduler for a
full kickoff window and looked identical to "nothing qualified".

**`telegram_commands.py` is polled from cron, not a daemon.** `getUpdates` is one
cheap call, cron already exists, and a long-running process is a third thing to
supervise for no gain. Its own lock file and a 2-minute cadence, deliberately not
sharing the capture's lock — recording a wager must never queue behind a 12-minute
Kalshi sweep. Two rules worth keeping: **only `TELEGRAM_CHAT_ID` is honoured** (a bot
username is discoverable, and `bets.csv` is append-only, so a stranger's message
would be a permanent silent row), and **there is no `/undo`** (a ledger that can
retract its own history cannot grade anything). The Telegram message's own timestamp
is recorded, not collection time.

`ledger.log_recommendations()` writes `recommendations.csv` — the numbered sheet as
sent. `slot` is what `/bet 2 25` resolves against, and logging every recommendation
whether or not it was backed is the only way to ever grade the screen itself:
`bets.csv` holds the handful actually placed, and a handful per season can never say
whether the +20 cap or the 25-75% window are the right numbers.

### Which venue to bet

The cost difference is decisive where liquidity allows:

| venue | hold | break-even win rate |
|---|---|---|
| DraftKings -110/-110 | ~4.5% | 52.4% |
| Kalshi (~1% fees) | ~1% | 50.5% |

That ~2-point gap is often larger than the entire model edge. Prefer Kalshi when a
market has liquidity; DK otherwise. Weekly Kalshi prop liquidity is still unknown —
those markets sit at zero between slates and reopen near gameday.

Kalshi and DraftKings both price props as threshold bets ("50+ receiving yards").
`projections.exceed_probability()` converts a point projection into `P(stat ≥
threshold)`, and `market_odds.py` turns both venues' captured lines into one
`(player_id, stat, line, implied_probability, venue, captured_at)` shape;
`market_odds.compute_edge()` joins the two. See Phase 8 in the Roadmap for how this
was measured and its scope.

## Why the engine was rebuilt (history)

The pre-rebuild `projection_engine/` and `team_projection_engine.py` (deleted; see
`git show 988d8f7^:projection_engine/core/base_projections.py` to recover) had two
defects that motivated `projections.py` from first principles rather than patching:

1. Its schedule-strength, opponent-matchup, and regression-to-mean steps wrote to
   `proj_*` columns that the projection frame never had, so the condition guarding
   every write was always false. Those steps ran and were silently discarded — what
   shipped was an unadjusted historical average.
2. A frozen, unmaintained snap-count CSV (no writer anywhere in the codebase) filtered
   the player pool down to 57 players with **zero wide receivers** for the weeks it was
   in effect. Root cause was the PFR scraper: `snap_pct` was null/zero in 56% of rows.
   nflverse measures 5.7% for the same skill positions.

Everything under **Direction** below is the design that replaced it, and every choice
there was measured, not assumed — see `projections.py`'s module docstring for the
numbers behind each one.

## Direction

- **Data source → nflverse (`nflreadpy`)**, replacing the PFR/ESPN Selenium scrapers.
  150 columns vs. 28 scraped, includes `target_share` and `air_yards_share`.
- **Model → usage × efficiency**, reliability-weighted shrinkage (see `projections.py`
  above). Averaging the *product* directly — what the old engine did — bakes efficiency
  noise into the forecast.
- **Output → raw stats, not a fantasy score.** A DraftKings-scoring layer existed
  through Phase 4 and was deliberately removed — the goal is the individual stat
  predictions (`pass_yd`, `rec_yd`, `receptions`, etc.), which is also what both
  fantasy roster decisions and prop bets actually key off. `backtest.py` scores each
  stat directly rather than through a composite point total.
- **Edge thesis:** volume props (receptions, carries, attempts) are usage-driven and
  predictable; yardage props are efficiency-driven and mostly noise. NFL sides/spreads
  are efficient — do not model them.
- **Venue: Kalshi where liquid, DraftKings otherwise** — see "Which venue to bet" above.
- **Steer on calibration and closing-line value, never on early ROI.** At ~50 bets, ROI
  is statistically indistinguishable from noise.
- **Delivery → Telegram push**, not a webapp. Built (`report.py` +
  `telegram_notify.py`), push-only by design — decisions happen at specific
  kickoff-adjacent moments (`kickoff_windows()`), so timing matters more than a
  queryable chat interface. Interactive polling could be added later behind the same
  bot token if that changes.

## Landmines

**nflverse is *largely* one team-abbreviation convention (32 teams; `LA` for the
Rams, `LAC` for the Chargers), but "largely" is doing real work in that sentence.**
The 2026 roster feed uses `AZ` where every schedule (and the 2023-25 rosters) uses
`ARI`, which silently dropped a whole team from the roster→schedule join until it was
caught — zero projections for an entire team, no error raised. Normalize through
`nfl_source.TEAM_ALIASES` rather than assuming cross-feed consistency, and never
assume a join "obviously" covers all 32 teams — assert it.

**`RUN.md` is a current, accurate, plain-language guide** — written for a
non-technical reader, kept in sync with the real system.

**Generated artifacts are gitignored except captured betting lines.**
`data/projections/`, `data/season_projections/`, `data/insights/`, `data/nuggets/`,
`data/fun_stats/`, `data/depth_charts/`, and root `nfl_week_*_analysis_*.html/csv` are
ignored — reproducible from source data, so not tracked. **`data/odds/` (legacy
`odds.py` output, weeks 1-10 already captured) and `data/odds_history/` (Kalshi/DK
capture) are both deliberately tracked** — captured betting lines cannot be
regenerated after the fact, and are the only source for closing-line-value analysis.

---

# Roadmap

Phases 0-5 are done (see "Why the engine was rebuilt" above). What follows is the
remaining work, ordered by **risk of being confidently wrong**, not by value.

**The hard deadline is Week 1 (2026-09-09).** Two things are true and shape everything
below:

- **Correctness gaps produce confidently wrong output.** A projection for a player
  who is inactive is worse than no projection — you can work around a missing number,
  you cannot work around a wrong one you trusted.
- **Some data cannot be backfilled.** Bets placed before a ledger exists are
  permanently unmeasurable, exactly like betting lines not captured before kickoff.

That gives a hard cutline: **Phases 6 and 7 must land before you bet.** Phase 8 (edge)
is the most *valuable* work but is not a blocker — you can compare a projection to a
line by eye in the meantime.

## Phase 6 — Correctness (done)

The system used to produce confident numbers for players who would not play, and no
numbers at all for ~45% of rostered players. All three fixes land in `project()`,
not `build()` — `build()` stays exactly as `backtest.py` validates it; these are
target-week-specific corrections layered on top, so the gate's methodology and
results are untouched by any of this.

**6a. Injury integration.** `project()` now excludes players ruled `Out`/`Doubtful`
and discounts `Questionable`, gated on the target season having actually started
(injury reports don't exist before then). The discount was measured, not guessed:
`build()`'s retrospective projection vs. actual production (0 for player-weeks with
no stats row at all, i.e. did not play) across 2023-25, restricted to skill
positions — unfiltered, the injury report is mostly linemen and defense who would
never appear in `weekly_stats` anyway, which understates any played-rate measured
against it. Result: `Out` (976 tagged) and `Doubtful` (134 tagged) both measured
under 1% played — indistinguishable, so both are excluded outright rather than given
a falsely-precise near-zero multiplier. `Questionable` (1205 tagged) measured 51.0%
played, and players who did play produced close to their normal projection — the
discount is essentially a play/no-play gate, not a diminished-performance one. Frozen
as `INJURY_DISCOUNT` in `projections.py`.

**6b. Rookie / no-history coverage.** No-history players (every rookie, among
others) are no longer silently dropped below `MIN_GAMES` — `project()` calls
`build(..., min_games=0)` and rescues them with `_apply_rookie_prior`, provided a
current depth-chart entry exists. Volume (targets/carries/attempts) is rescaled
toward `DEPTH_RANK_VOLUME_PRIOR`, a (position, depth-rank-tier) table measured on
2023-24 depth charts — the tiered pattern is a real, monotonic signal (e.g. TE
targets: 3.40 / 1.80 / 0.62 by tier) confirming depth-chart rank predicts volume,
not just proxies for it. These rows always land at **low** confidence purely from
`games_played` being under 2 — no separate pinning needed. A player with neither
trailing history nor a depth-chart entry is still dropped; there is no signal to
project them from.
  - **Landmine, not an oversight:** nflverse's depth-chart pipeline changed schema
    at the 2025 season boundary. 2024-and-earlier publish weekly, week-aligned
    snapshots (`season`/`week`/`depth_team`, where `depth_team` ranks players
    *within one formation slot* — three different WRs can each be "1", for the
    X/Z/Slot spots). 2025-onward publish a single rolling "current" snapshot instead
    (`dt`/`pos_rank`, a real overall rank, no week dimension). `nfl_source.
    depth_chart_ranks()` only supports the newer schema — the live prior only ever
    needs "who is starting right now," never a specific past week — and the 2023-24
    measurement first collapsed each player to his best slot rank, then dense-ranked
    within (season, week, team, position) to make the two schemas comparable.
  - Coverage measured on `project(2026, 1)`: 833 rows, up from 494 before this
    landed (roughly 400 more players, matching the originally measured gap).

**6c. `require_fresh()` wired in.** Called at the top of `project()`, but on a
narrower gate than 6a's: the target season must have both started *and* not yet be
complete (every game in `schedule_df` already carrying a final score means it's
complete — checked directly, not via a calendar date). A season that's fully over
can never go "more stale," since nothing will ever publish for it again — so a
retrospective query like `project(2025, 5)` run well after that season ended must
not fail just because nflverse's live feed (which only ever reflects whichever
season is *currently* in progress) happens to be quiet, which is most of the
offseason. This was originally gated on the same `season_started` flag as 6a's
injury filtering, which is correct for injuries (old reports don't go stale) but
was wrong for freshness — caught only once real droplet test runs during the 2026
offseason hit it directly, not by the unit tests, which had no way to exercise a
"started but not yet complete" season using only 2023-25 fixture data (all three are
long finished) until one was constructed synthetically to test it.

## Phase 7 — Measurement (done)

Nothing recorded what was predicted or bet, so nothing could be evaluated later.
Built as two new modules, both writing under `data/odds_history/` alongside the
captured lines (same reason: not regenerable).

- **`MODEL_VERSION`** in `projections.py` (currently `"1.0"`), bumped on any change
  to `build()`'s model logic so a change in results can be attributed to the change
  rather than confused with variance.
- **`ledger.py`** — `log_predictions(proj_df)` appends one row per (player, stat) in
  a `project()` result to `predictions.csv`
  (`ts_utc, model_version, season, week, player_id, stat, projection, confidence`),
  timestamped at call time — call it right before generating/sending a report, not
  on a schedule, so the timestamp reflects when the projection was actually acted on
  and pre-kickoff timestamps stay structurally impossible to edit retroactively.
  `record_bet(...)` appends one manually-entered row to `bets.csv`
  (`ts_utc, season, week, player_id, stat, venue, side, line, price, stake`) with a
  light validity check on `venue`/`side` — manual entry is still the intended path,
  this just guards against a typo silently corrupting an append-only, non-regenerable
  file.
- **`settle.py`** — `settle(season, week)` joins `bets.csv` to the latest matching
  `predictions.csv` row and to real actuals (`nfl_source.weekly_stats`) for `error`
  and win/push/loss `result`. **CLV is DraftKings-only.** `dk_capture.py`'s captured
  rows are structured (`player`, `market`, `line`, `side`), so a bet matches its
  closing line unambiguously on those four fields. Kalshi's captured rows have no
  such field — `title` is free text ("Will Justin Jefferson have 75+ receiving
  yards?") — and guessing a player/threshold out of it risks silently matching the
  wrong market's price, which is worse than reporting nothing; Kalshi bets still get
  `error`/`result`, just no `closing_price`. A further, honest gap: `dk_capture.py`
  only sweeps `player_receptions`/`player_rush_attempts`/`player_pass_attempts`
  (see its `MARKETS`), so even DraftKings CLV is only computable for `receptions`
  bets today — the other five tracked stats have no closing line captured to match
  against yet.

## Phase 8 — Edge (done)

Used to be the largest remaining piece and the only one that answers "is this bet
good?" The system used to say *"Puka Nacua, 90.7 rec yards, high confidence"* and be
unable to say whether that beats a posted line - it can now, wherever a line has
actually been captured.

**8a. Threshold probabilities.** `projections.exceed_probability(proj, stat,
threshold)` - a gamma distribution (method of moments, `cv = k/sqrt(mean)`), same
shape as the old (removed) `_exceed_probability`/`STAT_DISPERSION`, but **not pasted
back unchanged**: refit on 2023-24 across the full range of projections (not just
DK's three old bonus thresholds), restricted per stat to players with real volume in
it (`DISPERSION_VOLUME_FLOOR` - an unthrown QB's ~0 rec_yd projection is real but
irrelevant noise for calibrating a threshold nobody would query for that player),
and built as its own concern rather than welded to a scoring system. Only
`pass_yd`/`rush_yd`/`rec_yd`/`receptions` are supported - the stats with both a real
point projection and an actual captured market; touchdowns/interceptions are not
extended to, since their 0.09 reliability ceiling would make a "probability" false
precision on top of false precision. Checked (not fit) on 2025: predicted clear-rate
runs a few points low in several bins, inherited from the base model's own small
point-estimate bias rather than a defect in the dispersion fit - 2026 in-season is
the next genuinely clean check.

**8b. Odds normalization — `market_odds.py`.** Both venues into one shape:
`(player_id, stat, line, implied_probability, venue, captured_at)`.
- **Landmine that turned out not to be one:** Kalshi's weekly player-prop titles
  looked like free text going into this (that's why Phase 7's CLV matching skips
  Kalshi entirely - see there), but checked against real settled preseason markets,
  they're a strict template - `"{Player Name}: {N}+ {description}"` - not the
  freeform sentences game markets use (`"Will Kansas City win..."`). Parsed directly;
  no fuzzy matching needed. Player name still resolves to `player_id` via an exact
  match against `nfl_source.rosters()` - unmatched names are dropped and counted,
  not guessed.
- DK: two-sided American odds → implied probabilities → **de-vigged** using the
  paired Over/Under price for the same (event, market, line, capture) → fair
  probability, keeping only the Over side (matches Kalshi's "X+" framing). Still
  `receptions`-only, same gap noted in Phase 7 - `dk_capture.py` only sweeps
  `player_receptions`/`player_rush_attempts`/`player_pass_attempts`, and only the
  first has a corresponding build() stat.
- Kalshi's mid-price is used directly as the implied probability (~1% fee, far less
  work than DK's de-vig) - and is the better fair-value reference for that reason.

**8c. Edge, reported alongside confidence.** `market_odds.compute_edge()` = model
probability (8a) minus market implied probability (8b), for every captured line that
matches a projected player/stat. `report.py` shows it inline next to the stat it
prices, preferring Kalshi over DraftKings when both have a line (see "Which venue to
bet" below), and only when one exists - most players show no edge simply because
most player/stat combinations have no line captured yet, which is the common case
today, not a bug. Never collapsed into confidence: a high-confidence number can show
no edge (accurately priced), and a nonzero edge on a low-confidence number is a
weaker signal, not a stronger one - the disclaimer in every report says this
explicitly. (CLV - grading a bet already placed against a closing line - is the
separate, already-built piece in `settle.py`/Phase 7; this is the pre-bet version of
a similar comparison.)

**Open dependency, still unresolved:** `KXNFLREC` (receptions) and `KXNFLRSHATT`
(rush attempts) - the two purest volume-prop series, and the closest match to the
measured edge - exist on Kalshi but have never been observed with an actual live
market (re-confirmed 2026-08-16: 0 open, 0 settled for `KXNFLREC`). If they never
open, the strongest edge is DK-only, at DK's worse pricing. Week 1-3 capture answers
this; `market_odds.py` does not design around an assumption either way - it simply
finds nothing to normalize for a series with no rows, the same as it does today.

## Phase 9 — Automation (done)

Four manual commands at four different times per week was a plan that would not
survive contact with a real season - a laptop asleep at the wrong moment means a
permanently missing capture. Deployment target: a small always-on Ubuntu droplet,
not a local machine (see `deploy/setup.sh` and RUN.md's "Running this without you
having to remember any of it").

- **`schedule_captures.py`** — polls (intended: cron, every 15 min) and no-ops
  unless *now* falls inside one of `kickoff_windows()`'s actual windows for the
  current week, **not** a fixed weekly cron — a real week spans Wed/Thu/Sun/Mon and
  a fixed schedule misses closing lines entirely. `current_week()` finds that week
  from the real schedule (season = the NFL's own year label, a January game
  belongs to the *previous* year; "current" = earliest week whose games haven't
  all finished, 1-day grace past the latest kickoff). Idempotency is a marker file
  per (season, week, game_date) under `data/scheduler_state/` (gitignored -
  operational bookkeeping, not data) rather than a database, since cron re-invokes
  the script stateless every time. Each action (Kalshi capture, DK capture, log
  predictions, Telegram push) is isolated - one failing (e.g. DK's credit budget)
  does not block the others, and the window is still marked done rather than
  retried forever against a persistent error.
  - **Landmine worth flagging:** `kickoff_windows()`'s timestamps are tz-naive and
    *assumed* Eastern (see `nfl_source.py`), while `datetime.now(UTC)` is tz-aware -
    comparing them directly either raises or, worse, silently compares the wrong
    epoch if tzinfo is just stripped instead of actually converted. Fixed via
    `zoneinfo.ZoneInfo("America/New_York")` (stdlib, correct across the EDT/EST
    transition that falls inside every NFL season) - caught by an end-to-end smoke
    test against real dates, not by the unit tests, which had accidentally
    constructed already-consistent fixtures.
- **Weekly scorecard — `scorecard.py`.** Bias and rank correlation (reusing
  `backtest._score`) on the just-completed week's logged predictions vs. real
  outcomes; CLV-to-date and calibration (predicted vs. actual clear-rate, binned)
  aggregated across every settled bet this season via `settle.py`. Calibration
  recomputes `exceed_probability` from each bet's logged projection with
  `skip_volume_check=True` - settle.py's thin per-bet records don't carry the
  volume column that check needs, and a bet is itself proof the player had real
  volume in that stat, so the check is redundant there, not skipped for
  convenience. `most_recently_completed_week()` (deliberately separate from
  `schedule_captures.current_week()`, which answers a different question with a
  grace period tuned for that) picks the target week for the no-argument cron
  invocation (`scorecard.py --push`).
- **`deploy/setup.sh`** — one-time droplet setup: installs Python 3.14 via `uv`
  (Ubuntu 24.04's own `python3` is 3.12, and PPA availability for a specific recent
  version isn't something to assume), sets the system timezone to
  `America/New_York` (so cron's own wall clock lines up with `kickoff_windows()`
  without a per-line `TZ=` hack), and installs the two crontab lines. Does not
  create `.env` - secrets are copied over by hand, never generated or fetched by a
  script.

## Phase 10 — Narrow to what actually works (gate, ongoing)

Not a build phase. Once several weeks of ledger data exist:

- Confirm empirically whether **volume props beat yardage props**, as the reliability
  work predicts. Keep what clears the bar; drop what doesn't.
- **Steer on calibration and CLV, never on early ROI.** At ~50 bets the two are
  statistically indistinguishable, and steering on ROI means fitting noise while it
  feels like progress.

**Legacy agent stack retired.** `odds.py`, `picks_agent.py`, `stats_agent.py`,
`utils/insights_formatter.py`, and their dedicated inputs (`data/roster.xlsx`,
`data/team_map.xlsx`, `data/player_name_mapping.csv`,
`data/nfl-2025-EasternStandardTime.csv`) were removed rather than rebuilt or reduced
to a narrow LLM lookup — nothing else in the codebase imported them or their data
files (`projections.py` was already independent of this stack; see `nfl_source.py`
above). `openpyxl` dropped from `requirements.txt` as its last consumer. Recoverable
from history if ever needed: `git show 63c46ec:picks_agent.py`.

## Known limitations that no phase currently fixes

State these plainly rather than letting them be rediscovered as bugs:

- **Early-season bias.** Weeks 1-2 beat baseline on 7 of 8 stats but run high, and
  **interceptions actually lose to the naive baseline** in that window. Confidence
  labels are more optimistic than they should be in weeks 1-2.
- **Role/team-change blind spot.** Cross-season carry-forward gives a player his prior
  team's trailing average with no discount for a changed team, scheme, or role — worst
  exactly at Week 1.
- **The 2025 holdout is spent.** It has been observed multiple times. **2026 in-season
  results are the next genuinely clean test** — do not re-grade on 2025 and treat it
  as independent evidence.
- **QB stats are the weakest category** (holdout spearman 0.48 pass_yd vs 0.76 rush_yd),
  concentrated in the noisiest components. Lean on QB projections least.
