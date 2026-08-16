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
GROK_API_KEY=...  # xAI, used by picks_agent.py / stats_agent.py (legacy, see below)
```

`xai-sdk` (required by `picks_agent.py` / `stats_agent.py`) is **not** in
`requirements.txt` and not installed in the venv. Those two files predate the rebuild
and are not wired to `projections.py`'s output — see "Legacy agents" below before
touching them.

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
python odds_capture.py                            # Kalshi snapshot, append-only, free
python dk_capture.py 6 --dry-run                  # DK cost estimate, spends nothing
python dk_capture.py 6                            # DK closing lines, games within 6h
```

```python
import projections as P, nfl_source as src
stats = src.weekly_stats([2025, 2026])
sched = src.schedule([2025, 2026])
proj  = P.project(season=2026, week=1, seasons=[2025, 2026])
```

There is no single "run the week" entry point yet — `projections.py` and the capture
scripts are run independently. `RUN.md` describes the pre-rebuild pipeline and is
retained only as a historical record; do not follow it.

## Architecture

```
nfl_source.py (nflverse)  ─┬─>  projections.py  ─>  backtest.py  (tuning + holdout gate)
                            │
odds_capture.py (Kalshi)   ┤
dk_capture.py (DK closing) ┘
        │
        └─> data/odds_history/{kalshi,draftkings}.csv  (append-only, never regenerated)
```

Three independent pieces, not a pipeline: `projections.py` needs only `nfl_source.py`.
The two capture scripts need only network access and write straight to history files —
neither depends on the projection engine, and nothing currently joins projections
against captured odds. That join (compute model edge vs. market price) doesn't exist
yet; it's the natural next module once Kalshi's weekly prop liquidity is known.

`odds.py`, `picks_agent.py`, `stats_agent.py`, `utils/insights_formatter.py` are a
**separate, older stack** — see "Legacy agents" below. They still run, but they consume
`data/roster.xlsx` / `data/team_map.xlsx` / `data/player_name_mapping.csv`, not
anything `projections.py` produces.

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

Polars is an implementation detail: nflreadpy returns polars, everything converts to
pandas at this boundary (hence `pyarrow`). The rest of the project is pandas-only.

### `projections.py` — the model

Replaces the legacy engine. Usage × efficiency with reliability-weighted shrinkage.
`build(stats, schedule)` returns per-player-week projections; `project(season, week)`
wraps it for a single upcoming week.

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

Two things to preserve when editing:

- **DK bonuses need a distribution, not a threshold test.** `E[points] ≠ f(E[yards])`
  because of the +3 at 100/100/300. `_exceed_probability` fits a gamma by method of
  moments; a player projected for 85 rush yards still earns the bonus sometimes, and
  ignoring that biases every projection low.
- **`STAT_CV` and `LEAGUE_IMPLIED_TOTAL` are frozen constants on purpose.** Fitting
  them at run time computes them over the whole frame including future weeks, which
  leaks into backtests. `tests/test_projections.py::test_no_future_leakage` catches
  this — it already caught it once.

### Validation — `backtest.py`

`python backtest.py` prints the scorecard for tuning and holdout. **Gate passed:**

```
                         MAE    bias   spearman   vs last-4
TUNING  2023-24        3.856  -0.332      0.786      +7.19%
HOLDOUT 2025           3.842  -0.090      0.782      +6.83%
```

The tuning/holdout gap is **0.36pp** — the model generalizes, it is not fitted to
2023-24. Beats the baseline at every position: QB +4.6%, RB +5.9%, WR +6.7%, TE +11.5%.

Two things to carry forward:

- **QB is the weak position.** Holdout Spearman 0.372 vs 0.78-0.79 for RB/WR/TE.
  QB scoring is concentrated in TDs and rushing, both noisy. Lean on QB projections
  least, especially for props.
- **The 2025 holdout has now been observed.** Its miscalibration prompted the
  dispersion fix (the fix itself was fitted on 2023-24 only, so the number is still
  meaningful). It is no longer virgin. Use 2026 in-season results as the next clean
  test rather than re-grading on 2025.

### Legacy agents — `odds.py`, `picks_agent.py`, `stats_agent.py`, `utils/insights_formatter.py`

Pre-rebuild code, kept because nothing has replaced its functionality yet — not
because it's been reviewed or endorsed. Treat it as a separate, older system that
happens to live in the same repo:

- **Not wired to `projections.py`.** `picks_agent.py` reads
  `data/projections/nfl25_proj_week{N}.csv`, a file format the current engine does not
  write. Running it against the new pipeline's output will not work.
- **Depends on files the rebuild left in place on purpose**: `data/roster.xlsx`,
  `data/team_map.xlsx`, `data/player_name_mapping.csv`,
  `data/nfl-2025-EasternStandardTime.csv`. These carry the three-abbreviation problem
  described under Landmines below — `nfl_source.py` does not have this problem, these
  files do.
- **Needs `xai-sdk`**, not installed (see Environment above).
- `picks_agent` calls `grok-4` with live search plus `grok-3`; `stats_agent` calls
  `grok-3`. Both write JSON that `insights_formatter.py` renders to CSV/HTML.

Whether to rebuild this layer, retire it, or replace the LLM step with a narrow
injury/news lookup feeding `projections.py` is an open decision, not yet made — do
not assume any direction here without asking.

### `utils/`

Only `insights_formatter.py` remains. Eight experimental modules and
`season_projector.py` were removed in the rebuild (zero external imports each);
recover any from history if needed: `git show c71c61c:utils/bayesian_updater.py`.
[docs/ML.md](docs/ML.md) describes their intended role but was always aspirational.

### `odds_capture.py` — Kalshi capture

Append-only snapshot of Kalshi NFL markets into `data/odds_history/kalshi.csv`.
Deliberately does no analysis: lines cannot be reconstructed after the fact, so the
only requirement is never losing data. Run twice weekly (open + pre-kickoff).

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

**Kalshi has no weekly per-game player props** — only season-long totals and leader
markets. The measured edge (weekly volume props: receptions, carries, attempts) is
therefore *not executable on Kalshi* and needs a sportsbook feed.

Season-prop spreads are bimodal, not uniformly wide: prominent player/threshold
combinations quote 1-2 cents while obscure ones sit at 25-30. Tight markets are also
the better-priced ones, so edge and executability trade off directly.

### `dk_capture.py` — DraftKings closing lines

Narrow purpose: record what DK closed at, so CLV is computable. Bets are placed by
hand (price recorded manually) and Kalshi supplies free intra-week movement, so this
only needs the closing sweep.

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

### Which venue to bet

Model calibration was verified at Kalshi's exact threshold ladder on the 2025 holdout
— errors within ±2 points at every level (25/50/75/100 yards; 150-300 passing). The
model is applicable to Kalshi, and its native `_exceed_probability` output matches
Kalshi's threshold-binary format with no translation.

The cost difference is decisive where liquidity allows:

| venue | hold | break-even win rate |
|---|---|---|
| DraftKings -110/-110 | ~4.5% | 52.4% |
| Kalshi (~1% fees) | ~1% | 50.5% |

That ~2-point gap is often larger than the entire model edge. Prefer Kalshi when a
market has liquidity; DK otherwise. Weekly Kalshi prop liquidity is still unknown —
those markets sit at zero between slates and reopen near gameday.

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
- **Scoring → DraftKings classic** (full PPR, +3 bonuses at 100 rush / 100 rec / 300 pass
  yards). Bonuses are nonlinear, so expected fantasy points needs `P(yards ≥ threshold)`,
  not a threshold test on the mean — hence `_exceed_probability`.
- **Edge thesis:** volume props (receptions, carries, attempts) are usage-driven and
  predictable; yardage props are efficiency-driven and mostly noise. NFL sides/spreads
  are efficient — do not model them.
- **Venue: Kalshi where liquid, DraftKings otherwise** — see "Which venue to bet" above.
- **Steer on calibration and closing-line value, never on early ROI.** At ~50 bets, ROI
  is statistically indistinguishable from noise.
- **Delivery → Telegram push**, not a webapp. Not yet built.

## Landmines

**Three incompatible team-abbreviation conventions coexist, but only in the legacy
agent stack.** `data/team_map.xlsx` is the bridge (`full_team_name` / `team_abbrev` /
`roster_abbrev`) — needed only by `odds.py` / `picks_agent.py` / `stats_agent.py`.

| Source | Convention | Green Bay / Kansas City / New Orleans |
|---|---|---|
| `data/nfl-2025-EasternStandardTime.csv` schedule | PFR (`team_abbrev`) | `GNB` / `KAN` / `NOR` |
| `data/roster.xlsx` `team` col | ESPN (`roster_abbrev`) | `GB` / `KC` / `NO` |

`load_team_name_mapping()` in [picks_agent.py:87](picks_agent.py#L87) is the reference
implementation. **`nfl_source.py` does not have this problem** — nflverse uses one
convention throughout (32 teams; `LA` for the Rams, `LAC` for the Chargers) and joins
on stable `player_id`, so `projections.py` and everything downstream of it needs no
team-name mapping at all. If you're writing new code and reaching for
`team_map.xlsx`, that's a sign you're solving an already-solved problem.

**`RUN.md` describes the pre-rebuild pipeline and should not be followed.** It refers
to scripts (`nfl_data.py`, `run_projections.py`, `run_season_projections.py`,
`backtesting/backtester.py`) that no longer exist. Kept only as a historical record of
what the system used to do; see "Running things" above for current commands.

**Generated artifacts are gitignored except captured betting lines.**
`data/projections/`, `data/season_projections/`, `data/insights/`, `data/nuggets/`,
`data/fun_stats/`, `data/depth_charts/`, and root `nfl_week_*_analysis_*.html/csv` are
ignored — reproducible from source data, so not tracked. **`data/odds/` (legacy
`odds.py` output, weeks 1-10 already captured) and `data/odds_history/` (Kalshi/DK
capture) are both deliberately tracked** — captured betting lines cannot be
regenerated after the fact, and are the only source for closing-line-value analysis.
