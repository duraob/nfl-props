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
ODDS_API_KEY=...   # The Odds API — free tier: 3 req/min, 500 req/month
GROK_API_KEY=...   # xAI; being demoted, see "Direction" below
```

### Tests

`tests/test_data_source.py` guards the nflverse schema contract (target_share, snap
coverage, star players present). It should always pass; a failure means upstream
nflverse changed and the model's inputs moved underneath it.

`tests/test_pool_health.py` **is expected to fail** until the engine rebuild lands. It
encodes the bug described below and is the definition of done for that work.

## Weekly pipeline

All commands run **from the repo root** (every path in the codebase is CWD-relative). `RUN.md` has the full narrative; the canonical order is:

Data is no longer scraped. `nfl_source.py` fetches it from nflverse on demand; there
is no scrape step to run first.

```bash
python run_projections.py <week>           # player projections - LEGACY, see Known-broken
python run_season_projections.py           # team projections + standings (no week arg)
python odds.py <week>                      # The Odds API -> data/odds/week_NN/
python picks_agent.py <week> [--live-search]   # projections vs odds + Grok analysis
python stats_agent.py <week>               # statistical nuggets + Grok insights
python utils/insights_formatter.py <week> --csv --html
```

Backtesting (run from repo root, not from `backtesting/`):

```bash
python backtesting/backtester.py <test_season> <ref_season> <start_week> <end_week> [decay] [scenario]
python backtesting/backtester.py 2024 2023 1 17 0.7 historical
```

`scenario` is `historical` (no roster filter, min 2 games / 10% snaps) or `current` (roster filter, min 3 games / 20% snaps) — see [backtesting/backtest_config.py](backtesting/backtest_config.py). Note the default report path is `../backtest_results/`, which lands **outside** the repo; pass an explicit `output_file` to `generate_csv_report()` if that matters.

## Architecture

The system is a linear file-passing pipeline — modules communicate only through CSV/JSON/XLSX under `data/`, never through imports. Changing an output schema breaks every downstream stage silently.

```
source                   engines                          agents
──────                   ───────                          ──────
nfl_source.py ─┬─> run_projections.py             ─> data/projections/
 (nflverse)    │     (projection_engine/)              nfl25_proj_week{N}.csv
               │      LEGACY - being replaced          monte_carlo_week{N}.json
               │
               └─> run_season_projections.py      ─> data/season_projections/
                     (team_projection_engine.py)
                      LEGACY - being replaced
                                                           ↓
odds.py ─────────────────────────> picks_agent.py / stats_agent.py ─> data/insights/
 (The Odds API)                                            ↓           data/nuggets/
                                   utils/insights_formatter.py         data/fun_stats/
```

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

### The 10-game rolling window

Both engines share one non-obvious data-selection rule, duplicated in [projection_engine/core/data_loader.py:157](projection_engine/core/data_loader.py#L157) and [team_projection_engine.py:33](team_projection_engine.py#L33). **Change one, change the other.**

- Week 1: 10 games from 2024 (`target_weeks_2024`, default `[9..17]`)
- Week 2: 1 game from 2025 + 9 from 2024
- Week 3+: all completed 2025 weeks, backfilled from 2024 to reach 10

Weights are exponential decay across the combined ordering: most recent 2025 game = 1.0, each older game × `decay_coefficient` (default 0.7), with the 2024 block continuing the same decay chain. Every downstream aggregation is a `np.average(..., weights=df['time_weight'])`.

### `projection_engine/` (player-level)

`ProjectionEngine.run_projections()` in [projection_engine/projection_engine.py](projection_engine/projection_engine.py) is a 9-step orchestrator over single-responsibility components:

`DataLoader` (roster + injury + snap-count filtering) → team/player dataset construction (inside the orchestrator, not a component) → `ScheduleAnalyzer` (league-normalized schedule strength) → `OpponentAnalyzer` (defensive matchup adjustments) → `BaseProjections` → `VarianceAnalyzer` → `SimulationEngine` (10,000 Monte Carlo runs, `random_seed=42`) → export.

Defensive team stats are *derived*, not scraped: `_build_team_statistics()` groups game data by `opponent` and relabels the opponent's offensive columns as `def_*`.

Player filtering is three-stage and cumulative — roster `is_injured` flag, then `data/injuries.csv` status in `{Out, Injured Reserve}`, then `data/snap_filtering_report.csv` at a 20% recent-snap threshold. A player silently absent from projections is usually stage three.

### `team_projection_engine.py` (team-level)

A parallel, class-based reimplementation (`TeamDataLoader`, `TeamScheduleAnalyzer`, `TeamVarianceAnalyzer`, `TeamProbabilityEngine`, `TeamSimulationEngine`, `TeamOpponentAnalyzer`, `TeamProjectionEngine`) that mirrors the player engine's methodology for win/loss, standings, and playoff probabilities. It shares no code with `projection_engine/` — improvements to one must be ported by hand.

### Agents

`picks_agent.py` and `stats_agent.py` both join projections against odds, build a prompt, and call Grok via `xai_sdk`. `picks_agent` uses `grok-4` with `SearchParameters` live search for injury/news context and `grok-3` for the main analysis; `stats_agent` uses `grok-3`. Both write JSON that `utils/insights_formatter.py` renders.

### `utils/`

Only `insights_formatter.py` remains, invoked as a CLI script. Eight unreferenced
experimental modules (`ml_models`, `bayesian_updater`, `advanced_time_weighting`,
`enhanced_projections`, `historical_analyzer`, `injury_integration`,
`monte_carlo_processor`, `variance_tracker`) and `season_projector.py` were deleted —
all had zero external imports. Recover any of them from history if an idea is needed:

```bash
git show c71c61c:utils/bayesian_updater.py
```

[docs/ML.md](docs/ML.md) describes their intended role but is aspirational, not a
description of shipped behavior.

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

## Known-broken — do not trust current projection output

Two verified defects make everything in `data/projections/` unusable. Both are fixed by
the engine rebuild, not by patching.

**1. The engine never applies its own adjustments.** [base_projections.py:68](projection_engine/core/base_projections.py#L68)
writes results only `if f'proj_{stat}' in base_projections.columns`, but that frame is a
copy of `df_players`, whose columns are `pass_yd`, `rec_yd`, … — never `proj_pass_yd`. The
condition is always false, so schedule-strength, opponent-matchup, and regression-to-mean
(steps 3-5 of the 9-step orchestrator) are computed and discarded. No shipped CSV contains
any `proj_*` column. **What ships is the time-weighted historical average and nothing
else**, with Monte Carlo simulating around an unadjusted mean.

**2. A stale filter deletes 80% of the player pool.** `data/snap_filtering_report.csv` has
**no writer anywhere in the codebase** — it is frozen from commit `b36740c`.
[data_loader.py:254](projection_engine/core/data_loader.py#L254) reads it and drops 277 of
348 players. Week 10 output was 57 players containing **zero wide receivers**; A.J. Brown,
Ja'Marr Chase, and Justin Jefferson were all filtered out. Root cause is upstream: `snap_pct`
is null or zero in 56% of scraped rows because PFR snap-table parsing is unreliable. The
same measure via nflverse is 5.7% for skill positions.

Also note `_calculate_league_averages()` returns hardcoded constants under a
`# For now, return default values` comment, and `opponent_analyzer` assumes
`league_avg_pass = 200`.

## Direction

The system is mid-rebuild. Settled decisions:

- **Data source → nflverse (`nflreadpy`)**, replacing the PFR/ESPN Selenium scrapers.
  Validated: 150 columns vs 28 scraped, includes `target_share` and `air_yards_share`.
- **Model → usage × efficiency.** Project team volume, then player usage share, then
  efficiency shrunk toward positional baseline. Averaging the *product* (what the current
  engine does) bakes efficiency noise into the forecast.
- **Scoring → DraftKings classic** (full PPR, +3 bonuses at 100 rush / 100 rec / 300 pass
  yards). The bonuses are nonlinear, so expected fantasy points cannot be derived from
  expected yards — you need `P(yards ≥ 100)`, hence a distributional layer.
- **Edge thesis:** volume props (receptions, carries, attempts) are usage-driven and
  predictable; yardage props are efficiency-driven and mostly noise. NFL sides/spreads are
  efficient — do not model them.
- **Steer on calibration and closing-line value, never on early ROI.** At ~50 bets, ROI is
  statistically indistinguishable from noise.
- **Delivery → Telegram push**, not a webapp. Decisions happen on a phone on Sunday morning.

## Landmines

**Three incompatible team-abbreviation conventions coexist.** `data/team_map.xlsx` is the bridge (`full_team_name` / `team_abbrev` / `roster_abbrev`):

| Source | Convention | Green Bay / Kansas City / New Orleans |
|---|---|---|
| `data/game_data/*.csv` `team` col, `data/nfl-2025-*.csv` schedule | PFR (`team_abbrev`) | `GNB` / `KAN` / `NOR` |
| `data/roster.xlsx`, `data/projections/nfl25_proj_week*.csv` `team` col | ESPN (`roster_abbrev`) | `GB` / `KC` / `NO` |
| `data/game_data/*.csv` `opponent`, `home_team`, `away_team` cols | full names | `Green Bay Packers` |

The `team` column in game data is itself mixed — some rows carry PFR abbreviations, others full names. Any join on team identity needs an explicit normalization step; `load_team_name_mapping()` in [picks_agent.py:87](picks_agent.py#L87) is the reference implementation (its hardcoded fallback is what produces the ESPN-style abbreviations).

**Week-number zero-padding is inconsistent.** Odds use `data/odds/week_04/..._week_04.csv` (`{week:02d}`); projections use `data/projections/nfl25_proj_week4.csv` (unpadded). Insights/nuggets/fun_stats use padded (`grok_insights_week_04.json`).

**`RUN.md` is partly stale.** It refers to a `projection.py` that no longer exists (now `run_projections.py`), shows `run_season_projections.py <week>` when the script takes no week argument and auto-detects completed weeks, and lists pre-reorganization paths (`data/game_data_2024.csv` → now `data/game_data/game_data_2024.csv`; `data/team_season_totals.csv` → now `data/season_projections/team_season_totals.csv`). Trust the code over `RUN.md`.

**The three-abbreviation problem only affects legacy code.** `nfl_source.py` uses one
nflverse convention throughout (32 teams; note `LA` for the Rams, `LAC` for the
Chargers) and joins on stable `player_id`, so new code needs no `team_map.xlsx` and no
name matching. The table above still applies to `data/game_data/*.csv`, `roster.xlsx`,
and anything reading them — which is all legacy and slated for removal.

**Generated artifacts are committed.** `data/projections/`, `data/season_projections/`, `data/odds/`, `data/insights/`, and the root `nfl_week_*_analysis_*.html/csv` files are checked in. Re-running a pipeline stage rewrites tracked files — expect a dirty tree, and don't mistake regenerated output for a real diff.
