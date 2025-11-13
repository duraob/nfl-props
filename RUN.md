# NFL Data Pipeline - Weekly Run Guide

This document provides step-by-step instructions for running each core module in the NFL data pipeline for weekly operations.

# Run

1. python nfl_data.py 2025 --- Scrape 2025 season data (run weekly after games complete)
2. python roster_scraper.py --- Update active player roster [ DO THIS END OF DAY WEDNESDAY ]
3. python injuries.py --- Scrape current injury data [ DO THIS END OF DAY WEDNESDAY ]
4. python run_projections.py (CURRENT WEEK NUMBER) --- Week + player projections (uses available 2025 weeks + 2024 fill)
5. python run_season_projections.py (CURRENT WEEK NUMBER) --- Team projections and standings
6. python odds.py (CURRENT WEEK NUMBER) --- Scrape specific week odds
7. python picks_agent.py --- Analyze current week (automatically detects available weeks)
8. python stats_agent.py (CURRENT WEEK NUMBER) --- Generate insights for any week
9. python utils/insights_formatter.py (CURRENT WEEK NUMBER) --csv

## Overview

The pipeline consists of 8 core modules that work together to provide comprehensive NFL analysis:

1. **nfl_data.py** - Scrapes historical game data from Pro Football Reference
2. **roster_scraper.py** - Updates active player roster from ESPN
3. **injuries.py** - Scrapes current injury data from ESPN NFL injuries page
4. **projection.py** - Generates player projections using time-weighted historical data
5. **run_season_projections.py** - Generates team projections, standings, and playoff probabilities
6. **odds.py** - Scrapes current week odds and player props from The Odds API
7. **picks_agent.py** - Analyzes projections vs odds to identify betting opportunities
8. **stats_agent.py** - Generates statistical insights and betting nuggets

## Prerequisites

### Environment Setup
```bash
# Install required packages
pip install pandas numpy requests beautifulsoup4 undetected-chromedriver python-dotenv xai-sdk

# Create .env file with API keys
echo "ODDS_API_KEY=your_odds_api_key_here" >> .env
echo "GROK_API_KEY=your_grok_api_key_here" >> .env
```

### Required Files
- `data/roster.xlsx` - Active player roster
- `data/team_map.xlsx` - Team name mappings
- `data/player_name_mapping.csv` - Player name mappings (optional)
- `data/nfl-2025-EasternStandardTime.csv` - NFL schedule

## Weekly Run Sequence

### 1. Data Collection (nfl_data.py)

**Purpose**: Scrape fresh NFL game data from Pro Football Reference

**Usage**:
```bash
# Scrape 2024 season data (run once or when needed)
python nfl_data.py 2024

# Scrape 2025 season data (run weekly after games complete)
python nfl_data.py 2025

# Test basic functionality
python nfl_data.py test
```

**Output**: 
- `data/game_data_2024.csv` - Historical 2024 season data
- `data/game_data_2025.csv` - Current 2025 season data

**When to Run**: 
- 2024 data: Once at season start
- 2025 data: Weekly after games complete (typically Tuesday)

---

### 2. Roster Update (roster_scraper.py)

**Purpose**: Update active player roster from ESPN to ensure current team assignments

**Usage**:
```bash
# Update active player roster
python roster_scraper.py
```

**Output**:
- `data/roster.xlsx` - Updated active player roster with current team assignments

**When to Run**: Before generating projections (typically Tuesday-Wednesday)

**Key Features**:
- Scrapes all 32 NFL teams' active rosters from ESPN
- Updates player team assignments and positions
- Ensures projection engine uses current roster data

---

### 3. Injury Data Collection (injuries.py)

**Purpose**: Scrape current injury data from ESPN to exclude unavailable players from projections

**Usage**:
```bash
# Scrape current injury data
python injuries.py
```

**Output**:
- `data/injuries.csv` - Current injury data for all 32 teams

**When to Run**: Before generating projections (typically Tuesday-Wednesday)

**Key Features**:
- Scrapes all 32 NFL teams' injury reports
- Converts team names to standard abbreviations
- Excludes players who are "Out" or "Injured Reserve" from projections
- Uses undetected Chrome driver to avoid bot detection

---

### 4. Player Projection Generation (projection.py)

**Purpose**: Generate player projections using time-weighted historical data

**Usage**:
```bash
# Week 1 projections (uses 10 games from 2024)
python projection.py 1

# Week 2 projections (uses 9 games from 2024 + 1 from 2025)
python projection.py 2

# Week 3+ projections (uses available 2025 weeks + 2024 fill)
python projection.py 3
```

**Output**:
- `data/projections/nfl25_proj_week1.csv` - Week 1 player projections
- `data/projections/nfl25_proj_week2.csv` - Week 2 player projections

**When to Run**: Weekly, after fresh game data, roster, and injury data are available

**Key Features**:
- Time-weighted projections (recent games weighted more heavily)
- Player consolidation (players on current team only)
- Opponent-based matchup adjustments
- **Injury filtering**: Automatically excludes "Out" and "Injured Reserve" players

---

### 5. Team Projection Generation (run_season_projections.py)

**Purpose**: Generate team projections, standings, and playoff probabilities using enhanced methodology

**Usage**:
```bash
# Week 1 team projections (uses 10 games from 2024)
python run_season_projections.py 1

# Week 2 team projections (uses 9 games from 2024 + 1 from 2025)
python run_season_projections.py 2

# Week 3+ team projections (uses available 2025 weeks + 2024 fill)
python run_season_projections.py 3
```

**Output**:
- `data/team_season_totals.csv` - Team standings and season projections
- `data/projections/team_projections_week1.csv` - Week 1 team projections
- `data/projections/game_predictions_week1.csv` - Game outcome predictions

**When to Run**: Weekly, after player projections are complete

**Key Features**:
- **Dynamic week detection**: Only projects uncompleted weeks
- **Team-level projections**: Win/loss predictions, standings, playoff probabilities
- **Monte Carlo simulations**: 1,000 simulations per team for accuracy
- **Variance analysis**: Team-specific volatility modeling
- **Schedule strength**: Opponent difficulty analysis
- **Same methodology as player engine**: 10-game sample with time decay

---

### 6. Odds Collection (odds.py)

**Purpose**: Scrape current week odds and player props from The Odds API

**Usage**:
```bash
# Scrape current week odds (automatically detects week)
python odds.py

# Scrape specific week odds
python odds.py 2

# Test API connection
python -c "from odds import OddsScraper; scraper = OddsScraper(); print('API Test:', scraper.test_api_connection())"
```

**Output**:
- `data/odds/week_01/team_odds_week_01.csv` - Team odds (spreads, totals, moneylines)
- `data/odds/week_01/player_props_week_01.csv` - Player prop odds
- `cache/odds/` - Cached API responses

**When to Run**: Weekly, typically Wednesday-Thursday when odds are posted

**Rate Limits**: 
- Free tier: 3 requests/minute, 500 requests/month
- Automatically enforces rate limiting

---

### 7. Betting Analysis (picks_agent.py)

**Purpose**: Analyze projections vs odds to identify high-confidence betting opportunities

**Usage**:
```bash
# Analyze current week (automatically detects available weeks)
python picks_agent.py

# Analyze specific week
python -c "from picks_agent import main; main(2, use_ai=True)"
```

**Output**:
- `data/insights/grok_insights_week_01.json` - AI-powered betting recommendations
- Console output with top betting opportunities

**When to Run**: After projections and odds are available (typically Thursday-Friday)

**Features**:
- Mathematical edge calculation
- AI-powered contextual analysis using Grok
- Historical performance analysis
- Live search for current news/injuries

---

### 8. Statistical Insights (stats_agent.py)

**Purpose**: Generate statistical insights and betting nuggets from historical data

**Usage**:
```bash
# Generate insights for Week 1 (default)
python stats_agent.py

# Generate insights for specific week
python stats_agent.py 2

# Generate insights for any week
python stats_agent.py 3
```

**Output**:
- `data/nuggets/nuggets_week_01.json` - Raw statistical nuggets
- `data/fun_stats/stats_insights_week_01.json` - AI-generated ESPN-style insights

**When to Run**: After odds are available (typically Thursday-Friday)

**Features**:
- Historical trend analysis
- Career averages vs current lines
- ESPN-style betting insights via Grok AI

---

## Complete Weekly Workflow

### Tuesday (After Games Complete)
```bash
# 1. Scrape fresh game data
python nfl_data.py 2025

# 2. Update active player roster
python roster_scraper.py

# 3. Scrape current injury data
python injuries.py

# 4. Generate player projections for next week (excludes injured players)
python projection.py 2  # Replace 2 with current week + 1

# 5. Generate team projections and standings
python run_season_projections.py 2  # Replace 2 with current week + 1
```

### Wednesday-Thursday (When Odds Posted)
```bash
# 6. Scrape current week odds
python odds.py 2  # Replace 2 with current week

# 7. Generate statistical insights
python stats_agent.py 2  # Replace 2 with current week
```

### Thursday-Friday (Analysis)
```bash
# 8. Run betting analysis
python picks_agent.py

# 9. Format insights for readability
python insights_formatter.py 2 --csv  # Replace 2 with current week
```

## File Structure

```
data/
├── game_data_2024.csv          # Historical season data
├── game_data_2025.csv          # Current season data
├── injuries.csv                # Current injury data
├── roster.xlsx                 # Active player roster
├── team_map.xlsx              # Team name mappings
├── nfl-2025-EasternStandardTime.csv  # NFL schedule
├── projections/
│   ├── nfl25_proj_week1.csv   # Week 1 player projections
│   ├── nfl25_proj_week2.csv   # Week 2 player projections
│   ├── team_projections_week1.csv  # Week 1 team projections
│   └── game_predictions_week1.csv  # Week 1 game predictions
├── team_season_totals.csv     # Team standings and season projections
├── odds/
│   └── week_01/
│       ├── team_odds_week_01.csv
│       └── player_props_week_01.csv
├── insights/
│   └── grok_insights_week_01.json
├── nuggets/
│   └── nuggets_week_01.json
└── fun_stats/
    └── stats_insights_week_01.json
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**: 
   - Odds API: Wait 20 seconds between requests
   - Check monthly usage with `python -c "from odds import OddsScraper; print(OddsScraper().get_monthly_usage())"`

2. **Missing Data Files**:
   - Ensure all required Excel/CSV files are in the `data/` directory
   - Check file permissions and paths

3. **Projection Errors**:
   - Verify game data files exist and have recent data
   - Check that schedule file matches current week

4. **Player Name Mismatches**:
   - Update `data/player_name_mapping.csv` for new players
   - Check `data/roster.xlsx` for current team assignments

### Validation Commands

```bash
# Check data file sizes
ls -la data/game_data_*.csv

# Verify projections were created
ls -la data/projections/

# Check odds data
ls -la data/odds/week_*/

# Test API connections
python -c "from odds import OddsScraper; print('Odds API:', OddsScraper().test_api_connection())"
python -c "from picks_agent import call_grok_api; print('Grok API test')"
```

## Performance Notes

- **nfl_data.py**: Takes 5-10 minutes for full season scrape
- **roster_scraper.py**: Takes 3-5 minutes for all 32 teams
- **injuries.py**: Takes 2-3 minutes for all 32 teams
- **projection.py**: Takes 1-2 minutes for player projections
- **run_season_projections.py**: Takes 2-3 minutes for team projections
- **odds.py**: Takes 2-3 minutes (rate limited)
- **picks_agent.py**: Takes 3-5 minutes (includes AI analysis)
- **stats_agent.py**: Takes 1-2 minutes

## Support

For issues or questions:
1. Check log files: `nfl_data_enhanced.log`, `odds_scraper.log`, `injuries_scraper.log`
2. Verify API keys in `.env` file
3. Ensure all required data files are present
4. Check file permissions and directory structure
