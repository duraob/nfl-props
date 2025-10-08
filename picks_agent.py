"""
NFL Betting Picks Agent with AI Analysis

This module analyzes NFL player props against projections to identify high-confidence
betting opportunities using both mathematical edge calculations and AI-powered contextual
analysis. It integrates with Grok AI to provide intelligent betting recommendations
based on historical performance, matchups, and market conditions.

Key Features:
- Mathematical edge calculation (projection vs line)
- AI-powered contextual analysis using Grok
- Historical performance analysis
- Weather and matchup context
- Result tracking and feedback
"""

import os
import pandas as pd
import numpy as np
import json
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import user, system
from xai_sdk.search import SearchParameters, web_source, x_source

load_dotenv()

# CONFIGURATION - Easy to adjust
GROK_API_KEY = os.getenv('GROK_API_KEY')
STD_DEV_THRESHOLD = 1.5  # Adjust confidence level
MIN_EDGE_PERCENTAGE = 8.0  # Minimum edge to consider
CONFIDENCE_THRESHOLDS = {
    'high': 80,
    'medium': 60,
    'low': 40
}
GROK_API_URL = "https://api.x.ai/v1/chat/completions"


def load_master_roster():
    """
    Load master roster from Excel file to get player team mappings.
    
    Returns:
        dict: Player name to team abbreviation mapping
    """
    try:
        # Read Excel file from Master_Roster sheet
        roster_df = pd.read_excel('data/roster.xlsx', sheet_name='Master_Roster')
        
        # Create player to team mapping using correct column names
        player_team_mapping = dict(zip(roster_df['player_name'], roster_df['team']))
        
        print(f"Loaded {len(player_team_mapping)} player-team mappings from roster.xlsx")
        return player_team_mapping
    except Exception as e:
        print(f"Error loading master roster: {e}")
        return {}


def load_schedule_strength_data(week_number):
    """
    Load schedule strength data for a specific week.
    
    Args:
        week_number (int): Week number
        
    Returns:
        dict: Schedule strength data by team
    """
    try:
        schedule_file = f'data/projections/schedule_strength_week{week_number}.csv'
        if os.path.exists(schedule_file):
            df_schedule = pd.read_csv(schedule_file, index_col='team')
            print(f"Loaded schedule strength data for {len(df_schedule)} teams")
            return df_schedule.to_dict('index')
        else:
            print(f"Schedule strength file not found: {schedule_file}")
            return {}
    except Exception as e:
        print(f"Error loading schedule strength data: {e}")
        return {}


def load_team_name_mapping():
    """
    Load team name mapping from Excel file with roster abbreviation support.
    
    Returns:
        dict: Full team name to roster abbreviation mapping
    """
    try:
        # Read Excel file
        team_df = pd.read_excel('data/team_map.xlsx')
        
        # Create team name to roster abbreviation mapping
        team_mapping = dict(zip(team_df['full_team_name'], team_df['roster_abbrev']))
        
        print(f"Loaded {len(team_mapping)} team mappings from team_map.xlsx (using roster_abbrev)")
        return team_mapping
    except Exception as e:
        print(f"Error loading team mapping: {e}")
        # Fallback to hardcoded mapping if file fails
        return {
            'Arizona Cardinals': 'ARI',
            'Atlanta Falcons': 'ATL',
            'Baltimore Ravens': 'BAL',
            'Buffalo Bills': 'BUF',
            'Carolina Panthers': 'CAR',
            'Chicago Bears': 'CHI',
            'Cincinnati Bengals': 'CIN',
            'Cleveland Browns': 'CLE',
            'Dallas Cowboys': 'DAL',
            'Denver Broncos': 'DEN',
            'Detroit Lions': 'DET',
            'Green Bay Packers': 'GB',
            'Houston Texans': 'HOU',
            'Indianapolis Colts': 'IND',
            'Jacksonville Jaguars': 'JAC',
            'Kansas City Chiefs': 'KC',
            'Las Vegas Raiders': 'LV',
            'Los Angeles Chargers': 'LAC',
            'Los Angeles Rams': 'LAR',
            'Miami Dolphins': 'MIA',
            'Minnesota Vikings': 'MIN',
            'New England Patriots': 'NE',
            'New Orleans Saints': 'NOR',  # Fixed: Should be NOR not NO
            'New York Giants': 'NYG',
            'New York Jets': 'NYJ',
            'Philadelphia Eagles': 'PHI',
            'Pittsburgh Steelers': 'PIT',
            'San Francisco 49ers': 'SF',
            'Seattle Seahawks': 'SEA',
            'Tampa Bay Buccaneers': 'TB',
            'Tennessee Titans': 'TEN',
            'Washington Commanders': 'WAS'
        }


def determine_opponent(player_name, home_team, away_team, roster_mapping, player_mapping, team_name_mapping):
    """
    Determine the opponent team for a player using roster data and player mapping.
    
    Args:
        player_name (str): Player name from odds data
        home_team (str): Home team full name
        away_team (str): Away team full name
        roster_mapping (dict): Projection name to team mapping
        player_mapping (dict): Odds name to projection name mapping
        team_name_mapping (dict): Full team name to abbreviation mapping
    
    Returns:
        str: Opponent team full name
    """
    # Convert team names to abbreviations for comparison
    home_team_abbr = team_name_mapping.get(home_team, home_team)
    away_team_abbr = team_name_mapping.get(away_team, away_team)
    
    player_team = None
    
    # First, try to find the player in roster using the odds name directly
    if player_name in roster_mapping:
        player_team = roster_mapping[player_name]
        print(f"Found {player_name} in roster with team: {player_team}")
    # If not found, try using the player mapping to get projection name
    elif player_name in player_mapping:
        projection_name = player_mapping[player_name]
        print(f"Using player mapping: {player_name} -> {projection_name}")
        if projection_name in roster_mapping:
            player_team = roster_mapping[projection_name]
            print(f"Found {projection_name} in roster with team: {player_team}")
    
    if player_team:
        # Determine opponent based on which team the player is NOT on
        if player_team == home_team_abbr:
            print(f"{player_name} is on home team ({home_team}), opponent is {away_team}")
            return away_team
        elif player_team == away_team_abbr:
            print(f"{player_name} is on away team ({away_team}), opponent is {home_team}")
            return home_team
        else:
            print(f"Warning: {player_name} team {player_team} doesn't match game teams ({home_team_abbr} vs {away_team_abbr})")
            # Fallback: assume player is on home team
            return away_team
    else:
        # Fallback: assume player is on home team
        print(f"Warning: Player {player_name} not found in roster or mapping, defaulting to away team as opponent")
        return away_team


def create_player_mapping():
    """
    Create mapping between odds player names and projection player names.
    
    Returns:
        dict: Mapping of odds names to projection names
    """
    # Try to load comprehensive mapping first
    mapping_file = 'data/player_name_mapping.csv'
    if os.path.exists(mapping_file):
        try:
            mapping_df = pd.read_csv(mapping_file)
            mapping = dict(zip(mapping_df['odds_name'], mapping_df['projection_name']))
            print(f"Loaded {len(mapping)} player mappings from {mapping_file}")
            return mapping
        except Exception as e:
            print(f"Warning: Could not load mapping file: {e}")
    return {}

def calculate_player_edge(projection, line, prop_type):
    """
    Calculate edge percentage for player props using Monte Carlo data.
    
    Args:
        projection (float): Projected player performance
        line (float): Betting line/over-under
        prop_type (str): Type of prop (rush_yds, rec_yds, etc.)
    
    Returns:
        float: Edge percentage
    """
    # Initialize edge variable to ensure it's always defined
    edge = 0.0
    
    if prop_type in ['rush_yds', 'reception_yds', 'pass_yds', 'receptions', 'pass_attempts', 'pass_completions', 'pass_tds', 'pass_interceptions', 'rush_att']:
        # For all numeric props, use the standard edge calculation
        edge = ((projection - line) / line) * 100
    elif prop_type == 'anytime_td':
        # Convert odds to implied probability
        if line > 0:
            implied_prob = 100 / (line + 100)
        else:
            implied_prob = abs(line) / (abs(line) + 100)
        # Use projection probability (calculated with Monte Carlo data)
        edge = (projection - implied_prob) * 100
    else:
        # Handle unsupported prop types
        print(f"Warning: Unsupported prop type '{prop_type}' for edge calculation")
        edge = 0.0
    
    return edge


def identify_high_confidence_bets(projections_df, player_props_df, mapping):
    """
    Find bets meeting edge and confidence criteria.
    
    Args:
        projections_df (DataFrame): Player projections data
        player_props_df (DataFrame): Player props odds data
        mapping (dict): Player name mapping
    
    Returns:
        list: List of high confidence betting opportunities
    """
    high_confidence_bets = []
    
    # Debug: Show unique prop types in the data
    unique_prop_types = player_props_df['prop_type'].unique()
    print(f"Found prop types: {unique_prop_types}")
    
    # Track best odds for each unique player/prop combination
    best_odds_tracker = {}
    
    for _, prop in player_props_df.iterrows():
        player_name = prop['player_name']
        if player_name in mapping:
            projection_name = mapping[player_name]
            projection_data = projections_df[projections_df['name'] == projection_name]
            
            if not projection_data.empty:
                prop_type = prop['prop_type']
                line = prop['point']
                odds = prop['price']
                
                # Create unique key for this player/prop combination
                unique_key = (player_name, prop_type, line)
                
                # Get projection based on prop type using correct column names (no proj_ prefix)
                if prop_type == 'rush_yds':
                    projection = projection_data['rush_yd'].iloc[0]
                elif prop_type == 'reception_yds':
                    projection = projection_data['rec_yd'].iloc[0]
                elif prop_type == 'receptions':
                    projection = projection_data['rec'].iloc[0]
                elif prop_type == 'pass_yds':
                    projection = projection_data['pass_yd'].iloc[0]
                elif prop_type == 'pass_attempts':
                    projection = projection_data['pass_att'].iloc[0]
                elif prop_type == 'pass_completions':
                    projection = projection_data['pass_cmp'].iloc[0]
                elif prop_type == 'pass_tds':
                    projection = projection_data['pass_td'].iloc[0]
                elif prop_type == 'pass_interceptions':
                    projection = projection_data['pass_int'].iloc[0]
                elif prop_type == 'rush_att':
                    projection = projection_data['rush_att'].iloc[0]
                elif prop_type == 'anytime_td':
                    # Use Monte Carlo TD data for sophisticated anytime TD probability calculation
                    rush_td_p25 = projection_data['rush_td_p25'].iloc[0]
                    rec_td_p25 = projection_data['rec_td_p25'].iloc[0]
                    
                    # Calculate anytime TD probability using 25th percentile for conservative estimate
                    # Method: P(anytime TD) = 1 - P(no rush TD) * P(no rec TD)
                    rush_td_prob = min(1.0, max(0.0, rush_td_p25))
                    rec_td_prob = min(1.0, max(0.0, rec_td_p25))
                    
                    # Calculate anytime TD probability (1 - P(no rush TD) * P(no rec TD))
                    projection = 1 - (1 - rush_td_prob) * (1 - rec_td_prob)
                else:
                    print(f"Warning: Unsupported prop type '{prop_type}' - skipping")
                    continue
                
                # Calculate edge
                edge = calculate_player_edge(projection, line, prop_type)
                
                # Calculate confidence using Monte Carlo variance data
                if prop_type == 'anytime_td':
                    # Use Monte Carlo variance for anytime TD confidence
                    rush_td_std = projection_data['rush_td_std'].iloc[0]
                    rec_td_std = projection_data['rec_td_std'].iloc[0]
                    combined_variance = (rush_td_std + rec_td_std) / 2
                else:
                    # Map prop types to correct column names
                    column_mapping = {
                        'rush_yds': 'rush_yd_std',
                        'reception_yds': 'rec_yd_std', 
                        'pass_yds': 'pass_yd_std',
                        'receptions': 'rec_std',
                        'pass_attempts': 'pass_att_std',
                        'pass_completions': 'pass_cmp_std',
                        'pass_tds': 'pass_td_std',
                        'pass_interceptions': 'pass_int_std',
                        'rush_att': 'rush_att_std',
                        'anytime_td': 'anytime_td_std'
                    }
                    
                    std_column = column_mapping.get(prop_type, f'{prop_type}_std')
                    if std_column in projection_data.columns:
                        stat_std = projection_data[std_column].iloc[0]
                        combined_variance = stat_std
                    else:
                        # Fallback to a default variance if column not found
                        combined_variance = 0.5
                
                # Use Monte Carlo variance for confidence scoring
                if combined_variance <= 0.3:  # Low variance = high confidence
                    confidence = 'high'
                elif combined_variance <= 0.6:  # Medium variance = medium confidence
                    confidence = 'medium'
                else:
                    confidence = 'low'
                
                # Calculate deviations using Monte Carlo variance
                deviations = abs(projection - line) / (combined_variance + 0.1)  # Add small constant to avoid division by zero
                
                if edge >= MIN_EDGE_PERCENTAGE and deviations >= STD_DEV_THRESHOLD:
                    
                    bet_data = {
                        'player_name': player_name,
                        'prop_type': prop_type,
                        'line': line,
                        'projection': projection,
                        'edge_percentage': edge,
                        'standard_deviations': deviations,
                        'confidence': confidence,
                        'odds': odds,
                        'event_id': prop['event_id'],
                        'home_team': prop['home_team'],
                        'away_team': prop['away_team']
                    }
                    
                    # Check if we already have this player/prop combination
                    if unique_key in best_odds_tracker:
                        # Keep the entry with the maximum odds (best value for bettor)
                        existing_odds = best_odds_tracker[unique_key]['odds']
                        if odds > existing_odds:
                            # Replace with better odds
                            best_odds_tracker[unique_key] = bet_data
                    else:
                        # First time seeing this combination
                        best_odds_tracker[unique_key] = bet_data
    
    # Convert tracker to list
    high_confidence_bets = list(best_odds_tracker.values())
    
    print(f"Eliminated duplicates: {len(high_confidence_bets)} unique bets found")
    
    return high_confidence_bets


def get_historical_performance(player_name, opponent, weeks_back=3):
    """
    Get player's historical performance against specific opponent and recent form with Monte Carlo context.
    
    Args:
        player_name (str): Player name to analyze
        opponent (str): Opponent team abbreviation
        weeks_back (int): Number of recent weeks to analyze
    
    Returns:
        dict: Historical performance data with Monte Carlo context
    """
    try:
        # Load historical game data
        historical_data = []
        for year in [2024, 2023, 2022]:  # Last 3 years
            file_path = f'data/game_data/game_data_{year}.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                player_games = df[df['player'] == player_name]
                historical_data.append(player_games)
        
        if not historical_data:
            return {'error': 'No historical data found'}
        
        all_games = pd.concat(historical_data, ignore_index=True)
        
        # Get games vs this opponent
        vs_opponent = all_games[all_games['opponent'] == opponent]
        
        # Get recent games (last N weeks)
        recent_games = all_games.tail(weeks_back * 2)  # Approximate recent games
        
        performance = {
            'vs_opponent_games': len(vs_opponent),
            'recent_games': len(recent_games),
            'avg_snap_pct': recent_games['snap_pct'].mean() if not recent_games.empty else 0,
            'avg_targets': recent_games['targets'].mean() if not recent_games.empty else 0,
            'avg_receptions': recent_games['receptions'].mean() if not recent_games.empty else 0,
            'avg_rec_yds': recent_games['rec_yds'].mean() if not recent_games.empty else 0,
            'avg_rush_yds': recent_games['rush_yds'].mean() if not recent_games.empty else 0,
            'avg_pass_yds': recent_games['pass_yds'].mean() if not recent_games.empty else 0,
            'weather_conditions': recent_games['weather'].iloc[-1] if not recent_games.empty else 'Unknown',
            'monte_carlo_context': {
                'uncertainty_level': 'medium',  # Default uncertainty level
                'risk_assessment': 'moderate'   # Default risk assessment
            }
        }
        
        return performance
    except Exception as e:
        return {'error': f'Error analyzing historical data: {str(e)}'}


def validate_team_assignments(player_name, opponent, roster_mapping, player_mapping, team_name_mapping):
    """
    Validate team assignments and return corrected information.
    
    Args:
        player_name (str): Player name from odds data
        opponent (str): Determined opponent team
        roster_mapping (dict): Projection name to team mapping
        player_mapping (dict): Odds name to projection name mapping
        team_name_mapping (dict): Full team name to abbreviation mapping
    
    Returns:
        dict: Validation results with player team and opponent information
    """
    # Determine player's actual team
    player_team = None
    player_team_full = None
    
    # First, try to find the player in roster using the odds name directly
    if player_name in roster_mapping:
        player_team = roster_mapping[player_name]
        # Convert abbreviation to full name
        for full_name, abbr in team_name_mapping.items():
            if abbr == player_team:
                player_team_full = full_name
                break
    # If not found, try using the player mapping to get projection name
    elif player_name in player_mapping:
        projection_name = player_mapping[player_name]
        if projection_name in roster_mapping:
            player_team = roster_mapping[projection_name]
            # Convert abbreviation to full name
            for full_name, abbr in team_name_mapping.items():
                if abbr == player_team:
                    player_team_full = full_name
                    break
    
    validation_result = {
        'player_name': player_name,
        'player_team_abbr': player_team,
        'player_team_full': player_team_full,
        'opponent': opponent,
        'is_valid': player_team is not None,
        'validation_timestamp': datetime.now().isoformat()
    }
    
    if not validation_result['is_valid']:
        print(f"WARNING: Could not validate team assignment for {player_name}")
    
    return validation_result


def build_player_context(player_name, prop_type, line, projection, opponent, week, team_validation=None, 
                        schedule_strength_data=None, monte_carlo_data=None):
    """
    Build comprehensive context for a single player bet.
    
    Args:
        player_name (str): Player name
        prop_type (str): Type of prop bet
        line (float): Betting line
        projection (float): Model projection
        opponent (str): Opponent team
        week (int): Week number
        team_validation (dict): Team validation results
        schedule_strength_data (dict): Schedule strength data
        monte_carlo_data (dict): Monte Carlo simulation results
    
    Returns:
        dict: Complete player context for AI analysis
    """
    # Get historical performance
    historical = get_historical_performance(player_name, opponent)
    
    # Calculate mathematical edge
    edge = calculate_player_edge(projection, line, prop_type)
    
    context = {
        'player': player_name,
        'prop_type': prop_type,
        'line': line,
        'projection': projection,
        'edge_percentage': edge,
        'opponent': opponent,
        'week': week,
        'historical_performance': historical,
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    # Add team validation information if provided
    if team_validation:
        context['team_validation'] = team_validation
    
    # Add schedule strength context
    if schedule_strength_data and team_validation:
        player_team = team_validation.get('player_team_abbr')
        if player_team and player_team in schedule_strength_data:
            context['schedule_strength'] = schedule_strength_data[player_team]
    
    # Add Monte Carlo uncertainty context
    if monte_carlo_data and player_name in monte_carlo_data:
        player_mc_data = monte_carlo_data[player_name]
        context['monte_carlo_uncertainty'] = {
            'available_stats': list(player_mc_data.keys()),
            'uncertainty_level': 'high' if len(player_mc_data) > 3 else 'medium'
        }
        
        # Add specific P25/P75 data for the prop type
        prop_mapping = {
            'rush_yds': 'rush_yd',
            'reception_yds': 'rec_yd', 
            'pass_yds': 'pass_yd',
            'pass_attempts': 'pass_att',
            'pass_completions': 'pass_cmp',
            'pass_tds': 'pass_td',
            'pass_interceptions': 'pass_int',
            'rush_att': 'rush_att',
            'receptions': 'rec',
            'anytime_td': 'anytime_td'  # Special case for Monte Carlo TD probability
        }
        
        mc_stat = prop_mapping.get(prop_type)
        if mc_stat and mc_stat in player_mc_data:
            stat_data = player_mc_data[mc_stat]
            if 'percentiles' in stat_data:
                context['monte_carlo_uncertainty']['p25'] = stat_data['percentiles'].get('p25')
                context['monte_carlo_uncertainty']['p75'] = stat_data['percentiles'].get('p75')
                context['monte_carlo_uncertainty']['std'] = stat_data.get('std')
    
    return context


def get_live_search_context(player_name, week, opponent=None):
    """
    Get live search context for a player using Grok's live search functionality.
    
    Args:
        player_name (str): Player name to search for
        week (int): Current week number
        opponent (str, optional): Opponent team (for context in search prompt)
    
    Returns:
        dict: Live search context including injuries, news, and expert insights
    """
    if not GROK_API_KEY:
        return {'error': 'GROK_API_KEY not found for live search'}
    
    try:
        # Initialize X.AI client
        client = Client(api_key=GROK_API_KEY)
        
        # Create conversation for live search
        chat = client.chat.create(
            model="grok-4",
            search_parameters=SearchParameters(
                max_search_results=3,
                return_citations=True,
                from_date=datetime.now() - timedelta(days=2),
                sources=[
                    web_source(allowed_websites=["https://www.nfl.com/news/",
                                                 "https://www.espn.com/nfl/injuries"]),
                    x_source(included_x_handles=["rapsheet", "adamschefter"])
                ]
            )
        )
        
        # Create search prompt
        system_prompt = f"""
You are an expert NFL betting analyst with deep knowledge of player performance, matchups, and market conditions. Use live search to get the most current information when needed.
"""

        
        # Build search prompt with optional opponent context
        opponent_context = f"Opponent: {opponent}" if opponent else "Current team and upcoming games"
        
        search_prompt = f"""
Search for current information about {player_name} for Week {week} NFL games. Focus on:

1. Injury status and practice participation from NFL.com/injuries
2. Recent news and developments from ESPN.com/nfl and NFL.com/news
3. Expert analysis from RapSheet and AdamSchefter
4. Weather conditions for the game
5. Role changes or lineup adjustments

Player: {player_name}
{opponent_context}
Week: {week}
"""
        
        # Perform live search with domain restrictions
        chat.append(system(system_prompt))
        chat.append(user(search_prompt))
        response = chat.sample()
        
        return {
            'success': True,
            'live_search_results': response.content,
            'player': player_name,
            'opponent': opponent,
            'week': week
        }
        
    except Exception as e:
        return {'error': f'Live search failed: {str(e)}'}


def call_grok_api(prompt, include_live_search=True):
    """
    Make API call to Grok using X.AI SDK with optional live search.
    
    Args:
        prompt (str): Formatted prompt for Grok
        include_live_search (bool): Whether to enable live search functionality
    
    Returns:
        dict: Grok's response or error
    """
    if not GROK_API_KEY:
        return {'error': 'GROK_API_KEY not found in environment variables'}
    
    try:
        # Initialize X.AI client
        client = Client(api_key=GROK_API_KEY)
        
        # Create a conversation using the correct API
        chat = client.chat.create(model="grok-3")
        
        # Combine system message and user prompt
        if include_live_search:
            full_prompt = f"""You are an expert NFL betting analyst with deep knowledge of player performance, matchups, and market conditions. Use live search to get the most current information when needed.

{prompt}"""
        else:
            full_prompt = f"""You are an expert NFL betting analyst with deep knowledge of player performance, matchups, and market conditions.

{prompt}"""
        
        # Add user prompt and get response
        chat.append(system(full_prompt))
        response = chat.sample()
        
        return {
            'success': True,
            'analysis': response.content
        }
        
    except Exception as e:
        return {'error': f'X.AI API request failed: {str(e)}'}


def generate_ai_analysis_prompt(player_contexts):
    """
    Create focused prompt for Grok AI analysis with live search context and explicit team validation.
    
    Args:
        player_contexts (list): List of player context dictionaries with live search data
    
    Returns:
        str: Formatted prompt for Grok
    """
    # Filter to top candidates with good edges - increased to 25
    top_candidates = [ctx for ctx in player_contexts if ctx['edge_percentage'] >= MIN_EDGE_PERCENTAGE]
    top_candidates = sorted(top_candidates, key=lambda x: x['edge_percentage'], reverse=True)[:25]
    
    if not top_candidates:
        return "No qualifying betting opportunities found with sufficient edge."
    
    prompt = f"""
You are an expert NFL betting analyst with Monte Carlo uncertainty quantification expertise. Analyze these top 25 betting opportunities for Week {top_candidates[0]['week']} using Monte Carlo uncertainty data, historical performance, and real-time information.

CRITICAL INSTRUCTIONS - READ FIRST:
1. **USE ONLY THE PROVIDED TEAM ASSIGNMENTS**: The roster.xlsx file contains the current, verified team assignments for Week {top_candidates[0]['week']}. Use these assignments exactly as provided.
2. **AUTHORITATIVE DATA SOURCE**: roster.xlsx is the authoritative source for all current team assignments. Do not attempt to 'correct' or 'update' these assignments.
3. **MONTE CARLO FOCUS**: Use uncertainty quantification data for risk assessment
4. **DATA SOURCE ATTRIBUTION**: All information below is from verified sources as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
    
    for i, ctx in enumerate(top_candidates, 1):
        # Include live search results if available
        live_search_info = ""
        if 'live_search_results' in ctx and ctx['live_search_results']:
            live_search_info = f"\nLive Search Results (NFL.com/ESPN.com - real-time): {ctx['live_search_results']}"
        
        # Get team validation information with Monte Carlo context
        team_info = ""
        if 'team_validation' in ctx and ctx['team_validation']:
            validation = ctx['team_validation']
            if validation['is_valid']:
                # Add Monte Carlo context to team info
                monte_carlo_context = ""
                if 'monte_carlo_context' in ctx.get('historical_performance', {}):
                    mc = ctx['historical_performance']['monte_carlo_context']
                    monte_carlo_context = f" (Uncertainty: {mc['uncertainty_level']}, Risk: {mc['risk_assessment']})"
                
                team_info = f"TEAM ASSIGNMENT (VERIFIED): {ctx['player']} ({validation['player_team_full']}) vs {validation['opponent']}{monte_carlo_context}\nSource: roster.xlsx - Week {ctx['week']} current roster\nStatus: CONFIRMED - Use this assignment exactly as provided"
            else:
                team_info = f"TEAM ASSIGNMENT: {ctx['player']} - Team assignment not found in roster.xlsx"
        else:
            team_info = f"TEAM ASSIGNMENT: {ctx['player']} vs {ctx['opponent']} (roster.xlsx validation not available)"
        
        # Add schedule strength context
        schedule_info = ""
        if 'schedule_strength' in ctx:
            schedule_data = ctx['schedule_strength']
            schedule_info = f"\nSchedule Strength (schedule_strength_week{ctx['week']}.csv): Team has faced {schedule_data.get('games', 0)} games with strength ratio {schedule_data.get('schedstr_ratio_def_pass_yd', 1.0):.2f}"
        
        # Add Monte Carlo uncertainty context
        uncertainty_info = ""
        if 'monte_carlo_uncertainty' in ctx:
            mc_data = ctx['monte_carlo_uncertainty']
            uncertainty_info = f"\nMonte Carlo Analysis (monte_carlo_week{ctx['week']}.json): Uncertainty Level: {mc_data.get('uncertainty_level', 'unknown')}"
            if 'p25' in mc_data and 'p75' in mc_data:
                uncertainty_info += f", P25: {mc_data['p25']:.1f}, P75: {mc_data['p75']:.1f}"
            if 'std' in mc_data:
                uncertainty_info += f", Std Dev: {mc_data['std']:.1f}"
        
        prompt += f"""
OPPORTUNITY {i}:
{team_info}
Prop: {ctx['prop_type']} {ctx['line']}
Projection: {ctx['projection']:.1f} (nfl25_proj_week{ctx['week']}.csv)
Edge: {ctx['edge_percentage']:.1f}% (calculated from projection vs line)
Recent Performance: {ctx['historical_performance'].get('avg_snap_pct', 0):.1f}% snaps, {ctx['historical_performance'].get('avg_targets', 0):.1f} targets (game_data 2022-2024)
Weather: {ctx['historical_performance'].get('weather_conditions', 'Unknown')}{schedule_info}{uncertainty_info}
Latest News and Expert Insights: {live_search_info}

"""
    
    prompt += """
Based on this comprehensive data (Monte Carlo uncertainty + historical performance + real-time information), select the TOP 10 betting opportunities and provide:

1. PLAYER NAME
2. PROP TYPE and LINE
3. RECOMMENDATION (Over/Under/Yes/No)
4. CONFIDENCE LEVEL (High/Medium/Low)
5. KEY REASONING (2-3 sentences explaining why this bet is strong)
6. RISK FACTORS (What could go wrong)

CRITICAL REQUIREMENTS:
- Use ONLY the team assignments provided above from roster.xlsx - these are current and verified for Week {top_candidates[0]['week']}
- Do not attempt to 'correct' or 'update' team assignments based on your training data
- All team assignments are authoritative and current from roster.xlsx
- Use Monte Carlo uncertainty data for risk assessment and confidence scoring
- Trust the provided data over any training data you may have
- Focus on the mathematical edge and Monte Carlo uncertainty data provided

Focus on:
- Monte Carlo uncertainty quantification and variance analysis
- Player's recent form and snap percentage trends
- Historical performance vs this opponent
- Current injury status and practice participation
- Recent news and developments
- Expert analysis and insider information
- Weather conditions and their impact
- Projection vs line edge with uncertainty context
- Market context and value with risk assessment

Use live search to get the most current information when needed. Return your analysis in a clear, structured format.
"""
    
    return prompt


def validate_grok_response(ai_analysis, player_contexts):
    """
    Validate GROK response for team assignment accuracy and flag potential hallucinations.
    
    Args:
        ai_analysis (dict): GROK's analysis response
        player_contexts (list): Player context data used for analysis
    
    Returns:
        dict: Validation results with any discrepancies found
    """
    validation_results = {
        'total_players_analyzed': len(player_contexts),
        'team_assignment_discrepancies': [],
        'validation_timestamp': datetime.now().isoformat(),
        'is_valid': True
    }
    
    # Create a mapping of player names to their validated team information
    player_team_map = {}
    for ctx in player_contexts:
        if 'team_validation' in ctx and ctx['team_validation']['is_valid']:
            player_team_map[ctx['player']] = {
                'player_team': ctx['team_validation']['player_team_full'],
                'opponent': ctx['team_validation']['opponent']
            }
    
    # Check the AI analysis text for team assignment discrepancies
    analysis_text = ai_analysis.get('analysis', '')
    
    for player_name, team_info in player_team_map.items():
        # Look for the player in the analysis text
        if player_name in analysis_text:
            # Check if the analysis mentions the correct team
            correct_team = team_info['player_team']
            correct_opponent = team_info['opponent']
            
            # Add null check to prevent TypeError
            if correct_team is not None and correct_team not in analysis_text:
                validation_results['team_assignment_discrepancies'].append({
                    'player': player_name,
                    'expected_team': correct_team,
                    'expected_opponent': correct_opponent,
                    'issue': f'Analysis does not mention correct team {correct_team}',
                    'severity': 'high'
                })
                validation_results['is_valid'] = False
    
    if validation_results['team_assignment_discrepancies']:
        print(f"WARNING: Found {len(validation_results['team_assignment_discrepancies'])} team assignment discrepancies in GROK response")
        for discrepancy in validation_results['team_assignment_discrepancies']:
            print(f"  - {discrepancy['player']}: {discrepancy['issue']}")
    else:
        print("✅ Team assignment validation passed - no discrepancies found")
    
    return validation_results


def save_ai_insights(week_number, ai_analysis, player_contexts):
    """
    Save AI analysis insights to JSON file with validation results.
    
    Args:
        week_number (int): NFL week number
        ai_analysis (dict): Grok's analysis response
        player_contexts (list): Player context data used for analysis
    """
    insights_dir = "data/insights"
    os.makedirs(insights_dir, exist_ok=True)
    
    # Validate the GROK response
    validation_results = validate_grok_response(ai_analysis, player_contexts)
    
    insights_data = {
        'week': week_number,
        'analysis_timestamp': datetime.now().isoformat(),
        'ai_analysis': ai_analysis,
        'player_contexts': player_contexts,
        'validation_results': validation_results,
        'analysis_metadata': {
            'total_opportunities_analyzed': len(player_contexts),
            'edge_threshold': MIN_EDGE_PERCENTAGE,
            'std_dev_threshold': STD_DEV_THRESHOLD,
            'team_validation_enabled': True
        }
    }
    
    filename = f"{insights_dir}/grok_insights_week_{week_number:02d}.json"
    with open(filename, 'w') as f:
        json.dump(insights_data, f, indent=2)
    
    print(f"Saved AI insights to {filename}")
    
    # Log validation results
    if not validation_results['is_valid']:
        print(f"⚠️  Validation failed: {len(validation_results['team_assignment_discrepancies'])} discrepancies found")
    else:
        print("✅ Validation passed: All team assignments are accurate")


def analyze_with_ai(week_number, use_live_search=False):
    """
    Perform AI-powered analysis of betting opportunities.
    
    Args:
        week_number (int): NFL week number to analyze
        use_live_search (bool): Whether to enable live search functionality (default: False)
    
    Returns:
        dict: AI analysis results
    """
    print(f"Starting AI analysis for Week {week_number}...")
    
    # Load data
    if week_number == 0:
        projections_file = 'data/projections/nfl25_proj_week0.csv'
    else:
        projections_file = f'data/projections/nfl25_proj_week{week_number}.csv'
    
    if not os.path.exists(projections_file):
        print(f"Warning: {projections_file} not found, using week 0 projections")
        projections_file = 'data/projections/nfl25_proj_week0.csv'
    
    projections_df = pd.read_csv(projections_file)
    player_props_df = pd.read_csv(f'data/odds/week_{week_number:02d}/player_props_week_{week_number:02d}.csv')
    
    # Load enhanced data
    schedule_strength_data = load_schedule_strength_data(week_number)
    
    # Load Monte Carlo data
    monte_carlo_data = {}
    monte_carlo_file = f'data/projections/monte_carlo_week{week_number}.json'
    if os.path.exists(monte_carlo_file):
        try:
            with open(monte_carlo_file, 'r') as f:
                monte_carlo_data = json.load(f)
            print(f"Loaded Monte Carlo data for {len(monte_carlo_data)} players")
        except Exception as e:
            print(f"Error loading Monte Carlo data: {e}")
    
    # Create player mapping and load roster data
    mapping = create_player_mapping()
    roster_mapping = load_master_roster()
    team_name_mapping = load_team_name_mapping()
    
    # Build player contexts for AI analysis
    player_contexts = []
    
    for _, prop in player_props_df.iterrows():
        player_name = prop['player_name']
        if player_name in mapping:
            projection_name = mapping[player_name]
            projection_data = projections_df[projections_df['name'] == projection_name]
            
            if not projection_data.empty:
                prop_type = prop['prop_type']
                line = prop['point']
                odds = prop['price']
                
                # Get projection based on prop type using correct column names
                projection = None
                if prop_type == 'rush_yds':
                    projection = projection_data['rush_yd'].iloc[0]
                elif prop_type == 'reception_yds':
                    projection = projection_data['rec_yd'].iloc[0]
                elif prop_type == 'receptions':
                    projection = projection_data['rec'].iloc[0]
                elif prop_type == 'pass_yds':
                    projection = projection_data['pass_yd'].iloc[0]
                elif prop_type == 'pass_attempts':
                    projection = projection_data['pass_att'].iloc[0]
                elif prop_type == 'pass_completions':
                    projection = projection_data['pass_cmp'].iloc[0]
                elif prop_type == 'pass_tds':
                    projection = projection_data['pass_td'].iloc[0]
                elif prop_type == 'pass_interceptions':
                    projection = projection_data['pass_int'].iloc[0]
                elif prop_type == 'rush_att':
                    projection = projection_data['rush_att'].iloc[0]
                elif prop_type == 'anytime_td':
                    rush_td = projection_data['rush_td'].iloc[0]
                    rec_td = projection_data['rec_td'].iloc[0]
                    projection = rush_td + rec_td
                
                if projection is not None:
                    # Determine opponent using roster data and player mapping
                    opponent = determine_opponent(
                        player_name=player_name,
                        home_team=prop['home_team'],
                        away_team=prop['away_team'],
                        roster_mapping=roster_mapping,
                        player_mapping=mapping,
                        team_name_mapping=team_name_mapping
                    )
                    
                    # Validate team assignments
                    team_validation = validate_team_assignments(
                        player_name=player_name,
                        opponent=opponent,
                        roster_mapping=roster_mapping,
                        player_mapping=mapping,
                        team_name_mapping=team_name_mapping
                    )
                    
                    # Build context for this player
                    context = build_player_context(
                        player_name=player_name,
                        prop_type=prop_type,
                        line=line,
                        projection=projection,
                        opponent=opponent,
                        week=week_number,
                        team_validation=team_validation,
                        schedule_strength_data=schedule_strength_data,
                        monte_carlo_data=monte_carlo_data
                    )
                    
                    # Add additional prop data
                    context.update({
                        'odds': odds,
                        'event_id': prop['event_id'],
                        'home_team': prop['home_team'],
                        'away_team': prop['away_team']
                    })
                    
                    player_contexts.append(context)
    
    print(f"Built context for {len(player_contexts)} player props")
    
    # Add live search context for top candidates (with caching to avoid duplicates)
    if use_live_search:
        print("Gathering live search context for top candidates...")
        top_candidates = [ctx for ctx in player_contexts if ctx['edge_percentage'] >= MIN_EDGE_PERCENTAGE]
        top_candidates = sorted(top_candidates, key=lambda x: x['edge_percentage'], reverse=True)[:25]
        
        # Cache for live search results to avoid duplicate requests
        live_search_cache = {}
        unique_players = set()
        
        # First pass: identify unique players
        for context in top_candidates:
            unique_players.add(context['player'])
        
        print(f"Found {len(unique_players)} unique players out of {len(top_candidates)} candidates")
        
        # Second pass: get live search for unique players only
        for i, player in enumerate(unique_players, 1):
            print(f"Getting live search for {player} ({i}/{len(unique_players)})...")
            live_search = get_live_search_context(
                player, 
                week_number  # We don't need opponent for caching
            )
            
            if 'error' not in live_search:
                live_search_cache[player] = live_search.get('live_search_results', '')
            else:
                print(f"Live search failed for {player}: {live_search['error']}")
                live_search_cache[player] = ''
        
        # Third pass: assign cached results to all contexts
        for context in top_candidates:
            context['live_search_results'] = live_search_cache.get(context['player'], '')
    else:
        print("Live search disabled - using static data only")
        # Add empty live search results to maintain consistent data structure
        for context in player_contexts:
            context['live_search_results'] = ''
    
    # Generate AI prompt with live search context
    prompt = generate_ai_analysis_prompt(player_contexts)
    
    # Call Grok API with live search toggle
    if use_live_search:
        print("Calling Grok AI for analysis with live search enabled...")
        ai_response = call_grok_api(prompt, include_live_search=True)
    else:
        print("Calling Grok AI for analysis with live search disabled...")
        ai_response = call_grok_api(prompt, include_live_search=False)
    
    if 'error' in ai_response:
        print(f"AI analysis failed: {ai_response['error']}")
        return {'error': ai_response['error']}
    
    # Save insights
    save_ai_insights(week_number, ai_response, player_contexts)
    
    print("AI analysis completed successfully!")
    return {
        'success': True,
        'analysis': ai_response['analysis'],
        'player_contexts': player_contexts,
        'usage': ai_response.get('usage', {})
    }


def get_available_weeks():
    """
    Get list of available weeks with odds data.
    
    Returns:
        list: Available week numbers
    """
    odds_dir = "data/odds"
    if not os.path.exists(odds_dir):
        return []
    
    available_weeks = []
    for item in os.listdir(odds_dir):
        if item.startswith("week_") and os.path.isdir(os.path.join(odds_dir, item)):
            try:
                week_num = int(item.split("_")[1])
                available_weeks.append(week_num)
            except ValueError:
                continue
    
    return sorted(available_weeks)


def main(week_number=1, use_ai=True, use_live_search=False):
    """
    Main execution function for weekly betting analysis.
    
    Args:
        week_number (int): NFL week number to analyze
        use_ai (bool): Whether to use AI analysis (default: True)
        use_live_search (bool): Whether to enable live search functionality (default: False)
    
    Returns:
        dict: Analysis results including AI insights if enabled
    """
    print(f"=== NFL Betting Analysis - Week {week_number} ===")
    print(f"AI Analysis: {'Enabled' if use_ai else 'Disabled'}")
    print(f"Live Search: {'Enabled' if use_live_search else 'Disabled'}")
    print()
    
    if use_ai:
        # Use AI-powered analysis
        ai_results = analyze_with_ai(week_number, use_live_search=use_live_search)
        
        if 'error' in ai_results:
            print(f"AI analysis failed: {ai_results['error']}")
            print("Falling back to traditional analysis...")
            use_ai = False
        else:
            print("\n=== AI ANALYSIS RESULTS ===")
            print(ai_results['analysis'])
            print(f"\nAnalysis completed using {ai_results.get('usage', {}).get('total_tokens', 'unknown')} tokens")
            return ai_results
    else:
        print("AI Failed.")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    week_number = 1
    use_live_search = False
    
    if len(sys.argv) > 1:
        try:
            week_number = int(sys.argv[1])
        except ValueError:
            print(f"Invalid week number: {sys.argv[1]}")
            print("Usage: python picks_agent.py [week_number] [--live-search]")
            print("Example: python picks_agent.py 5")
            print("Example: python picks_agent.py 5 --live-search")
            sys.exit(1)
    
    # Check for live search flag
    if '--live-search' in sys.argv:
        use_live_search = True
    
    # Show available weeks
    available_weeks = get_available_weeks()
    if available_weeks:
        print(f"Available weeks: {available_weeks}")
        if week_number not in available_weeks:
            print(f"Warning: Week {week_number} not found in available weeks")
            print(f"Using most recent week: {available_weeks[-1]}")
            week_number = available_weeks[-1]
    
    print(f"Running AI analysis for Week {week_number}")
    main(week_number, use_ai=True, use_live_search=use_live_search)
