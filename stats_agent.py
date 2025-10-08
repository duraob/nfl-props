"""
NFL Prop Nugget Module

This module analyzes NFL historical stats against current prop lines to generate
statistical insights and betting nuggets. It filters props to only include players
with historical game data and generates structured insights for LLM analysis.
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import user, system

# Load environment variables
load_dotenv()
print("Loaded environment variables from .env file")


# Configuration constants
MIN_SAMPLE_SIZE = 3
MIN_DELTA_PERCENTAGE = 15.0
MAX_NUGGETS = 20

# GROK API Configuration
GROK_API_KEY = os.getenv('GROK_API_KEY')
GROK_MODEL = "grok-3"  # Using grok-3 for consistency with picks_agent


def load_data(week_number: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and normalize all required data sources.
    
    Args:
        week_number (int): NFL week number to load data for
    
    Returns:
        tuple: (historical_stats, schedule, player_props)
    """
    print(f"Loading data sources for Week {week_number}...")
    
    # Load historical game data (all seasons)
    historical_stats = []
    data_dir = "data"
    game_data_dir = os.path.join(data_dir, "game_data")
    
    for file in os.listdir(game_data_dir):
        if file.startswith("game_data_") and file.endswith(".csv"):
            file_path = os.path.join(game_data_dir, file)
            try:
                df = pd.read_csv(file_path)
                historical_stats.append(df)
                print(f"Loaded {file}: {len(df)} records")
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
    
    if not historical_stats:
        raise FileNotFoundError("No historical game data files found")
    
    # Combine all historical data
    historical_stats = pd.concat(historical_stats, ignore_index=True)
    
    # Normalize column names
    historical_stats.columns = historical_stats.columns.str.lower().str.replace(' ', '_')
    
    # Load schedule
    schedule_file = os.path.join(data_dir, "nfl-2025-EasternStandardTime.csv")
    if not os.path.exists(schedule_file):
        raise FileNotFoundError(f"Schedule file not found: {schedule_file}")
    
    schedule = pd.read_csv(schedule_file)
    schedule.columns = schedule.columns.str.lower().str.replace(' ', '_')
    
    # Load player props for specified week
    props_file = os.path.join(data_dir, "odds", f"week_{week_number:02d}", f"player_props_week_{week_number:02d}.csv")
    if not os.path.exists(props_file):
        raise FileNotFoundError(f"Player props file not found: {props_file}")
    
    player_props = pd.read_csv(props_file)
    player_props.columns = player_props.columns.str.lower().str.replace(' ', '_')
    
    print(f"Data loaded: {len(historical_stats)} historical records, {len(schedule)} games, {len(player_props)} props")
    
    return historical_stats, schedule, player_props


def create_player_mapping() -> Dict[str, str]:
    """
    Create mapping between odds player names and historical data player names.
    
    Returns:
        dict: Mapping of odds names to historical data names
    """
    mapping_file = 'data/player_name_mapping.csv'
    if os.path.exists(mapping_file):
        try:
            mapping_df = pd.read_csv(mapping_file)
            mapping = dict(zip(mapping_df['odds_name'], mapping_df['projection_name']))
            print(f"Loaded {len(mapping)} player mappings")
            return mapping
        except Exception as e:
            print(f"Warning: Could not load mapping file: {e}")
    
    # Fallback to basic mapping
    print("Using fallback player mapping")
    return {}


def filter_props_with_player_data(player_props: pd.DataFrame, 
                                historical_stats: pd.DataFrame,
                                player_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Filter props to only include players with historical game data.
    
    Args:
        player_props: Player props dataframe
        historical_stats: Historical game stats dataframe
        player_mapping: Player name mapping dictionary
    
    Returns:
        DataFrame: Filtered props with only players having historical data
    """
    print("Filtering props to players with historical data...")
    
    # Get unique players from historical data
    historical_players = set(historical_stats['player'].unique())
    
    # Filter props to only include players with historical data
    filtered_props = []
    
    for _, prop in player_props.iterrows():
        player_name = prop['player_name']
        
        # Try direct match first
        if player_name in historical_players:
            filtered_props.append(prop)
            continue
        
        # Try mapped name
        if player_name in player_mapping:
            mapped_name = player_mapping[player_name]
            if mapped_name in historical_players:
                # Update the player name to match historical data
                prop_copy = prop.copy()
                prop_copy['player_name'] = mapped_name
                filtered_props.append(prop_copy)
                continue
    
    filtered_df = pd.DataFrame(filtered_props)
    print(f"Filtered from {len(player_props)} to {len(filtered_df)} props with historical data")
    
    return filtered_df


def get_stat_column(prop_type: str) -> Optional[str]:
    """
    Map prop type to corresponding historical stat column.
    
    Args:
        prop_type: Type of prop (rush_yds, reception_yds, etc.)
    
    Returns:
        str: Corresponding historical stat column name
    """
    prop_mapping = {
        'rush_yds': 'rush_yds',
        'reception_yds': 'rec_yds',
        'receptions': 'receptions',
        'pass_yds': 'pass_yds',
        'pass_attempts': 'pass_att',
        'pass_completions': 'pass_cmp',
        'pass_tds': 'pass_tds',
        'pass_interceptions': 'pass_int',
        'rush_att': 'rush_att',
        'anytime_td': 'rush_tds'  # Will combine with rec_tds
    }
    
    return prop_mapping.get(prop_type)


def extract_opponent_from_props(prop: pd.Series) -> str:
    """
    Extract opponent team from player props data.
    
    Args:
        prop: Single row from player props dataframe
    
    Returns:
        str: Opponent team name
    """
    # Get home and away teams from the prop data
    home_team = prop.get('home_team', '')
    away_team = prop.get('away_team', '')
    
    # For now, we'll use a simple approach - determine opponent based on team context
    # This could be enhanced with more sophisticated team matching logic
    if home_team and away_team:
        # Return the opponent team (this is a simplified approach)
        # In a real implementation, you'd need to determine which team the player is on
        return f"{away_team} vs {home_team}"
    
    return "Unknown"


def compute_trends(filtered_props: pd.DataFrame, 
                  historical_stats: pd.DataFrame,
                  week_number: int) -> List[Dict]:
    """
    Compute player trends for each prop.
    
    Args:
        filtered_props: Filtered player props
        historical_stats: Historical game stats
        week_number: Current week number for context
    
    Returns:
        list: List of trend dictionaries
    """
    print(f"Computing player trends for Week {week_number}...")
    
    trends = []
    
    for _, prop in filtered_props.iterrows():
        player_name = prop['player_name']
        prop_type = prop['prop_type']
        line = prop['point']
        
        # Get stat column for this prop type
        stat_col = get_stat_column(prop_type)
        if not stat_col:
            continue
        
        # Get player's historical data
        player_data = historical_stats[historical_stats['player'] == player_name]
        if player_data.empty:
            continue
        
        # Extract opponent from props data
        opponent = extract_opponent_from_props(prop)
        
        # Calculate career average for this stat
        if prop_type == 'anytime_td':
            # Combine rushing and receiving TDs
            total_tds = player_data['rush_tds'].fillna(0) + player_data['rec_tds'].fillna(0)
            career_avg = total_tds.mean()
        else:
            career_avg = player_data[stat_col].fillna(0).mean()
        
        # Calculate last 3 games vs opponent (if available)
        # For now, use career average as placeholder - could be enhanced with opponent-specific data
        last_3_vs_opponent = career_avg
        
        # Calculate last 5 overall games (form check)
        last_5_games = player_data.tail(5)[stat_col].fillna(0).mean()
        
        # Calculate sample size
        sample_size = len(player_data)
        
        if sample_size >= MIN_SAMPLE_SIZE:
            trend_data = {
                'player': player_name,
                'opponent': opponent,
                'stat': prop_type,
                'sample_size': sample_size,
                'career_avg': career_avg,
                'last_3_vs_opponent': last_3_vs_opponent,
                'last_5_games': last_5_games,
                'line': line,
                'stat_column': stat_col,
                'week': week_number
            }
            trends.append(trend_data)
    
    print(f"Computed trends for {len(trends)} player/prop combinations")
    return trends


def generate_nuggets(trends: List[Dict]) -> List[Dict]:
    """
    Generate raw nugget dictionaries from trends.
    
    Args:
        trends: List of trend dictionaries
    
    Returns:
        list: List of nugget dictionaries
    """
    print("Generating nuggets...")
    
    nuggets = []
    seen_combinations = set()  # Track unique player/stat combinations
    
    for trend in trends:
        # Create unique key for this player/stat combination
        unique_key = (trend['player'], trend['stat'])
        
        # Skip if we've already seen this combination
        if unique_key in seen_combinations:
            continue
        
        seen_combinations.add(unique_key)
        
        # Use career average for now (could be enhanced with opponent-specific data)
        average = trend['career_avg']
        line = trend['line']
        week_number = trend.get('week', 1)
        
        # Calculate delta and percentage
        delta = average - line
        delta_pct = (delta / line) * 100 if line != 0 else 0
        
        # Apply filters
        if abs(delta_pct) >= MIN_DELTA_PERCENTAGE:
            nugget = {
                'player': trend['player'],
                'opponent': trend['opponent'],
                'stat': trend['stat'],
                'sample_size': trend['sample_size'],
                'average': round(average, 1),
                'line': line,
                'delta_pct': round(delta_pct, 1),
                'week': week_number,
                'nugget': f"{trend['player']} averages {average:.1f} {trend['stat']} in {trend['sample_size']} games, {delta_pct:+.1f}% vs Week {week_number} line of {line}."
            }
            nuggets.append(nugget)
    
    print(f"Generated {len(nuggets)} unique nuggets meeting criteria")
    return nuggets


def filter_and_rank_nuggets(nuggets: List[Dict]) -> List[Dict]:
    """
    Apply final filters and rank nuggets by delta percentage.
    
    Args:
        nuggets: List of raw nuggets
    
    Returns:
        list: Filtered and ranked nuggets
    """
    print("Filtering and ranking nuggets...")
    
    # Filter by sample size and delta percentage
    filtered_nuggets = [
        nugget for nugget in nuggets 
        if nugget['sample_size'] >= MIN_SAMPLE_SIZE 
        and abs(nugget['delta_pct']) >= MIN_DELTA_PERCENTAGE
    ]
    
    # Rank by absolute delta percentage (descending)
    ranked_nuggets = sorted(
        filtered_nuggets, 
        key=lambda x: abs(x['delta_pct']), 
        reverse=True
    )
    
    # Keep top nuggets
    final_nuggets = ranked_nuggets[:MAX_NUGGETS]
    
    print(f"Final result: {len(final_nuggets)} nuggets")
    return final_nuggets


def save_nuggets_to_json(nuggets: List[Dict], week_number: int = 1) -> str:
    """
    Save nuggets to JSON file.
    
    Args:
        nuggets: List of nugget dictionaries
        week_number: NFL week number
    
    Returns:
        str: Path to saved JSON file
    """
    # Create nuggets directory
    nuggets_dir = "data/nuggets"
    os.makedirs(nuggets_dir, exist_ok=True)
    
    filename = f"nuggets_week_{week_number:02d}.json"
    filepath = os.path.join(nuggets_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(nuggets, f, indent=2)
    
    print(f"Saved {len(nuggets)} nuggets to {filepath}")
    return filepath


def send_to_llm(json_file: str, prompt: str) -> str:
    """
    Send JSON data to LLM with prompt for analysis.
    
    Args:
        json_file: Path to JSON file with nuggets
        prompt: Prompt for LLM analysis
    
    Returns:
        str: LLM response (placeholder for now)
    """
    print("Preparing data for LLM analysis...")
    
    # Read the JSON file
    with open(json_file, 'r') as f:
        nuggets_data = json.load(f)
    
    # Create the fixed prompt as specified in the execution document
    llm_prompt = f"""You are an ESPN sports analytics writer. 
You are given structured NFL trend nuggets in JSON. 
Rewrite each nugget as a short, headline-style betting insight. 
- Keep it one sentence each. 
- Make it punchy and fan-friendly. 
- Always connect to the posted prop line. 
- Example: "Derrick Henry has punished the Ravens, averaging 122 rushing yards in 5 career games — well above his Week 1 line of 89.5."

Here are the nuggets to analyze:
{json.dumps(nuggets_data, indent=2)}

Return your analysis as a list of ESPN-style insights."""

    # For now, just return the prompt (in real implementation, this would call an LLM API)
    print("LLM prompt prepared:")
    print(llm_prompt)
    
    return llm_prompt


def call_grok_api(nuggets_data: List[Dict]) -> Dict:
    """
    Call GROK API using X.AI SDK to analyze nuggets and generate ESPN-style insights.
    
    Args:
        nuggets_data: List of nugget dictionaries
    
    Returns:
        dict: GROK API response with ESPN-style insights
    """
    if not GROK_API_KEY:
        print("❌ GROK_API_KEY not found in environment variables")
        print("Please add GROK_API_KEY to your .env file")
        return {"error": "GROK_API_KEY not configured"}
    
    print("🤖 Calling GROK API for ESPN-style analysis...")
    
    try:
        # Initialize X.AI client
        client = Client(api_key=GROK_API_KEY)
        
        # Create a conversation
        chat = client.chat.create(model=GROK_MODEL)
        
        # Get week number from the first nugget for context
        week_number = nuggets_data[0].get('week', 1) if nuggets_data else 1
        
        # Prepare the prompt for GROK
        grok_prompt = f"""You are an ESPN sports analytics writer. 
You are given structured NFL trend nuggets in JSON for Week {week_number}. 
Rewrite each nugget as a short, headline-style betting insight. 
- Keep it one sentence each. 
- Make it punchy and fan-friendly. 
- Always connect to the posted prop line. 
- Example: "Derrick Henry has punished the Ravens, averaging 122 rushing yards in 5 career games — well above his Week {week_number} line of 89.5."

Here are the nuggets to analyze:
{json.dumps(nuggets_data, indent=2)}

Return your analysis as a JSON list of ESPN-style insights."""

        # Add system message and user prompt
        chat.append(system("You are an ESPN sports analytics writer specializing in NFL betting insights."))
        chat.append(user(grok_prompt))
        
        print(f"📡 Sending request to GROK API...")
        response = chat.sample()
        
        print("✅ GROK API call successful")
        print("📝 GROK Response:")
        print(response.content)
        
        # Try to parse JSON response
        try:
            parsed_response = json.loads(response.content)
            return {
                "success": True,
                "insights": parsed_response,
                "raw_response": response.content
            }
        except json.JSONDecodeError:
            # If not JSON, return as text
            return {
                "success": True,
                "insights": [response.content],
                "raw_response": response.content
            }
            
    except Exception as e:
        print(f"❌ GROK API error: {e}")
        return {
            "success": False,
            "error": f"GROK API request failed: {str(e)}"
        }


def save_grok_insights(insights: Dict, week_number: int = 1) -> str:
    """
    Save GROK-generated insights to a file.
    
    Args:
        insights: GROK API response dictionary
        week_number: NFL week number
    
    Returns:
        str: Path to saved insights file
    """
    # Create fun_stats directory
    fun_stats_dir = "data/fun_stats"
    os.makedirs(fun_stats_dir, exist_ok=True)
    
    filename = f"stats_insights_week_{week_number:02d}.json"
    filepath = os.path.join(fun_stats_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(insights, f, indent=2)
    
    print(f"💾 Saved GROK insights to {filepath}")
    return filepath


def main(week_number: int = 1) -> Tuple[List[Dict], str]:
    """
    Main execution function for the NFL Prop Nugget Module.
    
    Args:
        week_number: NFL week number to analyze
    
    Returns:
        tuple: (nuggets, llm_prompt)
    """
    print(f"=== NFL Prop Nugget Module - Week {week_number} ===")
    
    try:
        # 1. Load and normalize data
        historical_stats, schedule, player_props = load_data(week_number)
        
        # 2. Create player mapping
        player_mapping = create_player_mapping()
        
        # 3. Filter props to players with historical data
        filtered_props = filter_props_with_player_data(
            player_props, historical_stats, player_mapping
        )
        
        # 4. Compute player trends
        trends = compute_trends(filtered_props, historical_stats, week_number)
        
        # 5. Generate nuggets
        raw_nuggets = generate_nuggets(trends)
        
        # 6. Filter and rank nuggets
        final_nuggets = filter_and_rank_nuggets(raw_nuggets)
        
        # 7. Save nuggets to JSON
        json_file = save_nuggets_to_json(final_nuggets, week_number)
        
        # 8. Prepare for LLM analysis
        llm_prompt = send_to_llm(json_file, "")
        
        # 9. Call GROK API for ESPN-style insights
        grok_response = call_grok_api(final_nuggets)
        
        # 10. Save GROK insights
        if grok_response.get("success"):
            insights_file = save_grok_insights(grok_response, week_number)
            print(f"🤖 GROK insights saved to: {insights_file}")
        else:
            print(f"⚠️ GROK API call failed: {grok_response.get('error', 'Unknown error')}")
        
        print(f"\n✅ Successfully processed Week {week_number}")
        print(f"📊 Generated {len(final_nuggets)} nuggets")
        print(f"💾 Saved to: {json_file}")
        print(f"🤖 LLM prompt prepared for analysis")
        print(f"🤖 GROK API integration: {'✅ Success' if grok_response.get('success') else '❌ Failed'}")
        
        return final_nuggets, llm_prompt
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    week_number = 1  # Default to Week 1
    if len(sys.argv) > 1:
        try:
            week_number = int(sys.argv[1])
        except ValueError:
            print(f"Invalid week number: {sys.argv[1]}")
            print("Usage: python stats_agent.py [week_number]")
            print("Example: python stats_agent.py 2")
            sys.exit(1)
    
    main(week_number)
