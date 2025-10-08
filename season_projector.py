"""
Season-Long NFL Projection System

Generates season-long projections by running the existing weekly projection engine
for all 18 weeks of the NFL season and accumulating the results.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime as dt
from projection_engine import ProjectionEngine

def load_player_positions(roster_file="data/roster.xlsx"):
    """
    Load player positions from the new roster format.
    
    Args:
        roster_file: Path to the roster Excel file (new format)
    
    Returns:
        dict: Player name to position mapping
    """
    try:
        df_roster = pd.read_excel(roster_file, sheet_name='Master_Roster')
        
        # Clean player names by removing common suffixes and extra whitespace
        def clean_player_name(name):
            if pd.isna(name):
                return None
            name = str(name).strip()
            suffixes_to_remove = ['(IR)', '(PUP)', '(NFI)', '(COVID)', '(SUSP)', '(RESERVE)', '(Out)', '(Questionable)', '(Doubtful)']
            for suffix in suffixes_to_remove:
                name = name.replace(suffix, '').strip()
            return ' '.join(name.split())
        
        # Create player to position mapping using new column names
        df_roster['clean_name'] = df_roster['player_name'].apply(clean_player_name)
        player_position_mapping = dict(zip(df_roster['clean_name'], df_roster['position']))
        
        print(f"Loaded {len(player_position_mapping)} player positions from roster")
        return player_position_mapping
    except Exception as e:
        print(f"Error loading player positions: {e}")
        return {}

def run_season_projections(week_2025_file="data/game_data/game_data_2025.csv", 
                          weeks_2024_file="data/game_data/game_data_2024.csv",
                          target_weeks_2024=[9, 10, 11, 12, 13, 14, 15, 16, 17],
                          schedule_file="data/nfl-2025-EasternStandardTime.csv",
                          roster_file="data/roster.xlsx"):
    """
    Run season-long projections by generating weekly projections for all 18 weeks
    and accumulating the results into season totals.
    
    Args:
        week_2025_file: Path to 2025 season data
        weeks_2024_file: Path to 2024 season data
        target_weeks_2024: List of 2024 weeks to include as fallback
        schedule_file: Path to the schedule CSV file
        roster_file: Path to the active roster Excel file
    """
    start = dt.now()
    print(f"\nSeason-Long Projection System Start - {start}:\n")
    
    # Create season projections directory
    season_dir = "data/season_projections"
    os.makedirs(season_dir, exist_ok=True)
    
    # Initialize season accumulation dataframes
    season_player_stats = None
    season_team_stats = None
    
    # Run projections for each week (1-18 for full season)
    for week in range(1, 19):  # Full season: weeks 1-18
        print(f"\n{'='*50}")
        print(f"Processing Week {week}")
        print(f"{'='*50}")
        
        try:
            # Initialize the new projection engine
            engine = ProjectionEngine(n_simulations=1000)  # Reduced for season-long efficiency
            
            # Run weekly projections using the new engine
            print(f"Running Week {week} projections with new engine...")
            results = engine.run_projections(
                week_2025_file=week_2025_file,
                weeks_2024_file=weeks_2024_file,
                target_weeks_2024=target_weeks_2024,
                projection_week=week,
                schedule_file=schedule_file,
                roster_file=roster_file
            )
            
            # Get the projections directly from results and save to season directory
            if results and 'base_projections' in results:
                df_weekly = results['base_projections'].copy()
                
                # Add week column
                df_weekly['week'] = week
                
                # Save directly to season directory (no copying needed)
                dest_file = os.path.join(season_dir, f"nfl25_proj_week{week}.csv")
                df_weekly.to_csv(dest_file, index=True)
                
                # Accumulate season stats
                if season_player_stats is None:
                    season_player_stats = df_weekly.copy()
                else:
                    season_player_stats = pd.concat([season_player_stats, df_weekly], ignore_index=True)
                
                print(f"Week {week}: {len(df_weekly)} players projected")
            else:
                print(f"Warning: No projections generated for Week {week}")
                
        except Exception as e:
            print(f"Error processing Week {week}: {e}")
            continue
    
    # Generate season-long summaries
    if season_player_stats is not None:
        generate_season_summaries(season_player_stats, season_dir)
    
    end = dt.now()
    print(f"\nSeason-Long Projection System End - {end}")
    print(f"Total Runtime: {end - start}\n")

def generate_season_summaries(season_data, output_dir):
    """
    Generate season-long summary statistics and projections.
    
    Args:
        season_data: DataFrame with all weekly projections
        output_dir: Directory to save season summaries
    """
    print("\nGenerating season-long summaries...")
    
    # Load player positions from master roster
    player_positions = load_player_positions()
    
    # Player season totals - updated for new projection engine format
    agg_dict = {
        'pass_att': 'sum',
        'rush_att': 'sum', 
        'tar': 'sum',
        'pass_yd': 'sum',
        'rush_yd': 'sum',
        'rec_yd': 'sum',
        'pass_td': 'sum',
        'rush_td': 'sum',
        'rec_td': 'sum',
        'pass_int': 'sum',
        'fum': 'sum',
        'team': 'first',
        'week': 'count'  # Games played
    }
    
    # Add Monte Carlo columns if they exist
    monte_carlo_columns = [
        # Attempts and targets
        'pass_att_mean', 'pass_att_std', 'pass_att_p25', 'pass_att_p75',
        'rush_att_mean', 'rush_att_std', 'rush_att_p25', 'rush_att_p75',
        'targets_mean', 'targets_std', 'targets_p25', 'targets_p75',
        'rec_mean', 'rec_std', 'rec_p25', 'rec_p75',
        
        # Yards
        'pass_yd_mean', 'pass_yd_std', 'pass_yd_p25', 'pass_yd_p75',
        'rush_yd_mean', 'rush_yd_std', 'rush_yd_p25', 'rush_yd_p75',
        'rec_yd_mean', 'rec_yd_std', 'rec_yd_p25', 'rec_yd_p75',
        
        # Touchdowns
        'pass_td_mean', 'pass_td_std', 'pass_td_p25', 'pass_td_p75',
        'rush_td_mean', 'rush_td_std', 'rush_td_p25', 'rush_td_p75',
        'rec_td_mean', 'rec_td_std', 'rec_td_p25', 'rec_td_p75',
        
        # Negative stats
        'int_mean', 'int_std', 'int_p25', 'int_p75',
        'fum_mean', 'fum_std', 'fum_p25', 'fum_p75'
    ]
    
    # Add Monte Carlo columns with proper aggregation
    for col in monte_carlo_columns:
        if col in season_data.columns:
            if col.endswith('_mean'):
                agg_dict[col] = 'sum'  # Sum means across weeks
            elif col.endswith('_std'):
                # For standard deviation, we need to calculate combined std
                # This is a simplified approach - in practice, you'd want proper variance combination
                agg_dict[col] = 'mean'  # Average std across weeks (approximation)
            elif col.endswith('_p25') or col.endswith('_p75'):
                agg_dict[col] = 'sum'  # Sum percentiles across weeks
            else:
                agg_dict[col] = 'sum'  # Default to sum
    
    player_season_totals = season_data.groupby('name').agg(agg_dict).rename(columns={'week': 'games_played'})
    
    # Add position data from master roster
    def get_player_position(player_name):
        """Get player position from master roster"""
        # Clean the player name to match roster format
        clean_name = ' '.join(str(player_name).split())
        return player_positions.get(clean_name, 'Unknown')
    
    player_season_totals['position'] = player_season_totals.index.map(get_player_position)
    
    # Move position column to the front for better visibility
    cols = list(player_season_totals.columns)
    cols.remove('position')
    cols.insert(1, 'position')  # Insert after 'name' (index)
    player_season_totals = player_season_totals[cols]
    
    # Calculate fantasy points (standard scoring) - updated for new format
    player_season_totals['fantasy_points'] = (
        player_season_totals['pass_yd'] * 0.04 +
        player_season_totals['rush_yd'] * 0.1 +
        player_season_totals['rec_yd'] * 0.1 +
        player_season_totals['pass_td'] * 4 +
        player_season_totals['rush_td'] * 6 +
        player_season_totals['rec_td'] * 6 +
        player_season_totals['pass_int'] * -2 +
        player_season_totals['fum'] * -2
    )
    
    # Calculate Monte Carlo enhanced fantasy points if available
    if 'pass_yd_mean' in player_season_totals.columns:
        # Use Monte Carlo means for all statistics
        player_season_totals['fantasy_points_mean'] = (
            player_season_totals['pass_yd_mean'] * 0.04 +
            player_season_totals['rush_yd_mean'] * 0.1 +
            player_season_totals['rec_yd_mean'] * 0.1 +
            player_season_totals['pass_td_mean'] * 4 +
            player_season_totals['rush_td_mean'] * 6 +
            player_season_totals['rec_td_mean'] * 6 +
            player_season_totals['int_mean'] * -2 +
            player_season_totals['fum_mean'] * -2
        )
        
        # Calculate fantasy points with uncertainty (std) - only from yards
        player_season_totals['fantasy_points_std'] = (
            player_season_totals['pass_yd_std'] * 0.04 +
            player_season_totals['rush_yd_std'] * 0.1 +
            player_season_totals['rec_yd_std'] * 0.1
        )
        
        # Calculate fantasy points confidence intervals using Monte Carlo data
        player_season_totals['fantasy_points_p25'] = (
            player_season_totals['pass_yd_p25'] * 0.04 +
            player_season_totals['rush_yd_p25'] * 0.1 +
            player_season_totals['rec_yd_p25'] * 0.1 +
            player_season_totals['pass_td_p25'] * 4 +
            player_season_totals['rush_td_p25'] * 6 +
            player_season_totals['rec_td_p25'] * 6 +
            player_season_totals['int_p25'] * -2 +
            player_season_totals['fum_p25'] * -2
        )
        
        player_season_totals['fantasy_points_p75'] = (
            player_season_totals['pass_yd_p75'] * 0.04 +
            player_season_totals['rush_yd_p75'] * 0.1 +
            player_season_totals['rec_yd_p75'] * 0.1 +
            player_season_totals['pass_td_p75'] * 4 +
            player_season_totals['rush_td_p75'] * 6 +
            player_season_totals['rec_td_p75'] * 6 +
            player_season_totals['int_p75'] * -2 +
            player_season_totals['fum_p75'] * -2
        )
    
    # Team season totals - updated for new format
    team_season_totals = season_data.groupby('team').agg({
        'pass_att': 'sum',
        'rush_att': 'sum',
        'tar': 'sum', 
        'pass_yd': 'sum',
        'rush_yd': 'sum',
        'rec_yd': 'sum',
        'pass_td': 'sum',
        'rush_td': 'sum',
        'rec_td': 'sum',
        'pass_int': 'sum',
        'fum': 'sum',
        'week': 'count'
    }).rename(columns={'week': 'total_games'})
    
    # Save season summaries
    player_season_totals.to_csv(os.path.join(output_dir, "player_season_totals.csv"))
    team_season_totals.to_csv(os.path.join(output_dir, "team_season_totals.csv"))
    
    # Generate top performers for display only (no CSV files)
    top_qbs = player_season_totals[player_season_totals['pass_att'] > 0].nlargest(10, 'fantasy_points')
    top_rbs = player_season_totals[player_season_totals['rush_att'] > 0].nlargest(10, 'fantasy_points')
    top_wrs = player_season_totals[player_season_totals['tar'] > 0].nlargest(10, 'fantasy_points')
    
    print(f"Season summaries saved to {output_dir}/")
    print(f"Total players projected: {len(player_season_totals)}")
    print(f"Total teams: {len(team_season_totals)}")
    
    # Print top performers
    print("\nTop 5 QBs by Fantasy Points:")
    if 'fantasy_points_mean' in top_qbs.columns:
        print(top_qbs[['team', 'fantasy_points', 'fantasy_points_mean', 'fantasy_points_std', 'pass_yd', 'pass_td']].head())
    else:
        print(top_qbs[['team', 'fantasy_points', 'pass_yd', 'pass_td']].head())
    
    print("\nTop 5 RBs by Fantasy Points:")
    if 'fantasy_points_mean' in top_rbs.columns:
        print(top_rbs[['team', 'fantasy_points', 'fantasy_points_mean', 'fantasy_points_std', 'rush_yd', 'rush_td']].head())
    else:
        print(top_rbs[['team', 'fantasy_points', 'rush_yd', 'rush_td']].head())
    
    print("\nTop 5 WRs by Fantasy Points:")
    if 'fantasy_points_mean' in top_wrs.columns:
        print(top_wrs[['team', 'fantasy_points', 'fantasy_points_mean', 'fantasy_points_std', 'rec_yd', 'rec_td']].head())
    else:
        print(top_wrs[['team', 'fantasy_points', 'rec_yd', 'rec_td']].head())

if __name__ == "__main__":
    run_season_projections()
