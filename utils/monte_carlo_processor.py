"""
Monte Carlo Data Processor

Extracts P25/P75 values from Monte Carlo simulation results
and integrates them into projection files for enhanced analysis.
"""

import json
import pandas as pd
import os
from typing import Dict, List, Optional


def load_monte_carlo_data(week: int, projections_dir: str = "data/projections") -> Dict:
    """
    Load Monte Carlo simulation results for a specific week.
    
    Args:
        week: Week number
        projections_dir: Directory containing projection files
        
    Returns:
        Dictionary with Monte Carlo results
    """
    monte_carlo_file = os.path.join(projections_dir, f"monte_carlo_week{week}.json")
    
    if not os.path.exists(monte_carlo_file):
        print(f"Monte Carlo file not found: {monte_carlo_file}")
        return {}
    
    try:
        with open(monte_carlo_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading Monte Carlo data: {e}")
        return {}


def extract_p25_p75_values(monte_carlo_data: Dict, player_name: str, stat: str) -> tuple:
    """
    Extract P25 and P75 values for a specific player and statistic.
    
    Args:
        monte_carlo_data: Monte Carlo simulation results
        player_name: Player name
        stat: Statistic name (e.g., 'pass_yd', 'rush_yd')
        
    Returns:
        Tuple of (p25, p75) values, or (None, None) if not found
    """
    if player_name not in monte_carlo_data:
        return None, None
    
    player_data = monte_carlo_data[player_name]
    if stat not in player_data:
        return None, None
    
    stat_data = player_data[stat]
    if 'percentiles' not in stat_data:
        return None, None
    
    percentiles = stat_data['percentiles']
    p25 = percentiles.get('p25')
    p75 = percentiles.get('p75')
    
    return p25, p75


def enhance_projections_with_monte_carlo(projections_file: str, monte_carlo_data: Dict) -> pd.DataFrame:
    """
    Enhance projection file with P25/P75 values from Monte Carlo simulations.
    
    Args:
        projections_file: Path to projection CSV file
        monte_carlo_data: Monte Carlo simulation results
        
    Returns:
        Enhanced DataFrame with P25/P75 columns
    """
    # Load base projections
    df_projections = pd.read_csv(projections_file, index_col='name')
    
    # Define stat mappings for Monte Carlo data
    stat_mappings = {
        'pass_yd': 'pass_yd',
        'rush_yd': 'rush_yd', 
        'rec_yd': 'rec_yd',
        'pass_td': 'pass_td',
        'rush_td': 'rush_td',
        'rec_td': 'rec_td'
    }
    
    # Add P25/P75 columns for each stat
    for projection_col, monte_carlo_stat in stat_mappings.items():
        if projection_col in df_projections.columns:
            p25_values = []
            p75_values = []
            
            for player_name in df_projections.index:
                p25, p75 = extract_p25_p75_values(monte_carlo_data, player_name, monte_carlo_stat)
                p25_values.append(p25 if p25 is not None else df_projections.loc[player_name, projection_col])
                p75_values.append(p75 if p75 is not None else df_projections.loc[player_name, projection_col])
            
            df_projections[f'{projection_col}_p25'] = p25_values
            df_projections[f'{projection_col}_p75'] = p75_values
    
    return df_projections


def process_week_projections(week: int, projections_dir: str = "data/projections") -> str:
    """
    Process projections for a specific week, adding Monte Carlo P25/P75 values.
    
    Args:
        week: Week number
        projections_dir: Directory containing projection files
        
    Returns:
        Path to enhanced projection file
    """
    # Load Monte Carlo data
    monte_carlo_data = load_monte_carlo_data(week, projections_dir)
    
    if not monte_carlo_data:
        print(f"No Monte Carlo data available for Week {week}")
        return None
    
    # Load and enhance projections
    projections_file = os.path.join(projections_dir, f"nfl25_proj_week{week}.csv")
    
    if not os.path.exists(projections_file):
        print(f"Projections file not found: {projections_file}")
        return None
    
    # Enhance with Monte Carlo data
    df_enhanced = enhance_projections_with_monte_carlo(projections_file, monte_carlo_data)
    
    # Save enhanced projections
    enhanced_file = os.path.join(projections_dir, f"nfl25_proj_enhanced_week{week}.csv")
    df_enhanced.to_csv(enhanced_file)
    
    print(f"Enhanced projections saved to {enhanced_file}")
    return enhanced_file


if __name__ == "__main__":
    # Test the processor
    enhanced_file = process_week_projections(1)
    if enhanced_file:
        print(f"✅ Enhanced projections created: {enhanced_file}")
    else:
        print("❌ Failed to create enhanced projections")
