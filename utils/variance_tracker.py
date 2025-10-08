"""
Variance Tracking Module for Monte Carlo Projections

This module handles continuous variance updates for player performance statistics,
enabling uncertainty quantification in fantasy football projections.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

def calculate_historical_variance(df_game_data, position_col='pos'):
    """
    Calculate position-specific variance from historical data.
    
    Args:
        df_game_data: DataFrame with player game data
        position_col: Column name containing player positions
    
    Returns:
        dict: Position-specific variance data for each stat category
    """
    variance_data = {}
    
    # Define position groups for variance calculation
    position_groups = {
        'QB': ['QB'],
        'RB': ['RB', 'FB'],
        'WR': ['WR'],
        'TE': ['TE']
    }
    
    for position_group, positions in position_groups.items():
        # Filter data for this position group
        pos_data = df_game_data[df_game_data[position_col].isin(positions)]
        
        if len(pos_data) > 0:
            variance_data[position_group] = {
                # Attempts and targets
                'pass_att_std': pos_data['pass_att'].std() if 'pass_att' in pos_data.columns else 0,
                'rush_att_std': pos_data['rush_att'].std() if 'rush_att' in pos_data.columns else 0,
                'targets_std': pos_data['targets'].std() if 'targets' in pos_data.columns else 0,
                'rec_std': pos_data['receptions'].std() if 'receptions' in pos_data.columns else 0,
                
                # Yards
                'pass_yds_std': pos_data['pass_yds'].std() if 'pass_yds' in pos_data.columns else 0,
                'rush_yds_std': pos_data['rush_yds'].std() if 'rush_yds' in pos_data.columns else 0,
                'rec_yds_std': pos_data['rec_yds'].std() if 'rec_yds' in pos_data.columns else 0,
                
                # Touchdowns
                'pass_td_std': pos_data['pass_tds'].std() if 'pass_tds' in pos_data.columns else 0,
                'rush_td_std': pos_data['rush_tds'].std() if 'rush_tds' in pos_data.columns else 0,
                'rec_td_std': pos_data['rec_tds'].std() if 'rec_tds' in pos_data.columns else 0,
                
                # Negative stats
                'int_std': pos_data['pass_int'].std() if 'pass_int' in pos_data.columns else 0,
                'fum_std': pos_data['fumbles'].std() if 'fumbles' in pos_data.columns else 0,
                
                'sample_size': len(pos_data)
            }
            
            # Handle missing columns gracefully
            all_stats = [
                'pass_att_std', 'rush_att_std', 'targets_std', 'rec_std',
                'pass_yds_std', 'rush_yds_std', 'rec_yds_std', 
                'pass_td_std', 'rush_td_std', 'rec_td_std',
                'int_std', 'fum_std'
            ]
            for stat in all_stats:
                if pd.isna(variance_data[position_group][stat]):
                    variance_data[position_group][stat] = 0
    
    return variance_data

def update_variance_cache(variance_data, cache_file="data/variance_cache.json"):
    """
    Update variance cache with new data.
    
    Args:
        variance_data: Dictionary containing variance data by position
        cache_file: Path to variance cache file
    
    Returns:
        dict: Updated cache with new variance data
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    # Load existing cache or create new
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            cache = {}
    else:
        cache = {}
    
    # Update with new variance data
    timestamp = datetime.now().isoformat()
    cache[timestamp] = variance_data
    
    # Keep only last 10 updates to prevent file bloat
    if len(cache) > 10:
        oldest_key = min(cache.keys())
        del cache[oldest_key]
    
    # Save updated cache
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)
    
    print(f"Variance cache updated with {len(variance_data)} position groups")
    return cache

def get_latest_variance_data(cache_file="data/variance_cache.json"):
    """
    Get most recent variance data from cache.
    
    Args:
        cache_file: Path to variance cache file
    
    Returns:
        dict: Most recent variance data or None if cache is empty
    """
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        
        if not cache:
            return None
        
        # Return most recent variance data
        latest_timestamp = max(cache.keys())
        return cache[latest_timestamp]
    except (json.JSONDecodeError, FileNotFoundError):
        return None

def validate_variance_data(variance_data, min_sample_size=10):
    """
    Validate variance data quality.
    
    Args:
        variance_data: Dictionary containing variance data by position
        min_sample_size: Minimum sample size required for valid variance
    
    Returns:
        dict: Validation results for each position
    """
    validation_results = {}
    
    for position, stats in variance_data.items():
        validation_results[position] = {
            'sample_size': stats.get('sample_size', 0),
            'is_valid': stats.get('sample_size', 0) >= min_sample_size,
            'missing_stats': []
        }
        
        # Check for missing or invalid variance data
        required_stats = ['pass_yds_std', 'rush_yds_std', 'rec_yds_std']
        optional_stats = [
            'pass_att_std', 'rush_att_std', 'targets_std', 'rec_std',
            'pass_td_std', 'rush_td_std', 'rec_td_std', 'int_std', 'fum_std'
        ]
        
        for stat in required_stats:
            if stat not in stats or pd.isna(stats[stat]) or stats[stat] <= 0:
                validation_results[position]['missing_stats'].append(stat)
        
        # Check optional stats (don't fail validation if missing)
        for stat in optional_stats:
            if stat not in stats or pd.isna(stats[stat]):
                validation_results[position]['missing_stats'].append(stat)
    
    return validation_results

def update_variance_continuously(df_game_data, cache_file="data/variance_cache.json"):
    """
    Update variance data continuously as new data becomes available.
    
    Args:
        df_game_data: DataFrame with latest game data
        cache_file: Path to variance cache file
    
    Returns:
        dict: Updated variance data
    """
    print("Updating variance data from latest game data...")
    
    # Calculate new variance data
    new_variance_data = calculate_historical_variance(df_game_data)
    
    # Update cache
    updated_cache = update_variance_cache(new_variance_data, cache_file)
    
    # Validate the new variance data
    validation_results = validate_variance_data(new_variance_data)
    
    # Log validation results
    for position, results in validation_results.items():
        if results['is_valid']:
            print(f"  {position}: Valid variance data (n={results['sample_size']})")
        else:
            print(f"  {position}: Insufficient data (n={results['sample_size']})")
            if results['missing_stats']:
                print(f"    Missing stats: {results['missing_stats']}")
    
    print(f"Variance data updated. Cache now contains {len(updated_cache)} updates.")
    
    return new_variance_data

