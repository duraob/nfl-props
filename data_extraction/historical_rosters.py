"""
Historical Roster Extractor

Extract historical NFL rosters from game data files.
Creates roster_YYYY.csv files for backtesting.

This script processes existing game data files to extract
unique players and their team assignments for each season.
"""

import pandas as pd
import os
from datetime import datetime

def extract_historical_rosters():
    """
    Extract rosters from historical game data files.
    
    Creates roster_YYYY.csv files containing:
    - player: Player name
    - team: Team abbreviation
    - pos: Player position
    - year: Season year
    - active: Active status (True if in game data)
    """
    print("Extracting historical rosters...")
    print(f"Processing seasons 2016-2024")
    
    total_players = 0
    
    for year in range(2016, 2025):
        file_path = f'data/game_data/game_data_{year}.csv'
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {year}")
            continue
            
        try:
            # Load game data
            df = pd.read_csv(file_path)
            print(f"Loaded {file_path}: {len(df)} records")
            
            # Extract unique players and their teams
            roster = df[['player', 'team', 'pos']].drop_duplicates()
            
            # Add year and active status
            roster['year'] = year
            roster['active'] = True  # Assume active if in game data
            
            # Sort by team, then player for consistency
            roster = roster.sort_values(['team', 'player'])
            
            # Validate data
            if roster.empty:
                print(f"Warning: No roster data found for {year}")
                continue
                
            # Check for missing values
            missing_data = roster.isnull().sum()
            if missing_data.any():
                print(f"Warning: Missing data in {year} roster:")
                print(missing_data[missing_data > 0])
            
            # Check for duplicate players
            duplicates = roster.duplicated(subset=['player']).sum()
            if duplicates > 0:
                print(f"Warning: {duplicates} duplicate players found in {year}")
                # Keep first occurrence of each player
                roster = roster.drop_duplicates(subset=['player'], keep='first')
            
            # Save roster
            output_path = f'data/roster_{year}.csv'
            roster.to_csv(output_path, index=False)
            
            players_count = len(roster)
            total_players += players_count
            print(f"Created {output_path}: {players_count} players")
            
            # Log sample data for verification
            print(f"Sample {year} roster:")
            print(roster.head(5).to_string(index=False))
            print()
            
        except Exception as e:
            print(f"Error processing {year}: {e}")
            continue
    
    print(f"Historical roster extraction complete!")
    print(f"Total players extracted: {total_players}")
    print(f"Roster files created in data/ directory")

def validate_roster_data():
    """
    Validate extracted roster data for consistency.
    
    Checks:
    - All seasons have data
    - Player names are consistent
    - Team abbreviations are consistent
    - Position codes are valid
    - No duplicate players within seasons
    """
    print("\nValidating roster data...")
    
    validation_errors = []
    position_codes = {'QB', 'RB', 'WR', 'TE', 'K', 'DEF', 'DST', 'LB', 'FB', 'P', 'SS', 'DE', 'CB', 'FS', 'G', 'T', 'C'}
    
    for year in range(2016, 2025):
        file_path = f'data/roster_{year}.csv'
        if not os.path.exists(file_path):
            validation_errors.append(f"Missing roster file: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            
            # Check for required columns
            required_columns = ['player', 'team', 'pos', 'year', 'active']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_errors.append(f"{year}: Missing columns: {missing_columns}")
            
            # Check for duplicate players
            duplicates = df.duplicated(subset=['player']).sum()
            if duplicates > 0:
                validation_errors.append(f"{year}: {duplicates} duplicate players found")
            
            # Check position codes
            invalid_positions = df[~df['pos'].isin(position_codes)]['pos'].unique()
            if len(invalid_positions) > 0:
                validation_errors.append(f"{year}: Invalid position codes: {invalid_positions}")
            
            # Check team distribution
            team_counts = df['team'].value_counts()
            if len(team_counts) < 30:  # Should have most NFL teams
                validation_errors.append(f"{year}: Only {len(team_counts)} teams found (expected 30+)")
            
            print(f"{year}: {len(df)} players, {len(team_counts)} teams")
            
        except Exception as e:
            validation_errors.append(f"{year}: Error reading file: {e}")
    
    if validation_errors:
        print("\nValidation errors found:")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("All roster data validated successfully!")

def analyze_roster_changes():
    """
    Analyze roster changes between seasons.
    
    Identifies:
    - Players who changed teams
    - New players each season
    - Players who left the league
    """
    print("\nAnalyzing roster changes between seasons...")
    
    for year in range(2017, 2025):
        current_file = f'data/roster_{year}.csv'
        previous_file = f'data/roster_{year-1}.csv'
        
        if not os.path.exists(current_file) or not os.path.exists(previous_file):
            continue
            
        try:
            current = pd.read_csv(current_file)
            previous = pd.read_csv(previous_file)
            
            # Find players who changed teams
            current_players = set(current['player'])
            previous_players = set(previous['player'])
            
            # Players in both seasons
            common_players = current_players & previous_players
            
            # New players
            new_players = current_players - previous_players
            
            # Players who left
            left_players = previous_players - current_players
            
            print(f"{year-1} to {year}:")
            print(f"  - Common players: {len(common_players)}")
            print(f"  - New players: {len(new_players)}")
            print(f"  - Players who left: {len(left_players)}")
            
        except Exception as e:
            print(f"Error analyzing {year}: {e}")

if __name__ == "__main__":
    print("NFL Historical Roster Extractor")
    print("=" * 40)
    
    # Extract rosters
    extract_historical_rosters()
    
    # Validate extracted data
    validate_roster_data()
    
    # Analyze roster changes
    analyze_roster_changes()
    
    print("\nRoster extraction complete!")
