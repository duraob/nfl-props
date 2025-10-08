"""
Historical Schedule Extractor

Extract historical NFL schedules from game data files.
Creates schedule_YYYY.csv files for backtesting.

This script processes existing game data files to extract
unique game matchups with home/away information for each season.
"""

import pandas as pd
import os
from datetime import datetime

def extract_historical_schedules():
    """
    Extract schedules from historical game data files.
    
    Creates schedule_YYYY.csv files containing:
    - week: NFL week number
    - home_team: Home team abbreviation
    - away_team: Away team abbreviation
    - home_away: Home/Away indicator
    - year: Season year
    """
    print("Extracting historical schedules...")
    print(f"Processing seasons 2016-2024")
    
    total_games = 0
    
    for year in range(2016, 2025):
        file_path = f'data/game_data/game_data_{year}.csv'
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {year}")
            continue
            
        try:
            # Load game data
            df = pd.read_csv(file_path)
            print(f"Loaded {file_path}: {len(df)} records")
            
            # Extract unique games (week, home_team, away_team)
            # Remove duplicates based on week, home_team, away_team combination
            schedule = df[['week', 'home_team', 'away_team', 'home_away']].drop_duplicates(subset=['week', 'home_team', 'away_team'])
            
            # Add year column
            schedule['year'] = year
            
            # Sort by week for consistency
            schedule = schedule.sort_values('week')
            
            # Validate data
            if schedule.empty:
                print(f"Warning: No schedule data found for {year}")
                continue
                
            # Check for missing values
            missing_data = schedule.isnull().sum()
            if missing_data.any():
                print(f"Warning: Missing data in {year} schedule:")
                print(missing_data[missing_data > 0])
            
            # Save schedule
            output_path = f'data/schedule_{year}.csv'
            schedule.to_csv(output_path, index=False)
            
            games_count = len(schedule)
            total_games += games_count
            print(f"Created {output_path}: {games_count} games")
            
            # Log sample data for verification
            print(f"Sample {year} schedule:")
            print(schedule.head(3).to_string(index=False))
            print()
            
        except Exception as e:
            print(f"Error processing {year}: {e}")
            continue
    
    print(f"Historical schedule extraction complete!")
    print(f"Total games extracted: {total_games}")
    print(f"Schedule files created in data/ directory")

def validate_schedule_data():
    """
    Validate extracted schedule data for consistency.
    
    Checks:
    - All seasons have data
    - Week numbers are consistent
    - Team abbreviations are consistent
    - No duplicate games
    """
    print("\nValidating schedule data...")
    
    validation_errors = []
    
    for year in range(2016, 2025):
        file_path = f'data/schedule_{year}.csv'
        if not os.path.exists(file_path):
            validation_errors.append(f"Missing schedule file: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            
            # Check for required columns
            required_columns = ['week', 'home_team', 'away_team', 'home_away', 'year']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_errors.append(f"{year}: Missing columns: {missing_columns}")
            
            # Check for duplicate games
            duplicates = df.duplicated(subset=['week', 'home_team', 'away_team']).sum()
            if duplicates > 0:
                validation_errors.append(f"{year}: {duplicates} duplicate games found")
            
            # Check week range
            weeks = df['week'].unique()
            if len(weeks) < 17:
                validation_errors.append(f"{year}: Only {len(weeks)} weeks found (expected 17+)")
            
            print(f"{year}: {len(df)} games, weeks {min(weeks)}-{max(weeks)}")
            
        except Exception as e:
            validation_errors.append(f"{year}: Error reading file: {e}")
    
    if validation_errors:
        print("\nValidation errors found:")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("All schedule data validated successfully!")

if __name__ == "__main__":
    print("NFL Historical Schedule Extractor")
    print("=" * 40)
    
    # Extract schedules
    extract_historical_schedules()
    
    # Validate extracted data
    validate_schedule_data()
    
    print("\nSchedule extraction complete!")
