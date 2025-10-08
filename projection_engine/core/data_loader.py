"""
Data Loading Module for NFL Projection Engine

Handles loading and filtering of:
- Active roster (excluding injured players)
- Injury reports
- Game data with time weighting
- Player-team mappings
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Set, Tuple, Optional
from datetime import datetime


class DataLoader:
    """
    Centralized data loading and filtering for the projection engine.
    
    Key responsibilities:
    1. Load active roster (excluding injured players)
    2. Apply injury filtering
    3. Load and time-weight game data
    4. Create player-team mappings
    """
    
    def __init__(self, roster_file: str = "data/roster.xlsx", 
                 injuries_file: str = "data/injuries.csv"):
        """
        Initialize data loader with file paths.
        
        Args:
            roster_file: Path to roster Excel file (new format from roster_scraper.py)
            injuries_file: Path to injuries CSV file
        """
        self.roster_file = roster_file
        self.injuries_file = injuries_file
        self.active_roster: Set[str] = set()
        self.injured_players: Set[str] = set()
        self.player_team_mapping: Dict[str, str] = {}
        self.roster_data: pd.DataFrame = pd.DataFrame()
        
    def load_active_roster(self) -> Set[str]:
        """
        Load active roster excluding injured players from the new roster format.
        
        Returns:
            Set of active player names
        """
        print("Loading active roster and filtering injured players...")
        
        # Load roster from new format
        if not os.path.exists(self.roster_file):
            raise FileNotFoundError(f"Roster file not found: {self.roster_file}")
        
        # Load from Master_Roster sheet
        self.roster_data = pd.read_excel(self.roster_file, sheet_name='Master_Roster')
        
        # Clean player names
        def clean_player_name(name):
            if pd.isna(name):
                return None
            name = str(name).strip()
            # Remove common suffixes that might appear in the new format
            suffixes_to_remove = ['(IR)', '(PUP)', '(NFI)', '(COVID)', '(SUSP)', '(RESERVE)', '(Out)', '(Questionable)', '(Doubtful)']
            for suffix in suffixes_to_remove:
                name = name.replace(suffix, '').strip()
            return ' '.join(name.split())
        
        # Apply cleaning to player names
        self.roster_data['clean_name'] = self.roster_data['player_name'].apply(clean_player_name)
        
        # Filter out injured players based on the new format's injury status
        healthy_players = self.roster_data[
            (~self.roster_data['is_injured']) & 
            (self.roster_data['clean_name'].notna())
        ]
        
        # Get active players
        roster_players = set(healthy_players['clean_name'].dropna())
        
        # Also load from external injuries file if available
        self.injured_players = self._load_injured_players()
        
        # Filter out any additional injured players from external source
        self.active_roster = roster_players - self.injured_players
        
        print(f"Active roster: {len(self.active_roster)} players")
        print(f"Excluded injured: {len(self.injured_players)} players")
        
        return self.active_roster
    
    def _load_injured_players(self) -> Set[str]:
        """Load injured players from injuries file."""
        if not os.path.exists(self.injuries_file):
            print(f"Injuries file not found: {self.injuries_file}")
            return set()
        
        try:
            df_injuries = pd.read_csv(self.injuries_file)
            injured_players = df_injuries[
                df_injuries['status'].isin(['Out', 'Injured Reserve'])
            ]['name'].tolist()
            
            # Clean player names
            def clean_player_name(name):
                if pd.isna(name):
                    return None
                return ' '.join(str(name).strip().split())
            
            injured_players_clean = set()
            for player in injured_players:
                clean_name = clean_player_name(player)
                if clean_name:
                    injured_players_clean.add(clean_name)
            
            return injured_players_clean
            
        except Exception as e:
            print(f"Error loading injuries: {e}")
            return set()
    
    def create_player_team_mapping(self) -> Dict[str, str]:
        """
        Create mapping of players to their current teams from the new roster format.
        
        Returns:
            Dict mapping player names to team abbreviations
        """
        if self.roster_data.empty:
            # Load roster data if not already loaded
            if not os.path.exists(self.roster_file):
                raise FileNotFoundError(f"Roster file not found: {self.roster_file}")
            self.roster_data = pd.read_excel(self.roster_file, sheet_name='Master_Roster')
        
        # Clean player names (reuse the same logic)
        def clean_player_name(name):
            if pd.isna(name):
                return None
            name = str(name).strip()
            suffixes_to_remove = ['(IR)', '(PUP)', '(NFI)', '(COVID)', '(SUSP)', '(RESERVE)', '(Out)', '(Questionable)', '(Doubtful)']
            for suffix in suffixes_to_remove:
                name = name.replace(suffix, '').strip()
            return ' '.join(name.split())
        
        # Apply cleaning and filter for valid mappings
        self.roster_data['clean_name'] = self.roster_data['player_name'].apply(clean_player_name)
        valid_mappings = self.roster_data.dropna(subset=['clean_name', 'team'])
        
        self.player_team_mapping = dict(zip(valid_mappings['clean_name'], valid_mappings['team']))
        
        print(f"Created player-team mapping for {len(self.player_team_mapping)} players")
        return self.player_team_mapping
    
    def load_time_weighted_data(self, week_2025_file: str, weeks_2024_file: str, 
                               target_weeks_2024: list, projection_week: int, 
                               decay_coefficient: float = 0.7) -> pd.DataFrame:
        """
        Load and time-weight game data based on projection week.
        
        Args:
            week_2025_file: Path to 2025 season data
            weeks_2024_file: Path to 2024 season data
            target_weeks_2024: List of 2024 weeks to include
            projection_week: Current week for projections
            decay_coefficient: Exponential decay factor
            
        Returns:
            Time-weighted game data DataFrame
        """
        print(f"Loading time-weighted data for Week {projection_week}...")
        
        # Load 2025 data
        df_2025 = pd.read_csv(week_2025_file) if os.path.exists(week_2025_file) else pd.DataFrame()
        
        # Load 2024 data
        df_2024 = pd.read_csv(weeks_2024_file) if os.path.exists(weeks_2024_file) else pd.DataFrame()
        
        # Determine data selection based on projection week
        if projection_week == 1:
            # Week 1: Use 10 games from 2024
            weeks_to_include_2025 = []
            weeks_to_include_2024 = sorted(target_weeks_2024, reverse=True)[:10]
            df_2025_filtered = pd.DataFrame()
            df_2024_filtered = df_2024[df_2024['week'].isin(weeks_to_include_2024)].copy()
            
        elif projection_week == 2:
            # Week 2: Use 9 games from 2024 + 1 game from 2025
            available_2025_weeks = sorted(df_2025['week'].unique()) if not df_2025.empty else []
            weeks_to_include_2025 = [available_2025_weeks[0]] if available_2025_weeks else []
            weeks_to_include_2024 = sorted(target_weeks_2024, reverse=True)[:9]
            df_2025_filtered = df_2025[df_2025['week'].isin(weeks_to_include_2025)].copy() if weeks_to_include_2025 else pd.DataFrame()
            df_2024_filtered = df_2024[df_2024['week'].isin(weeks_to_include_2024)].copy()
            
        else:
            # Week 3+: Use available 2025 weeks + fill with 2024
            available_2025_weeks = sorted(df_2025['week'].unique()) if not df_2025.empty else []
            weeks_to_include_2025 = [w for w in available_2025_weeks if w < projection_week]
            weeks_needed_from_2024 = 10 - len(weeks_to_include_2025)
            weeks_to_include_2024 = sorted(target_weeks_2024, reverse=True)[:weeks_needed_from_2024]
            df_2025_filtered = df_2025[df_2025['week'].isin(weeks_to_include_2025)].copy() if weeks_to_include_2025 else pd.DataFrame()
            df_2024_filtered = df_2024[df_2024['week'].isin(weeks_to_include_2024)].copy()
        
        # Apply time weights
        weight_map = {}
        
        # 2025 weeks (most recent = 1.0)
        if not df_2025_filtered.empty:
            for i, week in enumerate(sorted(weeks_to_include_2025, reverse=True)):
                weight_map[('2025', week)] = decay_coefficient ** i
            df_2025_filtered['time_weight'] = df_2025_filtered['week'].apply(lambda w: weight_map[('2025', w)])
        
        # 2024 weeks (continuing decay)
        if not df_2024_filtered.empty:
            start_weight = decay_coefficient ** len(weeks_to_include_2025)
            for i, week in enumerate(sorted(weeks_to_include_2024, reverse=True)):
                weight_map[('2024', week)] = start_weight * (decay_coefficient ** i)
            df_2024_filtered['time_weight'] = df_2024_filtered['week'].apply(lambda w: weight_map[('2024', w)])
        
        # Combine datasets
        if not df_2024_filtered.empty and not df_2025_filtered.empty:
            df_combined = pd.concat([df_2025_filtered, df_2024_filtered], ignore_index=True)
        elif not df_2025_filtered.empty:
            df_combined = df_2025_filtered.copy()
        elif not df_2024_filtered.empty:
            df_combined = df_2024_filtered.copy()
        else:
            raise ValueError("No game data available")
        
        print(f"Loaded {len(df_combined)} game records with time weighting")
        return df_combined
    
    def filter_active_players(self, df_game_data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter game data to only include active players.
        
        Args:
            df_game_data: Game data DataFrame
            
        Returns:
            Filtered DataFrame with only active players
        """
        if not self.active_roster:
            self.load_active_roster()
        
        # Filter to active players only
        df_filtered = df_game_data[df_game_data['player'].isin(self.active_roster)].copy()
        
        print(f"Filtered to {len(df_filtered)} records for {df_filtered['player'].nunique()} active players")
        return df_filtered
    
    def filter_by_snap_count(self, df_game_data: pd.DataFrame, 
                            snap_threshold: float = 20.0,
                            snap_file: str = "data/snap_filtering_report.csv") -> pd.DataFrame:
        """
        Filter players based on recent snap count percentage.
        
        Args:
            df_game_data: Game data DataFrame
            snap_threshold: Minimum snap percentage threshold
            snap_file: Path to snap filtering report
            
        Returns:
            Filtered DataFrame with only players meeting snap threshold
        """
        if not os.path.exists(snap_file):
            print(f"Snap filtering file not found: {snap_file}")
            return df_game_data
        
        try:
            # Load snap count data
            df_snaps = pd.read_csv(snap_file)
            
            # Filter to players meeting snap threshold
            active_snap_players = df_snaps[
                (df_snaps['included'] == True) | 
                (df_snaps['snap_pct_recent'] >= snap_threshold)
            ]['player'].tolist()
            
            # Filter game data to only include players with sufficient snaps
            df_filtered = df_game_data[df_game_data['player'].isin(active_snap_players)].copy()
            
            print(f"Snap filtering: {len(active_snap_players)} players meet {snap_threshold}% snap threshold")
            print(f"Filtered to {len(df_filtered)} records for {df_filtered['player'].nunique()} players with significant snaps")
            
            return df_filtered
            
        except Exception as e:
            print(f"Error filtering by snap count: {e}")
            return df_game_data
