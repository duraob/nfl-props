"""
Schedule Strength Analysis Module

Calculates schedule strength for each team to normalize their base statistics.
This is crucial for accurate projections as teams face different quality opponents.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import ast


class ScheduleAnalyzer:
    """
    Analyzes team schedule strength to normalize base statistics.
    
    Key responsibilities:
    1. Calculate team schedule strength ratios
    2. Normalize team statistics against league averages
    3. Apply schedule strength adjustments to projections
    """
    
    def __init__(self):
        self.team_schedule_strength: Dict[str, Dict] = {}
        self.league_averages: Dict[str, float] = {}
        
    def calculate_schedule_strength(self, df_team: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate schedule strength for each team based on opponents faced.
        
        Args:
            df_team: Team statistics DataFrame with 'opps' column
            
        Returns:
            DataFrame with schedule strength ratios
        """
        print("Calculating schedule strength for each team...")
        
        # Clean and prepare team data
        df_team_clean = df_team.copy()
        df_team_clean = df_team_clean.fillna(0.0)
        df_team_clean.set_index("team", inplace=True)
        
        # Handle opps column (convert string representation to list)
        df_team_clean['opps'] = df_team_clean['opps'].fillna("[]")
        df_team_clean['opps'] = df_team_clean['opps'].replace(0.0, "[]")
        df_team_clean['opps'] = df_team_clean['opps'].apply(ast.literal_eval)
        df_team_clean['games'] = df_team_clean['opps'].apply(len)
        
        # Calculate league averages
        self.league_averages = self._calculate_league_averages(df_team_clean)
        
        # Calculate team ratios to league average
        df_team_ratios = self._calculate_team_ratios(df_team_clean)
        
        # Calculate schedule strength for each defensive stat
        df_schedule_strength = self._calculate_schedule_strength_ratios(df_team_ratios)
        
        print(f"Schedule strength calculated for {len(df_schedule_strength)} teams")
        return df_schedule_strength
    
    def _calculate_league_averages(self, df_team: pd.DataFrame) -> Dict[str, float]:
        """Calculate league averages for normalization."""
        num_games = df_team["games"].sum()
        num_rush_att = df_team["rush_att"].sum()
        num_pass_att = df_team["pass_att"].sum()
        num_def_rush_att = df_team["def_rush_att"].sum()
        num_def_pass_att = df_team["def_pass_att"].sum()
        
        league_avg = {
            "pass_cmp": df_team["pass_cmp"].sum() / num_games,
            "pass_att": df_team["pass_att"].sum() / num_games,
            "pass_yd": df_team["pass_yd"].sum() / num_pass_att,
            "pass_td": df_team["pass_td"].sum() / num_pass_att,
            "pass_int": df_team["pass_int"].sum() / num_pass_att,
            "sacks": df_team["sacks"].sum() / num_games,
            "rush_att": df_team["rush_att"].sum() / num_games,
            "rush_yd": df_team["rush_yd"].sum() / num_rush_att,
            "rush_td": df_team["rush_td"].sum() / num_rush_att,
            "targets": df_team["targets"].sum() / num_pass_att,
            "rec": df_team["rec"].sum() / num_pass_att,
            "rec_yd": df_team["rec_yd"].sum() / num_pass_att,
            "rec_td": df_team["rec_td"].sum() / num_pass_att,
            "off_fum": df_team["off_fum"].sum() / (num_pass_att + num_rush_att),
            "def_pass_cmp": df_team["def_pass_cmp"].sum() / num_games,
            "def_pass_att": df_team["def_pass_att"].sum() / num_games,
            "def_pass_yd": df_team["def_pass_yd"].sum() / num_def_pass_att,
            "def_pass_td": df_team["def_pass_td"].sum() / num_def_pass_att,
            "def_int": df_team["def_int"].sum() / num_def_pass_att,
            "def_sacks": df_team["def_sacks"].sum() / num_games,
            "def_rush_att": df_team["def_rush_att"].sum() / num_games,
            "def_rush_yd": df_team["def_rush_yd"].sum() / num_def_rush_att,
            "def_rush_td": df_team["def_rush_td"].sum() / num_def_rush_att,
            "def_targets": df_team["def_targets"].sum() / num_def_pass_att,
            "def_rec": df_team["def_rec"].sum() / num_def_pass_att,
            "def_rec_yd": df_team["def_rec_yd"].sum() / num_def_pass_att,
            "def_rec_td": df_team["def_rec_td"].sum() / num_def_pass_att,
            "def_fum": df_team["def_fum"].sum() / (num_def_pass_att + num_def_rush_att)
        }
        
        return league_avg
    
    def _calculate_team_ratios(self, df_team: pd.DataFrame) -> pd.DataFrame:
        """Calculate team ratios to league average."""
        df_ratios = df_team.copy()
        
        # Calculate per-attempt rates for offensive stats
        df_ratios['pass_cmp'] = df_ratios['pass_cmp'] / df_ratios['pass_att']
        df_ratios['pass_yd'] = df_ratios['pass_yd'] / df_ratios['pass_att']
        df_ratios['pass_td'] = df_ratios['pass_td'] / df_ratios['pass_att']
        df_ratios['pass_int'] = df_ratios['pass_int'] / df_ratios['pass_att']
        df_ratios['sacks'] = df_ratios['sacks'] / df_ratios['games']
        
        df_ratios['rush_att'] = df_ratios["rush_att"] / df_ratios['games']
        df_ratios['rush_yd'] = df_ratios["rush_yd"] / df_ratios['rush_att'].replace(0, 0.0001)
        df_ratios['rush_td'] = df_ratios["rush_td"] / df_ratios['rush_att'].replace(0, 0.0001)
        
        df_ratios['targets'] = df_ratios["targets"] / df_ratios['pass_att'].replace(0, 0.0001)
        df_ratios['rec'] = df_ratios["rec"] / df_ratios['targets'].replace(0, 0.0001)
        df_ratios['rec_yd'] = df_ratios["rec_yd"] / df_ratios['pass_att'].replace(0, 0.0001)
        df_ratios['rec_td'] = df_ratios["rec_td"] / df_ratios['pass_att'].replace(0, 0.0001)
        df_ratios['off_fum'] = df_ratios["off_fum"] / (df_ratios['pass_att'] + df_ratios['rush_att']).replace(0, 0.0001)
        
        df_ratios['pass_att'] = df_ratios["pass_att"] / df_ratios['games']
        df_ratios['rush_att'] = df_ratios["rush_att"] / df_ratios['games']
        
        # Calculate per-attempt rates for defensive stats
        df_ratios['def_pass_cmp'] = df_ratios['def_pass_cmp'] / df_ratios['def_pass_att'].replace(0, 0.0001)
        df_ratios['def_pass_yd'] = df_ratios['def_pass_yd'] / df_ratios['def_pass_att'].replace(0, 0.0001)
        df_ratios['def_pass_td'] = df_ratios['def_pass_td'] / df_ratios['def_pass_att'].replace(0, 0.0001)
        df_ratios['def_int'] = df_ratios['def_int'] / df_ratios['def_pass_att'].replace(0, 0.0001)
        df_ratios['def_sacks'] = df_ratios['def_sacks'] / df_ratios['games']
        
        df_ratios['def_rush_yd'] = df_ratios["def_rush_yd"] / df_ratios['def_rush_att'].replace(0, 0.0001)
        df_ratios['def_rush_td'] = df_ratios["def_rush_td"] / df_ratios['def_rush_att'].replace(0, 0.0001)
        
        df_ratios['def_targets'] = df_ratios["def_targets"] / df_ratios['def_pass_att'].replace(0, 0.0001)
        df_ratios['def_rec'] = df_ratios["def_rec"] / df_ratios['def_pass_att'].replace(0, 0.0001)
        df_ratios['def_rec_yd'] = df_ratios["def_rec_yd"] / df_ratios['def_pass_att'].replace(0, 0.0001)
        df_ratios['def_rec_td'] = df_ratios["def_rec_td"] / df_ratios['def_pass_att'].replace(0, 0.0001)
        df_ratios['def_fum'] = df_ratios["def_fum"] / (df_ratios['def_pass_att'] + df_ratios['def_rush_att']).replace(0, 0.0001)
        
        df_ratios['def_pass_att'] = df_ratios["def_pass_att"] / df_ratios['games']
        df_ratios['def_rush_att'] = df_ratios["def_rush_att"] / df_ratios['games']
        
        # Clean up any infinite values
        df_ratios = df_ratios.replace([np.inf, -np.inf], 0)
        df_ratios = df_ratios.fillna(0)
        
        # Calculate ratios to league average
        for col in df_ratios.columns:
            if col in self.league_averages and col not in ['opps', 'games', 'next_op', 'next_game_location', 'next_game_week']:
                stat_value = self.league_averages[col]
                if stat_value != 0:
                    df_ratios[f"ratio_{col}"] = df_ratios[col] / stat_value
                else:
                    df_ratios[f"ratio_{col}"] = 1.0
        
        return df_ratios
    
    def _calculate_schedule_strength_ratios(self, df_ratios: pd.DataFrame) -> pd.DataFrame:
        """Calculate schedule strength ratios for defensive stats."""
        df_schedule = df_ratios.copy()
        
        # Calculate schedule strength for ratio columns (defensive stats)
        ratio_columns = [col for col in df_schedule.columns if col.startswith('ratio_')]
        
        for col in ratio_columns:
            df_schedule[f"schedstr_{col}"] = df_schedule.apply(
                lambda row: self._calc_matchup_strength(row, col, df_schedule), axis=1
            )
        
        # Clean up any infinite values
        df_schedule = df_schedule.replace([np.inf, -np.inf], 0)
        df_schedule = df_schedule.fillna(0)
        
        return df_schedule
    
    def _calc_matchup_strength(self, row, stat, df):
        """
        Calculate matchup strength based on opponents faced.
        
        Args:
            row: Team row data
            stat: Statistic column name
            df: Full team DataFrame
            
        Returns:
            Schedule strength ratio
        """
        stat_agg = 0
        for team in row["opps"]:
            try:
                if team in df.index and pd.api.types.is_numeric_dtype(df[stat]):
                    stat_agg += df.loc[team, stat]
            except (KeyError, TypeError):
                continue
        
        if row["games"] == 0:
            return 0
        
        return stat_agg / row["games"]
    
    def apply_schedule_strength_adjustment(self, player_stats: pd.DataFrame, 
                                        team_schedule_strength: pd.DataFrame) -> pd.DataFrame:
        """
        Apply schedule strength adjustments to player projections.
        
        Args:
            player_stats: Player statistics DataFrame
            team_schedule_strength: Team schedule strength DataFrame
            
        Returns:
            Adjusted player statistics
        """
        print("Applying schedule strength adjustments...")
        
        adjusted_stats = player_stats.copy()
        
        # Apply schedule strength adjustments to base stats
        def apply_adjustment(row, stat_name):
            try:
                team = row['team']
                if team not in team_schedule_strength.index:
                    return 1.0
                
                schedstr_col = f"schedstr_ratio_def_{stat_name}"
                if schedstr_col in team_schedule_strength.columns:
                    schedstr_value = team_schedule_strength.loc[team, schedstr_col]
                    if pd.isna(schedstr_value) or schedstr_value == 0:
                        return 1.0
                    
                    # Apply moderate adjustment to avoid over-correction
                    if schedstr_value > 1.0:
                        # Harder schedule: boost projection
                        adjustment = 1.0 + (schedstr_value - 1.0) * 0.2
                    else:
                        # Easier schedule: reduce projection
                        adjustment = 1.0 + (schedstr_value - 1.0) * 0.2
                    
                    # Clamp to reasonable range
                    return max(0.7, min(1.3, adjustment))
                return 1.0
            except (KeyError, TypeError):
                return 1.0
        
        # Apply adjustments to base stats
        stat_adjustments = {
            'pass_att': 'pass_att',
            'rush_att': 'rush_att', 
            'tar': 'pass_att'  # Targets are based on pass attempts
        }
        
        for player_stat, team_stat in stat_adjustments.items():
            if player_stat in adjusted_stats.columns:
                adjusted_stats[f"proj_{player_stat}"] = adjusted_stats.apply(
                    lambda row: row[player_stat] * apply_adjustment(row, team_stat), axis=1
                )
        
        print("Schedule strength adjustments applied")
        return adjusted_stats
