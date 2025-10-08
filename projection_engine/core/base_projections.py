"""
Base Projections Module

Generates initial player projections based on:
- Historical performance
- Schedule strength adjustments
- Opponent matchups
- Time weighting

These base projections serve as the foundation for Monte Carlo simulations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class BaseProjections:
    """
    Generates base projections for all players.
    
    Key responsibilities:
    1. Calculate base projections from historical data
    2. Apply schedule strength adjustments
    3. Apply opponent matchup adjustments
    4. Generate initial projection estimates
    """
    
    def __init__(self):
        self.base_projections: pd.DataFrame = pd.DataFrame()
        self.projection_columns = [
            'proj_pass_yd', 'proj_pass_td', 'proj_int',
            'proj_rush_yd', 'proj_rush_td',
            'proj_rec_yd', 'proj_rec_td', 'proj_rec'
        ]
        
    def generate_base_projections(self, df_players: pd.DataFrame, 
                                 df_team: pd.DataFrame,
                                 opponent_adjustments: Dict) -> pd.DataFrame:
        """
        Generate base projections for all players.
        
        Args:
            df_players: Player statistics DataFrame
            df_team: Team statistics DataFrame
            opponent_adjustments: Opponent adjustment multipliers
            
        Returns:
            DataFrame with base projections
        """
        print("Generating base projections...")
        
        # Start with player data
        base_projections = df_players.copy()
        
        # Calculate base projections for each player
        for idx, player_row in base_projections.iterrows():
            team = player_row['team']
            
            # Get opponent adjustments for this team
            team_adjustments = opponent_adjustments.get(team, {})
            
            # Calculate projections for each stat
            projections = self._calculate_player_projections(player_row, team_adjustments)
            
            # Update projection columns
            for stat, value in projections.items():
                if f'proj_{stat}' in base_projections.columns:
                    base_projections.loc[idx, f'proj_{stat}'] = value
        
        # Apply regression to mean
        base_projections = self._apply_regression_to_mean(base_projections, df_team)
        
        # Validate and cap projections
        base_projections = self._validate_and_cap_projections(base_projections)
        
        print(f"Base projections generated for {len(base_projections)} players")
        return base_projections
    
    def _calculate_player_projections(self, player_row: pd.Series, 
                                   team_adjustments: Dict) -> Dict:
        """
        Calculate projections for a single player.
        
        Args:
            player_row: Player statistics row
            team_adjustments: Team-specific adjustments
            
        Returns:
            Dictionary with player projections
        """
        projections = {}
        
        # Key statistics to project
        stat_mappings = {
            'pass_yd': ('pass_yd', 'pass_yd_per_att'),
            'pass_td': ('pass_td', 'pass_td_per_att'),
            'int': ('pass_int', 'pass_int_per_att'),
            'rush_yd': ('rush_yd', 'rush_yd_per_att'),
            'rush_td': ('rush_td', 'rush_td_per_att'),
            'rec_yd': ('rec_yd', 'rec_yd_per_tar'),
            'rec_td': ('rec_td', 'rec_td_per_tar'),
            'rec': ('rec', 'rec_rate')
        }
        
        for proj_stat, (base_stat, rate_stat) in stat_mappings.items():
            # Get base values
            base_value = player_row.get(base_stat, 0)
            rate_value = player_row.get(rate_stat, 0)
            attempts = self._get_attempts_for_stat(proj_stat, player_row)
            
            # Calculate base projection
            if attempts > 0 and rate_value > 0:
                base_projection = attempts * rate_value
            else:
                base_projection = base_value
            
            # Apply opponent adjustment
            adjustment = team_adjustments.get(proj_stat, 1.0)
            final_projection = base_projection * adjustment
            
            projections[proj_stat] = max(0, final_projection)
        
        return projections
    
    def _get_attempts_for_stat(self, stat: str, player_row: pd.Series) -> float:
        """Get attempt count for a statistic."""
        if stat in ['pass_yd', 'pass_td', 'int']:
            return player_row.get('proj_pass_att', player_row.get('pass_att', 0))
        elif stat in ['rush_yd', 'rush_td']:
            return player_row.get('proj_rush_att', player_row.get('rush_att', 0))
        elif stat in ['rec_yd', 'rec_td', 'rec']:
            return player_row.get('proj_tar', player_row.get('tar', 0))
        else:
            return 0
    
    def _apply_regression_to_mean(self, df_players: pd.DataFrame, 
                                 df_team: pd.DataFrame) -> pd.DataFrame:
        """
        Apply regression to mean to prevent overfitting.
        
        Args:
            df_players: Player projections DataFrame
            df_team: Team statistics DataFrame
            
        Returns:
            DataFrame with regressed projections
        """
        print("Applying regression to mean...")
        
        regressed_projections = df_players.copy()
        
        # Calculate league averages
        league_averages = self._calculate_league_averages(df_team)
        
        # Apply regression to each projection column
        for col in self.projection_columns:
            if col in regressed_projections.columns:
                # Calculate regression factor based on stat type
                regression_factor = self._calculate_regression_factor(col)
                
                # Apply regression
                regressed_projections[col] = regressed_projections.apply(
                    lambda row: self._regress_to_mean(
                        row[col], league_averages.get(col.replace('proj_', ''), 0), 
                        regression_factor
                    ), axis=1
                )
        
        return regressed_projections
    
    def _calculate_league_averages(self, df_team: pd.DataFrame) -> Dict:
        """Calculate league averages for regression."""
        # This would calculate league averages from team data
        # For now, return default values
        return {
            'pass_yd': 200,
            'pass_td': 1.5,
            'int': 1.0,
            'rush_yd': 80,
            'rush_td': 0.8,
            'rec_yd': 60,
            'rec_td': 0.5,
            'rec': 4.0
        }
    
    def _calculate_regression_factor(self, stat: str) -> float:
        """Calculate regression factor for a statistic."""
        # TDs need more regression (higher variance)
        if 'td' in stat:
            return 0.15
        # Yards need moderate regression
        elif 'yd' in stat:
            return 0.08
        # Other stats need minimal regression
        else:
            return 0.05
    
    def _regress_to_mean(self, projection: float, league_avg: float, 
                        regression_factor: float) -> float:
        """Apply regression to mean."""
        if league_avg == 0:
            return projection
        
        # Regression formula: (1-factor) * projection + factor * league_avg
        regressed = (1 - regression_factor) * projection + regression_factor * league_avg
        return max(0, regressed)
    
    def _validate_and_cap_projections(self, df_players: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and cap projections to realistic ranges.
        
        Args:
            df_players: Player projections DataFrame
            
        Returns:
            DataFrame with validated projections
        """
        print("Validating and capping projections...")
        
        validated_projections = df_players.copy()
        
        # Define caps for each projection type
        caps = {
            'proj_pass_yd': 500,    # Single-game NFL record
            'proj_rush_yd': 300,    # Single-game NFL record
            'proj_rec_yd': 400,     # Single-game NFL record
            'proj_pass_td': 7,      # Single-game NFL record
            'proj_rush_td': 5,      # Single-game NFL record
            'proj_rec_td': 5,       # Single-game NFL record
            'proj_rec': 20          # Reasonable reception cap
        }
        
        # Apply caps
        for col, cap in caps.items():
            if col in validated_projections.columns:
                # Count unrealistic projections
                unrealistic = validated_projections[validated_projections[col] > cap]
                if not unrealistic.empty:
                    print(f"⚠️ Capping {len(unrealistic)} players with {col} > {cap}")
                    validated_projections[col] = validated_projections[col].clip(upper=cap)
        
        # Ensure non-negative values
        for col in self.projection_columns:
            if col in validated_projections.columns:
                validated_projections[col] = validated_projections[col].clip(lower=0)
        
        print("Projection validation complete")
        return validated_projections
    
    def get_projection_summary(self, df_projections: pd.DataFrame) -> Dict:
        """
        Get summary of projections.
        
        Args:
            df_projections: Projections DataFrame
            
        Returns:
            Dictionary with projection summary
        """
        summary = {
            'total_players': len(df_projections),
            'projection_stats': {}
        }
        
        # Calculate summary statistics for each projection column
        for col in self.projection_columns:
            if col in df_projections.columns:
                values = df_projections[col].dropna()
                if not values.empty:
                    summary['projection_stats'][col] = {
                        'mean': values.mean(),
                        'median': values.median(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max()
                    }
        
        return summary
