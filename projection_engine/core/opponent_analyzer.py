"""
Opponent Strength Analysis Module

Analyzes opponent defensive strength to adjust projections for upcoming matchups.
This is crucial for accurate projections as different opponents have varying defensive capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class OpponentAnalyzer:
    """
    Analyzes opponent defensive strength for matchup adjustments.
    
    Key responsibilities:
    1. Calculate opponent defensive ratings
    2. Determine matchup advantages/disadvantages
    3. Generate opponent adjustment multipliers
    4. Apply opponent-specific adjustments to projections
    """
    
    def __init__(self):
        self.opponent_ratings: Dict[str, Dict] = {}
        self.matchup_adjustments: Dict[str, Dict] = {}
        
    def analyze_opponent_strength(self, df_team: pd.DataFrame, 
                                 df_players: pd.DataFrame) -> Dict:
        """
        Analyze opponent strength for all teams.
        
        Args:
            df_team: Team statistics DataFrame
            df_players: Player statistics DataFrame
            
        Returns:
            Dictionary with opponent adjustments for each team
        """
        print("Analyzing opponent defensive strength...")
        
        # Calculate opponent ratings
        self.opponent_ratings = self._calculate_opponent_ratings(df_team)
        
        # Generate matchup adjustments
        self.matchup_adjustments = self._generate_matchup_adjustments(df_team, df_players)
        
        print(f"Opponent analysis complete for {len(self.opponent_ratings)} teams")
        return self.matchup_adjustments
    
    def _calculate_opponent_ratings(self, df_team: pd.DataFrame) -> Dict:
        """
        Calculate defensive ratings for each team.
        
        Args:
            df_team: Team statistics DataFrame
            
        Returns:
            Dictionary with opponent ratings
        """
        opponent_ratings = {}
        
        for team in df_team.index:
            team_data = df_team.loc[team]
            
            # Calculate defensive ratings (lower is better for defense)
            defensive_stats = {
                'pass_defense': self._calculate_pass_defense_rating(team_data),
                'rush_defense': self._calculate_rush_defense_rating(team_data),
                'rec_defense': self._calculate_rec_defense_rating(team_data),
                'overall_defense': 0
            }
            
            # Calculate overall defensive rating
            defensive_stats['overall_defense'] = np.mean([
                defensive_stats['pass_defense'],
                defensive_stats['rush_defense'],
                defensive_stats['rec_defense']
            ])
            
            opponent_ratings[team] = defensive_stats
        
        return opponent_ratings
    
    def _calculate_pass_defense_rating(self, team_data: pd.Series) -> float:
        """Calculate pass defense rating (lower = better defense)."""
        # Key pass defense metrics
        pass_yds_allowed = team_data.get('def_pass_yd', 0)
        pass_tds_allowed = team_data.get('def_pass_td', 0)
        ints_forced = team_data.get('def_int', 0)
        sacks = team_data.get('def_sacks', 0)
        
        # Calculate rating (lower is better)
        # Weight: 60% yards, 30% TDs, 10% turnovers
        rating = (pass_yds_allowed * 0.6 + pass_tds_allowed * 30 * 0.3) - (ints_forced * 20 + sacks * 2) * 0.1
        
        return max(0, rating)  # Ensure non-negative
    
    def _calculate_rush_defense_rating(self, team_data: pd.Series) -> float:
        """Calculate rush defense rating (lower = better defense)."""
        # Key rush defense metrics
        rush_yds_allowed = team_data.get('def_rush_yd', 0)
        rush_tds_allowed = team_data.get('def_rush_td', 0)
        
        # Calculate rating (lower is better)
        rating = rush_yds_allowed * 0.7 + rush_tds_allowed * 30 * 0.3
        
        return max(0, rating)
    
    def _calculate_rec_defense_rating(self, team_data: pd.Series) -> float:
        """Calculate receiving defense rating (lower = better defense)."""
        # Key receiving defense metrics
        rec_yds_allowed = team_data.get('def_rec_yd', 0)
        rec_tds_allowed = team_data.get('def_rec_td', 0)
        targets_allowed = team_data.get('def_targets', 0)
        
        # Calculate rating (lower is better)
        rating = rec_yds_allowed * 0.5 + rec_tds_allowed * 30 * 0.3 + targets_allowed * 0.2
        
        return max(0, rating)
    
    def _generate_matchup_adjustments(self, df_team: pd.DataFrame, 
                                    df_players: pd.DataFrame) -> Dict:
        """
        Generate matchup adjustments for each team.
        
        Args:
            df_team: Team statistics DataFrame
            df_players: Player statistics DataFrame
            
        Returns:
            Dictionary with matchup adjustments for each team
        """
        matchup_adjustments = {}
        
        for team in df_team.index:
            if 'next_op' not in df_team.columns:
                continue
                
            next_opponent = df_team.loc[team, 'next_op']
            
            if pd.isna(next_opponent) or next_opponent not in self.opponent_ratings:
                # Default neutral adjustments
                matchup_adjustments[team] = self._get_neutral_adjustments()
                continue
            
            # Get opponent defensive ratings
            opponent_ratings = self.opponent_ratings[next_opponent]
            
            # Calculate adjustments based on opponent strength
            adjustments = self._calculate_opponent_adjustments(opponent_ratings)
            matchup_adjustments[team] = adjustments
        
        return matchup_adjustments
    
    def _get_neutral_adjustments(self) -> Dict:
        """Get neutral adjustment multipliers."""
        return {
            'pass_yd': 1.0, 'pass_td': 1.0, 'int': 1.0,
            'rush_yd': 1.0, 'rush_td': 1.0,
            'rec_yd': 1.0, 'rec_td': 1.0, 'rec': 1.0
        }
    
    def _calculate_opponent_adjustments(self, opponent_ratings: Dict) -> Dict:
        """
        Calculate adjustment multipliers based on opponent ratings.
        
        Args:
            opponent_ratings: Opponent defensive ratings
            
        Returns:
            Dictionary with adjustment multipliers
        """
        # Get opponent ratings
        pass_def = opponent_ratings['pass_defense']
        rush_def = opponent_ratings['rush_defense']
        rec_def = opponent_ratings['rec_defense']
        overall_def = opponent_ratings['overall_defense']
        
        # Calculate adjustments (weaker defense = higher multiplier)
        # Use league average as baseline (assume 200 yards, 2 TDs as average)
        league_avg_pass = 200
        league_avg_rush = 100
        league_avg_rec = 150
        
        # Calculate multipliers
        pass_multiplier = self._calculate_multiplier(pass_def, league_avg_pass)
        rush_multiplier = self._calculate_multiplier(rush_def, league_avg_rush)
        rec_multiplier = self._calculate_multiplier(rec_def, league_avg_rec)
        
        return {
            'pass_yd': pass_multiplier,
            'pass_td': pass_multiplier,
            'int': 1.0 / pass_multiplier,  # Inverse relationship for interceptions
            'rush_yd': rush_multiplier,
            'rush_td': rush_multiplier,
            'rec_yd': rec_multiplier,
            'rec_td': rec_multiplier,
            'rec': rec_multiplier
        }
    
    def _calculate_multiplier(self, opponent_rating: float, league_average: float) -> float:
        """
        Calculate adjustment multiplier based on opponent rating.
        
        Args:
            opponent_rating: Opponent defensive rating
            league_average: League average for this stat
            
        Returns:
            Adjustment multiplier
        """
        # Calculate ratio (higher ratio = weaker defense)
        ratio = opponent_rating / league_average if league_average > 0 else 1.0
        
        # Convert to multiplier (weaker defense = higher multiplier)
        # Clamp to reasonable range (0.7 to 1.3)
        multiplier = max(0.7, min(1.3, ratio))
        
        return multiplier
    
    def apply_opponent_adjustments(self, df_players: pd.DataFrame) -> pd.DataFrame:
        """
        Apply opponent adjustments to player projections.
        
        Args:
            df_players: Player projections DataFrame
            
        Returns:
            DataFrame with opponent-adjusted projections
        """
        print("Applying opponent adjustments to projections...")
        
        adjusted_players = df_players.copy()
        
        for idx, player_row in adjusted_players.iterrows():
            team = player_row['team']
            
            if team in self.matchup_adjustments:
                adjustments = self.matchup_adjustments[team]
                
                # Apply adjustments to projection columns
                projection_columns = [
                    'proj_pass_yd', 'proj_pass_td', 'proj_int',
                    'proj_rush_yd', 'proj_rush_td',
                    'proj_rec_yd', 'proj_rec_td', 'proj_rec'
                ]
                
                for col in projection_columns:
                    if col in adjusted_players.columns:
                        stat_name = col.replace('proj_', '')
                        if stat_name in adjustments:
                            multiplier = adjustments[stat_name]
                            adjusted_players.loc[idx, col] *= multiplier
        
        print("Opponent adjustments applied")
        return adjusted_players
    
    def get_opponent_summary(self, team: str) -> Dict:
        """
        Get opponent analysis summary for a team.
        
        Args:
            team: Team name
            
        Returns:
            Dictionary with opponent analysis
        """
        if team not in self.opponent_ratings:
            return {}
        
        ratings = self.opponent_ratings[team]
        adjustments = self.matchup_adjustments.get(team, {})
        
        return {
            'team': team,
            'defensive_ratings': ratings,
            'adjustment_multipliers': adjustments,
            'overall_defense_rank': self._get_defense_rank(team)
        }
    
    def _get_defense_rank(self, team: str) -> int:
        """Get overall defensive ranking for a team."""
        if team not in self.opponent_ratings:
            return 0
        
        # Get all team ratings for ranking
        all_ratings = [(t, r['overall_defense']) for t, r in self.opponent_ratings.items()]
        all_ratings.sort(key=lambda x: x[1])  # Sort by defense rating (lower is better)
        
        # Find team's rank
        for rank, (t, rating) in enumerate(all_ratings, 1):
            if t == team:
                return rank
        
        return 0
