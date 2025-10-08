"""
Historical Variance Analysis Module

Analyzes historical variance patterns by:
- Player position (QB, RB, WR, TE)
- Individual player consistency
- Home/away splits
- Weather conditions
- Opponent strength

This data is used to create realistic probability distributions for Monte Carlo simulations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class VarianceAnalyzer:
    """
    Analyzes historical variance patterns to inform Monte Carlo simulations.
    
    Key responsibilities:
    1. Calculate position-specific variance patterns
    2. Analyze individual player consistency
    3. Determine home/away variance differences
    4. Create variance models for simulation
    """
    
    def __init__(self):
        self.position_variance: Dict[str, Dict] = {}
        self.player_variance: Dict[str, Dict] = {}
        self.home_away_variance: Dict[str, Dict] = {}
        self.weather_variance: Dict[str, Dict] = {}
        
    def analyze_historical_variance(self, df_game_data: pd.DataFrame, 
                                  df_players: pd.DataFrame) -> Dict:
        """
        Comprehensive variance analysis across all dimensions.
        
        Args:
            df_game_data: Historical game data
            df_players: Player statistics DataFrame
            
        Returns:
            Dictionary containing all variance analysis results
        """
        print("Analyzing historical variance patterns...")
        
        variance_results = {}
        
        # 1. Position-specific variance analysis
        variance_results['position'] = self._analyze_position_variance(df_game_data, df_players)
        
        # 2. Individual player variance analysis
        variance_results['player'] = self._analyze_player_variance(df_game_data, df_players)
        
        # 3. Home/away variance analysis
        variance_results['home_away'] = self._analyze_home_away_variance(df_game_data)
        
        # 4. Weather variance analysis (if available)
        if 'weather' in df_game_data.columns:
            variance_results['weather'] = self._analyze_weather_variance(df_game_data)
        
        print("Historical variance analysis complete")
        return variance_results
    
    def _analyze_position_variance(self, df_game_data: pd.DataFrame, 
                                  df_players: pd.DataFrame) -> Dict:
        """Analyze variance patterns by position."""
        print("Analyzing position-specific variance...")
        
        position_variance = {}
        positions = ['QB', 'RB', 'WR', 'TE']
        
        # Get position mapping
        player_positions = dict(zip(df_players['name'], df_players.get('pos', 'QB')))
        
        for position in positions:
            # Get players of this position
            position_players = [p for p, pos in player_positions.items() if pos == position]
            
            if not position_players:
                continue
            
            # Filter game data for this position
            position_data = df_game_data[df_game_data['player'].isin(position_players)]
            
            if position_data.empty:
                continue
            
            # Calculate variance for key stats
            stat_columns = ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td']
            position_stats = {}
            
            for stat in stat_columns:
                if stat in position_data.columns:
                    stat_values = position_data[stat].dropna()
                    if len(stat_values) > 10:  # Need sufficient data
                        mean_val = stat_values.mean()
                        std_val = stat_values.std()
                        cv = std_val / mean_val if mean_val > 0 else 0
                        
                        # Calculate distribution parameters
                        if len(stat_values) > 30:
                            # Fit normal distribution
                            mu, sigma = stats.norm.fit(stat_values)
                            position_stats[stat] = {
                                'mean': mean_val,
                                'std': std_val,
                                'cv': cv,
                                'distribution': 'normal',
                                'mu': mu,
                                'sigma': sigma,
                                'samples': len(stat_values)
                            }
                        else:
                            # Use empirical distribution
                            position_stats[stat] = {
                                'mean': mean_val,
                                'std': std_val,
                                'cv': cv,
                                'distribution': 'empirical',
                                'values': stat_values.tolist(),
                                'samples': len(stat_values)
                            }
            
            position_variance[position] = position_stats
        
        return position_variance
    
    def _analyze_player_variance(self, df_game_data: pd.DataFrame, 
                                df_players: pd.DataFrame) -> Dict:
        """Analyze individual player consistency."""
        print("Analyzing individual player variance...")
        
        player_variance = {}
        
        for _, player_row in df_players.iterrows():
            player_name = player_row['name']
            
            # Get player's game data
            player_data = df_game_data[df_game_data['player'] == player_name]
            
            if len(player_data) < 3:  # Need minimum games for variance analysis
                continue
            
            # Calculate variance for key stats
            stat_columns = ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td']
            player_stats = {}
            
            for stat in stat_columns:
                if stat in player_data.columns:
                    stat_values = player_data[stat].dropna()
                    if len(stat_values) > 2:
                        mean_val = stat_values.mean()
                        std_val = stat_values.std()
                        cv = std_val / mean_val if mean_val > 0 else 0
                        
                        # Calculate consistency score (lower CV = more consistent)
                        consistency_score = max(0, 1 - cv) if cv < 2 else 0
                        
                        player_stats[stat] = {
                            'mean': mean_val,
                            'std': std_val,
                            'cv': cv,
                            'consistency_score': consistency_score,
                            'samples': len(stat_values)
                        }
            
            if player_stats:
                player_variance[player_name] = player_stats
        
        return player_variance
    
    def _analyze_home_away_variance(self, df_game_data: pd.DataFrame) -> Dict:
        """Analyze home/away variance differences."""
        print("Analyzing home/away variance...")
        
        if 'home_team' not in df_game_data.columns or 'away_team' not in df_game_data.columns:
            return {}
        
        # Add home/away indicator
        df_with_location = df_game_data.copy()
        df_with_location['is_home'] = df_with_location['team'] == df_with_location['home_team']
        
        home_away_variance = {}
        
        # Analyze variance by location
        for location in [True, False]:  # True = home, False = away
            location_data = df_with_location[df_with_location['is_home'] == location]
            
            if location_data.empty:
                continue
            
            location_name = 'home' if location else 'away'
            stat_columns = ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td']
            location_stats = {}
            
            for stat in stat_columns:
                if stat in location_data.columns:
                    stat_values = location_data[stat].dropna()
                    if len(stat_values) > 10:
                        mean_val = stat_values.mean()
                        std_val = stat_values.std()
                        cv = std_val / mean_val if mean_val > 0 else 0
                        
                        location_stats[stat] = {
                            'mean': mean_val,
                            'std': std_val,
                            'cv': cv,
                            'samples': len(stat_values)
                        }
            
            home_away_variance[location_name] = location_stats
        
        return home_away_variance
    
    def _analyze_weather_variance(self, df_game_data: pd.DataFrame) -> Dict:
        """Analyze weather impact on variance."""
        print("Analyzing weather variance...")
        
        weather_variance = {}
        
        # Extract weather conditions
        weather_conditions = ['clear', 'rain', 'snow', 'wind', 'cold', 'hot']
        
        for condition in weather_conditions:
            # Filter data for this weather condition
            condition_data = df_game_data[
                df_game_data['weather'].str.contains(condition, case=False, na=False)
            ]
            
            if len(condition_data) < 10:
                continue
            
            stat_columns = ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td']
            condition_stats = {}
            
            for stat in stat_columns:
                if stat in condition_data.columns:
                    stat_values = condition_data[stat].dropna()
                    if len(stat_values) > 5:
                        mean_val = stat_values.mean()
                        std_val = stat_values.std()
                        cv = std_val / mean_val if mean_val > 0 else 0
                        
                        condition_stats[stat] = {
                            'mean': mean_val,
                            'std': std_val,
                            'cv': cv,
                            'samples': len(stat_values)
                        }
            
            if condition_stats:
                weather_variance[condition] = condition_stats
        
        return weather_variance
    
    def get_variance_parameters(self, player_name: str, stat: str, 
                               position: str, is_home: bool = True,
                               weather: str = 'clear') -> Dict:
        """
        Get variance parameters for a specific player and stat.
        
        Args:
            player_name: Player name
            stat: Statistic name
            position: Player position
            is_home: Whether playing at home
            weather: Weather condition
            
        Returns:
            Dictionary with variance parameters for simulation
        """
        # Start with position-specific variance
        if position in self.position_variance and stat in self.position_variance[position]:
            base_params = self.position_variance[position][stat].copy()
        else:
            # Default parameters if no position data
            base_params = {
                'mean': 0,
                'std': 1,
                'cv': 0.5,
                'distribution': 'normal',
                'mu': 0,
                'sigma': 1
            }
        
        # Adjust for individual player consistency
        if player_name in self.player_variance and stat in self.player_variance[player_name]:
            player_params = self.player_variance[player_name][stat]
            
            # Blend position and player-specific parameters
            # Weight: 70% player-specific, 30% position average
            base_params['mean'] = 0.7 * player_params['mean'] + 0.3 * base_params['mean']
            base_params['std'] = 0.7 * player_params['std'] + 0.3 * base_params['std']
            base_params['cv'] = 0.7 * player_params['cv'] + 0.3 * base_params['cv']
        
        # Adjust for home/away
        location = 'home' if is_home else 'away'
        if location in self.home_away_variance and stat in self.home_away_variance[location]:
            location_params = self.home_away_variance[location][stat]
            
            # Apply location adjustment (typically 5-10% variance difference)
            location_multiplier = 1.0
            if is_home:
                location_multiplier = 0.95  # Slightly less variance at home
            else:
                location_multiplier = 1.05  # Slightly more variance away
            
            base_params['std'] *= location_multiplier
        
        # Adjust for weather
        if weather in self.weather_variance and stat in self.weather_variance[weather]:
            weather_params = self.weather_variance[weather][stat]
            
            # Weather typically increases variance
            weather_multiplier = 1.1 if weather != 'clear' else 1.0
            base_params['std'] *= weather_multiplier
        
        return base_params
    
    def calculate_confidence_intervals(self, mean: float, std: float, 
                                      distribution: str = 'normal') -> Dict:
        """
        Calculate confidence intervals for a given distribution.
        
        Args:
            mean: Distribution mean
            std: Distribution standard deviation
            distribution: Distribution type ('normal', 'lognormal', 'beta')
            
        Returns:
            Dictionary with confidence intervals
        """
        if distribution == 'normal':
            return {
                'p10': mean - 1.28 * std,
                'p25': mean - 0.675 * std,
                'p50': mean,
                'p75': mean + 0.675 * std,
                'p90': mean + 1.28 * std,
                'p95': mean + 1.645 * std,
                'p99': mean + 2.33 * std
            }
        elif distribution == 'lognormal':
            # For lognormal, work with log parameters
            mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
            sigma = np.sqrt(np.log(1 + std**2 / mean**2))
            
            return {
                'p10': np.exp(mu - 1.28 * sigma),
                'p25': np.exp(mu - 0.675 * sigma),
                'p50': np.exp(mu),
                'p75': np.exp(mu + 0.675 * sigma),
                'p90': np.exp(mu + 1.28 * sigma),
                'p95': np.exp(mu + 1.645 * sigma),
                'p99': np.exp(mu + 2.33 * sigma)
            }
        else:
            # Default to normal
            return self.calculate_confidence_intervals(mean, std, 'normal')
