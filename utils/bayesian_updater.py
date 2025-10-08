#!/usr/bin/env python3
"""
Bayesian Updating Module for NFL Projections

Implements Bayesian updating to continuously improve player projections
as new game data becomes available. Uses historical performance as prior
knowledge and updates with new evidence.

Key Features:
- Real-time learning from new game data
- Prior knowledge integration from historical performance
- Uncertainty quantification with confidence intervals
- Position-specific updating patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings

class BayesianUpdater:
    """
    Bayesian updating system for NFL player projections.
    
    Uses historical data as prior knowledge and updates projections
    as new game data becomes available.
    """
    
    def __init__(self, learning_rate: float = 0.1, min_games: int = 3):
        """
        Initialize Bayesian updater.
        
        Args:
            learning_rate: How quickly to adapt to new data (0.0-1.0)
            min_games: Minimum games required for reliable updating
        """
        self.learning_rate = learning_rate
        self.min_games = min_games
        self.player_priors = {}
        
    def calculate_prior_from_historical(self, player_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate prior distributions from historical player data.
        
        Args:
            player_data: DataFrame with player's historical game stats
            
        Returns:
            Dictionary with prior parameters for each statistic
        """
        if player_data.empty:
            return {}
            
        priors = {}
        stats = ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td', 
                'pass_att', 'rush_att', 'targets', 'rec']
        
        for stat in stats:
            if stat in player_data.columns:
                values = player_data[stat].dropna()
                if len(values) >= self.min_games:
                    # Calculate prior parameters
                    mean = values.mean()
                    variance = values.var()
                    
                    # Use conjugate prior (normal-gamma) for normal likelihood
                    # Prior precision (inverse variance)
                    prior_precision = 1.0 / max(variance, 0.1)  # Avoid division by zero
                    
                    priors[stat] = {
                        'mean': mean,
                        'variance': variance,
                        'precision': prior_precision,
                        'sample_size': len(values)
                    }
                else:
                    # Use default priors for insufficient data
                    priors[stat] = {
                        'mean': 0.0,
                        'variance': 1.0,
                        'precision': 1.0,
                        'sample_size': 0
                    }
        
        return priors
    
    def update_with_new_game(self, player_name: str, new_game_data: Dict, 
                           historical_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Update player projections using Bayesian updating with new game data.
        
        Args:
            player_name: Name of the player
            new_game_data: Dictionary with new game statistics
            historical_data: DataFrame with player's historical data
            
        Returns:
            Updated projection parameters
        """
        # Get or calculate prior from historical data
        if player_name not in self.player_priors:
            self.player_priors[player_name] = self.calculate_prior_from_historical(historical_data)
        
        priors = self.player_priors[player_name]
        updated_projections = {}
        
        for stat, prior in priors.items():
            if stat in new_game_data and not pd.isna(new_game_data[stat]):
                new_value = new_game_data[stat]
                
                # Bayesian updating formula
                # Posterior precision = prior precision + new data precision
                # Posterior mean = (prior_precision * prior_mean + new_precision * new_value) / posterior_precision
                
                # Assume new data has same precision as historical variance
                new_precision = 1.0 / max(prior['variance'], 0.1)
                
                # Calculate posterior parameters
                posterior_precision = prior['precision'] + new_precision
                posterior_mean = (prior['precision'] * prior['mean'] + 
                                new_precision * new_value) / posterior_precision
                
                # Update with learning rate for gradual adaptation
                final_mean = (1 - self.learning_rate) * prior['mean'] + self.learning_rate * posterior_mean
                final_variance = 1.0 / posterior_precision
                
                updated_projections[stat] = {
                    'mean': final_mean,
                    'variance': final_variance,
                    'precision': posterior_precision,
                    'sample_size': prior['sample_size'] + 1,
                    'confidence_interval': self._calculate_confidence_interval(final_mean, final_variance)
                }
                
                # Update stored prior for next iteration
                self.player_priors[player_name][stat] = {
                    'mean': final_mean,
                    'variance': final_variance,
                    'precision': posterior_precision,
                    'sample_size': prior['sample_size'] + 1
                }
            else:
                # No new data, keep prior
                updated_projections[stat] = prior.copy()
        
        return updated_projections
    
    def _calculate_confidence_interval(self, mean: float, variance: float, 
                                    confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate confidence interval for Bayesian estimate.
        
        Args:
            mean: Posterior mean
            variance: Posterior variance
            confidence: Confidence level (default 0.95)
            
        Returns:
            Dictionary with confidence interval bounds
        """
        std = np.sqrt(variance)
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99% confidence
        
        return {
            'lower': mean - z_score * std,
            'upper': mean + z_score * std,
            'std': std
        }
    
    def get_player_projection(self, player_name: str, stat: str) -> Optional[Dict]:
        """
        Get current projection for a specific player and statistic.
        
        Args:
            player_name: Name of the player
            stat: Statistic name
            
        Returns:
            Projection parameters or None if not available
        """
        if player_name in self.player_priors and stat in self.player_priors[player_name]:
            return self.player_priors[player_name][stat]
        return None
    
    def update_player_ensemble(self, player_data: pd.DataFrame, 
                             bootstrap_results: Dict) -> Dict[str, Dict]:
        """
        Combine Bayesian updating with bootstrap sampling for ensemble projections.
        
        Args:
            player_data: DataFrame with player's historical data
            bootstrap_results: Results from bootstrap sampling
            
        Returns:
            Ensemble projections combining Bayesian and bootstrap methods
        """
        ensemble_projections = {}
        
        # Get Bayesian priors
        bayesian_priors = self.calculate_prior_from_historical(player_data)
        
        for stat, bootstrap_result in bootstrap_results.items():
            if stat in bayesian_priors:
                bayesian_prior = bayesian_priors[stat]
                
                # Combine Bayesian and bootstrap results
                # Weight: 60% bootstrap (data-driven), 40% Bayesian (prior knowledge)
                ensemble_mean = (0.6 * bootstrap_result.get('mean', 0) + 
                               0.4 * bayesian_prior['mean'])
                
                # Use bootstrap variance (more realistic for NFL stats)
                ensemble_variance = bootstrap_result.get('std', 1.0) ** 2
                
                ensemble_projections[stat] = {
                    'mean': ensemble_mean,
                    'std': np.sqrt(ensemble_variance),
                    'p25': ensemble_mean - 0.675 * np.sqrt(ensemble_variance),
                    'p75': ensemble_mean + 0.675 * np.sqrt(ensemble_variance),
                    'bayesian_weight': 0.4,
                    'bootstrap_weight': 0.6,
                    'confidence_interval': self._calculate_confidence_interval(ensemble_mean, ensemble_variance)
                }
            else:
                # Use bootstrap results if no Bayesian prior
                ensemble_projections[stat] = bootstrap_result
        
        return ensemble_projections

def apply_bayesian_updating_to_projections(df_projections: pd.DataFrame, 
                                         df_game_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Bayesian updating to existing projections.
    
    Args:
        df_projections: DataFrame with existing projections
        df_game_data: DataFrame with historical game data
        
    Returns:
        DataFrame with Bayesian-enhanced projections
    """
    updater = BayesianUpdater(learning_rate=0.1, min_games=3)
    enhanced_projections = df_projections.copy()
    
    # Add Bayesian columns
    bayesian_columns = [
        'bayesian_pass_yd_mean', 'bayesian_pass_yd_std', 'bayesian_pass_yd_p25', 'bayesian_pass_yd_p75',
        'bayesian_rush_yd_mean', 'bayesian_rush_yd_std', 'bayesian_rush_yd_p25', 'bayesian_rush_yd_p75',
        'bayesian_rec_yd_mean', 'bayesian_rec_yd_std', 'bayesian_rec_yd_p25', 'bayesian_rec_yd_p75',
        'bayesian_pass_td_mean', 'bayesian_pass_td_std', 'bayesian_pass_td_p25', 'bayesian_pass_td_p75',
        'bayesian_rush_td_mean', 'bayesian_rush_td_std', 'bayesian_rush_td_p25', 'bayesian_rush_td_p75',
        'bayesian_rec_td_mean', 'bayesian_rec_td_std', 'bayesian_rec_td_p25', 'bayesian_rec_td_p75'
    ]
    
    # Initialize columns
    for col in bayesian_columns:
        enhanced_projections[col] = 0.0
    
    # Apply Bayesian updating to each player
    for idx, row in enhanced_projections.iterrows():
        player_name = row.get('name', '')
        if not player_name:
            continue
            
        # Get player's historical data
        player_games = df_game_data[df_game_data['player'] == player_name].copy()
        
        if player_games.empty:
            continue
        
        # Calculate Bayesian priors from historical data
        bayesian_priors = updater.calculate_prior_from_historical(player_games)
        
        # Update projections with Bayesian information
        for stat, prior in bayesian_priors.items():
            if stat in ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td']:
                # Map to column names
                stat_mapping = {
                    'pass_yd': 'bayesian_pass_yd',
                    'rush_yd': 'bayesian_rush_yd', 
                    'rec_yd': 'bayesian_rec_yd',
                    'pass_td': 'bayesian_pass_td',
                    'rush_td': 'bayesian_rush_td',
                    'rec_td': 'bayesian_rec_td'
                }
                
                if stat in stat_mapping:
                    prefix = stat_mapping[stat]
                    enhanced_projections.loc[idx, f'{prefix}_mean'] = prior['mean']
                    enhanced_projections.loc[idx, f'{prefix}_std'] = np.sqrt(prior['variance'])
                    enhanced_projections.loc[idx, f'{prefix}_p25'] = prior['mean'] - 0.675 * np.sqrt(prior['variance'])
                    enhanced_projections.loc[idx, f'{prefix}_p75'] = prior['mean'] + 0.675 * np.sqrt(prior['variance'])
    
    return enhanced_projections
