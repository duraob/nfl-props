"""
Advanced Time Weighting for NFL Projections

This module implements sophisticated time weighting that considers:
- Recent performance momentum
- Opponent strength context
- Game situation factors
- Player role changes over time

Key Features:
- Context-aware momentum analysis
- Opponent strength-adjusted weighting
- Recent vs historical performance comparison
- Dynamic weighting based on player trends
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class AdvancedTimeWeighting:
    def __init__(self):
        self.momentum_weights = {}
        self.context_weights = {}
        self.trend_analysis = {}
        
    def calculate_momentum_weights(self, historical_data: pd.DataFrame, 
                                  current_week: int) -> Dict:
        """Calculate momentum-based weights for each player"""
        print("Calculating momentum-based time weights...")
        
        momentum_weights = {}
        
        for player in historical_data['player'].unique():
            if pd.isna(player):
                continue
                
            player_data = historical_data[historical_data['player'] == player].copy()
            player_data = player_data.sort_values(['year', 'week'], ascending=False)
            
            if len(player_data) < 3:  # Need minimum data for momentum
                continue
            
            # Get recent games (last 3) vs previous games (4-6)
            recent_games = player_data.head(3)
            previous_games = player_data.iloc[3:6] if len(player_data) >= 6 else player_data.iloc[3:]
            
            if len(recent_games) == 0 or len(previous_games) == 0:
                continue
            
            # Calculate momentum for each stat
            momentum_factors = {}
            stats = ['pass_yds', 'rush_yds', 'rec_yds', 'pass_tds', 'rush_tds', 'rec_tds', 'snap_pct']
            
            for stat in stats:
                if stat in player_data.columns:
                    recent_avg = recent_games[stat].mean()
                    previous_avg = previous_games[stat].mean()
                    
                    if previous_avg > 0:
                        momentum = recent_avg / previous_avg
                        # Clamp momentum to reasonable range
                        momentum = max(0.5, min(2.0, momentum))
                        momentum_factors[stat] = momentum
                    else:
                        momentum_factors[stat] = 1.0
            
            # Calculate overall momentum
            if momentum_factors:
                overall_momentum = np.mean(list(momentum_factors.values()))
                momentum_weights[player] = {
                    'factors': momentum_factors,
                    'overall_momentum': overall_momentum,
                    'recent_games': len(recent_games),
                    'previous_games': len(previous_games)
                }
        
        print(f"Calculated momentum weights for {len(momentum_weights)} players")
        return momentum_weights
    
    def calculate_context_weights(self, historical_data: pd.DataFrame) -> Dict:
        """Calculate context-aware weights based on opponent strength"""
        print("Calculating context-aware weights...")
        
        context_weights = {}
        
        for player in historical_data['player'].unique():
            if pd.isna(player):
                continue
                
            player_data = historical_data[historical_data['player'] == player].copy()
            
            if len(player_data) < 5:  # Need minimum data for context
                continue
            
            # Calculate opponent strength for each game
            opponent_strengths = {}
            
            for _, game in player_data.iterrows():
                opponent = game['opponent']
                week = game['week']
                year = game['year']
                
                if pd.isna(opponent):
                    continue
                
                # Get opponent's defensive performance in that season
                opp_season_data = historical_data[
                    (historical_data['year'] == year) & 
                    (historical_data['opponent'] == opponent)
                ]
                
                if len(opp_season_data) > 0:
                    # Calculate opponent's defensive strength
                    def_stats = {
                        'pass_yds_allowed': opp_season_data['pass_yds'].mean() if 'pass_yds' in opp_season_data.columns else 0,
                        'rush_yds_allowed': opp_season_data['rush_yds'].mean() if 'rush_yds' in opp_season_data.columns else 0,
                        'rec_yds_allowed': opp_season_data['rec_yds'].mean() if 'rec_yds' in opp_season_data.columns else 0
                    }
                    
                    # Calculate strength factor (lower allowed = stronger defense)
                    league_avg_pass = historical_data['pass_yds'].mean() if 'pass_yds' in historical_data.columns else 250
                    league_avg_rush = historical_data['rush_yds'].mean() if 'rush_yds' in historical_data.columns else 100
                    league_avg_rec = historical_data['rec_yds'].mean() if 'rec_yds' in historical_data.columns else 150
                    
                    pass_strength = league_avg_pass / def_stats['pass_yds_allowed'] if def_stats['pass_yds_allowed'] > 0 else 1.0
                    rush_strength = league_avg_rush / def_stats['rush_yds_allowed'] if def_stats['rush_yds_allowed'] > 0 else 1.0
                    rec_strength = league_avg_rec / def_stats['rec_yds_allowed'] if def_stats['rec_yds_allowed'] > 0 else 1.0
                    
                    # Overall opponent strength
                    overall_strength = (pass_strength + rush_strength + rec_strength) / 3
                    opponent_strengths[f"{year}_{week}"] = overall_strength
            
            if opponent_strengths:
                # Calculate context weights based on opponent strength
                context_factors = {}
                
                for stat in ['pass_yds', 'rush_yds', 'rec_yds', 'pass_tds', 'rush_tds', 'rec_tds']:
                    if stat in player_data.columns:
                        # Weight games by opponent strength
                        weighted_performance = []
                        weights = []
                        
                        for _, game in player_data.iterrows():
                            game_key = f"{game['year']}_{game['week']}"
                            if game_key in opponent_strengths:
                                strength = opponent_strengths[game_key]
                                performance = game[stat]
                                
                                # Stronger opponent = higher weight for good performance
                                if performance > 0:
                                    weight = strength
                                    weighted_performance.append(performance * weight)
                                    weights.append(weight)
                        
                        if weights:
                            context_factors[stat] = {
                                'weighted_avg': np.average(weighted_performance, weights=weights),
                                'opponent_strength_avg': np.mean(list(opponent_strengths.values())),
                                'games_analyzed': len(weights)
                            }
                
                context_weights[player] = {
                    'factors': context_factors,
                    'opponent_strengths': opponent_strengths
                }
        
        print(f"Calculated context weights for {len(context_weights)} players")
        return context_weights
    
    def calculate_trend_analysis(self, historical_data: pd.DataFrame) -> Dict:
        """Calculate trend analysis for each player"""
        print("Calculating trend analysis...")
        
        trend_analysis = {}
        
        for player in historical_data['player'].unique():
            if pd.isna(player):
                continue
                
            player_data = historical_data[historical_data['player'] == player].copy()
            player_data = player_data.sort_values(['year', 'week'])
            
            if len(player_data) < 8:  # Need minimum data for trend analysis
                continue
            
            # Calculate rolling averages and trends
            stats = ['pass_yds', 'rush_yds', 'rec_yds', 'pass_tds', 'rush_tds', 'rec_tds', 'snap_pct']
            trends = {}
            
            for stat in stats:
                if stat in player_data.columns:
                    # Calculate 4-game rolling average
                    player_data[f'{stat}_rolling'] = player_data[stat].rolling(window=4, min_periods=2).mean()
                    
                    # Calculate trend (slope of rolling average)
                    if len(player_data) >= 4:
                        x = np.arange(len(player_data))
                        y = player_data[f'{stat}_rolling'].fillna(0)
                        
                        if len(y[y > 0]) >= 2:  # Need at least 2 non-zero values
                            # Calculate linear trend
                            slope = np.polyfit(x, y, 1)[0]
                            trends[stat] = {
                                'slope': slope,
                                'trend_direction': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
                                'recent_avg': player_data[stat].tail(4).mean(),
                                'overall_avg': player_data[stat].mean()
                            }
            
            if trends:
                # Calculate overall trend
                slopes = [trend['slope'] for trend in trends.values()]
                overall_slope = np.mean(slopes)
                
                trend_analysis[player] = {
                    'individual_trends': trends,
                    'overall_slope': overall_slope,
                    'overall_direction': 'improving' if overall_slope > 0 else 'declining' if overall_slope < 0 else 'stable',
                    'games_analyzed': len(player_data)
                }
        
        print(f"Calculated trend analysis for {len(trend_analysis)} players")
        return trend_analysis
    
    def apply_advanced_weighting(self, projections: pd.DataFrame, 
                                historical_data: pd.DataFrame,
                                momentum_weights: Dict, context_weights: Dict,
                                trend_analysis: Dict) -> pd.DataFrame:
        """Apply advanced time weighting to projections"""
        print("Applying advanced time weighting to projections...")
        
        enhanced_projections = projections.copy()
        
        # Add weighting columns
        enhanced_projections['momentum_multiplier'] = 1.0
        enhanced_projections['context_multiplier'] = 1.0
        enhanced_projections['trend_multiplier'] = 1.0
        enhanced_projections['overall_weighting'] = 1.0
        
        for idx, row in enhanced_projections.iterrows():
            player_name = row.get('name', '')
            
            if not player_name:
                continue
            
            # Apply momentum weighting
            if player_name in momentum_weights:
                momentum_data = momentum_weights[player_name]
                momentum_multiplier = momentum_data['overall_momentum']
                
                # Apply momentum to relevant stats
                for stat, factor in momentum_data['factors'].items():
                    proj_stat = stat.replace('_yds', '_yd').replace('_tds', '_td')
                    proj_col = f'proj_{proj_stat}'
                    
                    if proj_col in enhanced_projections.columns:
                        enhanced_projections.loc[idx, proj_col] *= factor
                
                enhanced_projections.loc[idx, 'momentum_multiplier'] = momentum_multiplier
            
            # Apply context weighting
            if player_name in context_weights:
                context_data = context_weights[player_name]
                
                # Apply context factors to relevant stats
                for stat, factor_data in context_data['factors'].items():
                    proj_stat = stat.replace('_yds', '_yd').replace('_tds', '_td')
                    proj_col = f'proj_{proj_stat}'
                    
                    if proj_col in enhanced_projections.columns:
                        # Use opponent strength to adjust projections
                        opp_strength = factor_data['opponent_strength_avg']
                        context_multiplier = 1 + (opp_strength - 1) * 0.3  # Scale down impact
                        context_multiplier = max(0.8, min(1.2, context_multiplier))
                        
                        enhanced_projections.loc[idx, proj_col] *= context_multiplier
                
                # Overall context multiplier
                overall_context = np.mean([data['opponent_strength_avg'] for data in context_data['factors'].values()])
                context_multiplier = 1 + (overall_context - 1) * 0.2
                context_multiplier = max(0.9, min(1.1, context_multiplier))
                enhanced_projections.loc[idx, 'context_multiplier'] = context_multiplier
            
            # Apply trend analysis
            if player_name in trend_analysis:
                trend_data = trend_analysis[player_name]
                overall_slope = trend_data['overall_slope']
                
                # Apply trend to relevant stats
                for stat, trend_info in trend_data['individual_trends'].items():
                    proj_stat = stat.replace('_yds', '_yd').replace('_tds', '_td')
                    proj_col = f'proj_{proj_stat}'
                    
                    if proj_col in enhanced_projections.columns:
                        slope = trend_info['slope']
                        trend_multiplier = 1 + (slope * 0.1)  # Scale down impact
                        trend_multiplier = max(0.9, min(1.1, trend_multiplier))
                        
                        enhanced_projections.loc[idx, proj_col] *= trend_multiplier
                
                # Overall trend multiplier
                trend_multiplier = 1 + (overall_slope * 0.05)
                trend_multiplier = max(0.95, min(1.05, trend_multiplier))
                enhanced_projections.loc[idx, 'trend_multiplier'] = trend_multiplier
            
            # Calculate overall weighting
            momentum = enhanced_projections.loc[idx, 'momentum_multiplier']
            context = enhanced_projections.loc[idx, 'context_multiplier']
            trend = enhanced_projections.loc[idx, 'trend_multiplier']
            
            overall_weighting = (momentum + context + trend) / 3
            enhanced_projections.loc[idx, 'overall_weighting'] = overall_weighting
        
        print(f"Applied advanced weighting to {len(enhanced_projections)} players")
        return enhanced_projections
    
    def run_advanced_weighting(self, projections: pd.DataFrame, 
                             historical_data: pd.DataFrame) -> pd.DataFrame:
        """Run complete advanced time weighting analysis"""
        print("Starting advanced time weighting analysis...")
        
        # Calculate momentum weights
        momentum_weights = self.calculate_momentum_weights(historical_data, 1)
        
        # Calculate context weights
        context_weights = self.calculate_context_weights(historical_data)
        
        # Calculate trend analysis
        trend_analysis = self.calculate_trend_analysis(historical_data)
        
        # Apply advanced weighting
        enhanced_projections = self.apply_advanced_weighting(
            projections, historical_data, momentum_weights, 
            context_weights, trend_analysis
        )
        
        print("Advanced time weighting analysis complete")
        return enhanced_projections
