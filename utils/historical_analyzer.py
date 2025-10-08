"""
Multi-Season Historical Analysis for NFL Projections

This module implements advanced historical analysis using 2022-2025 data
to improve projection accuracy through better historical context.

Key Features:
- Player career trajectory analysis
- Team performance evolution tracking
- Opponent-specific historical matchups
- Multi-season trend analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional

class HistoricalAnalyzer:
    def __init__(self):
        self.historical_data = {}
        self.player_trajectories = {}
        self.team_evolution = {}
        self.opponent_matchups = {}
        
    def load_multi_season_data(self) -> pd.DataFrame:
        """Load and combine historical data from 2022-2025"""
        print("Loading multi-season historical data...")
        
        historical_files = [
            'data/game_data/game_data_2022.csv',
            'data/game_data/game_data_2023.csv', 
            'data/game_data/game_data_2024.csv',
            'data/game_data/game_data_2025.csv'
        ]
        
        all_data = []
        
        for file_path in historical_files:
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    print(f"Loaded {len(df)} records from {file_path}")
                    all_data.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
            else:
                print(f"Warning: File not found: {file_path}")
        
        if not all_data:
            raise ValueError("No historical data files found")
        
        # Combine all historical data
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Total historical records: {len(combined_data)}")
        
        # Add season identifier
        combined_data['season'] = combined_data['year']
        
        return combined_data
    
    def analyze_player_trajectories(self, historical_data: pd.DataFrame) -> Dict:
        """Analyze player career trajectories and performance trends"""
        print("Analyzing player career trajectories...")
        
        trajectories = {}
        
        for player in historical_data['player'].unique():
            if pd.isna(player):
                continue
                
            player_data = historical_data[historical_data['player'] == player].copy()
            player_data = player_data.sort_values(['year', 'week'])
            
            if len(player_data) < 5:  # Need minimum data for trajectory
                continue
            
            # Calculate season-by-season averages
            season_stats = {}
            for year in player_data['year'].unique():
                year_data = player_data[player_data['year'] == year]
                
                season_avg = {
                    'games_played': len(year_data),
                    'pass_yd': year_data['pass_yds'].mean() if 'pass_yds' in year_data.columns else 0,
                    'rush_yd': year_data['rush_yds'].mean() if 'rush_yds' in year_data.columns else 0,
                    'rec_yd': year_data['rec_yds'].mean() if 'rec_yds' in year_data.columns else 0,
                    'pass_td': year_data['pass_tds'].mean() if 'pass_tds' in year_data.columns else 0,
                    'rush_td': year_data['rush_tds'].mean() if 'rush_tds' in year_data.columns else 0,
                    'rec_td': year_data['rec_tds'].mean() if 'rec_tds' in year_data.columns else 0,
                    'snap_pct': year_data['snap_pct'].mean() if 'snap_pct' in year_data.columns else 0
                }
                season_stats[year] = season_avg
            
            # Calculate trajectory trends
            if len(season_stats) >= 2:
                years = sorted(season_stats.keys())
                recent_year = years[-1]
                previous_year = years[-2] if len(years) > 1 else years[0]
                
                # Calculate year-over-year changes
                trajectory = {
                    'recent_season': recent_year,
                    'recent_stats': season_stats[recent_year],
                    'previous_stats': season_stats[previous_year],
                    'trends': {}
                }
                
                # Calculate trend for each stat
                for stat in ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td', 'snap_pct']:
                    recent_val = season_stats[recent_year][stat]
                    previous_val = season_stats[previous_year][stat]
                    
                    if previous_val > 0:
                        trend = (recent_val - previous_val) / previous_val
                        trajectory['trends'][stat] = trend
                    else:
                        trajectory['trends'][stat] = 0
                
                # Determine if player is trending up, down, or stable
                overall_trend = np.mean(list(trajectory['trends'].values()))
                if overall_trend > 0.1:
                    trajectory['direction'] = 'improving'
                elif overall_trend < -0.1:
                    trajectory['direction'] = 'declining'
                else:
                    trajectory['direction'] = 'stable'
                
                trajectories[player] = trajectory
        
        print(f"Analyzed trajectories for {len(trajectories)} players")
        return trajectories
    
    def analyze_team_evolution(self, historical_data: pd.DataFrame) -> Dict:
        """Analyze team performance evolution over time"""
        print("Analyzing team performance evolution...")
        
        team_evolution = {}
        
        for team in historical_data['team'].unique():
            if pd.isna(team):
                continue
                
            team_data = historical_data[historical_data['team'] == team].copy()
            
            # Calculate team-level stats by season
            season_performance = {}
            for year in team_data['year'].unique():
                year_data = team_data[team_data['year'] == year]
                
                # Aggregate team stats
                team_stats = {
                    'games_played': len(year_data['week'].unique()),
                    'avg_pass_yd': year_data['pass_yds'].mean() if 'pass_yds' in year_data.columns else 0,
                    'avg_rush_yd': year_data['rush_yds'].mean() if 'rush_yds' in year_data.columns else 0,
                    'avg_rec_yd': year_data['rec_yds'].mean() if 'rec_yds' in year_data.columns else 0,
                    'avg_pass_td': year_data['pass_tds'].mean() if 'pass_tds' in year_data.columns else 0,
                    'avg_rush_td': year_data['rush_tds'].mean() if 'rush_tds' in year_data.columns else 0,
                    'avg_rec_td': year_data['rec_tds'].mean() if 'rec_tds' in year_data.columns else 0
                }
                season_performance[year] = team_stats
            
            # Calculate evolution trends
            if len(season_performance) >= 2:
                years = sorted(season_performance.keys())
                recent_year = years[-1]
                previous_year = years[-2] if len(years) > 1 else years[0]
                
                evolution = {
                    'recent_season': recent_year,
                    'recent_performance': season_performance[recent_year],
                    'previous_performance': season_performance[previous_year],
                    'evolution_trends': {}
                }
                
                # Calculate evolution for each stat
                for stat in ['avg_pass_yd', 'avg_rush_yd', 'avg_rec_yd', 'avg_pass_td', 'avg_rush_td', 'avg_rec_td']:
                    recent_val = season_performance[recent_year][stat]
                    previous_val = season_performance[previous_year][stat]
                    
                    if previous_val > 0:
                        evolution_rate = (recent_val - previous_val) / previous_val
                        evolution['evolution_trends'][stat] = evolution_rate
                    else:
                        evolution['evolution_trends'][stat] = 0
                
                team_evolution[team] = evolution
        
        print(f"Analyzed evolution for {len(team_evolution)} teams")
        return team_evolution
    
    def analyze_opponent_matchups(self, historical_data: pd.DataFrame) -> Dict:
        """Analyze historical opponent-specific matchups"""
        print("Analyzing opponent-specific matchups...")
        
        opponent_matchups = {}
        
        for player in historical_data['player'].unique():
            if pd.isna(player):
                continue
                
            player_data = historical_data[historical_data['player'] == player].copy()
            
            # Group by opponent
            opponent_stats = {}
            for opponent in player_data['opponent'].unique():
                if pd.isna(opponent):
                    continue
                    
                opp_data = player_data[player_data['opponent'] == opponent]
                
                if len(opp_data) >= 2:  # Need multiple games vs opponent
                    opp_performance = {
                        'games_vs_opponent': len(opp_data),
                        'avg_pass_yd': opp_data['pass_yds'].mean() if 'pass_yds' in opp_data.columns else 0,
                        'avg_rush_yd': opp_data['rush_yds'].mean() if 'rush_yds' in opp_data.columns else 0,
                        'avg_rec_yd': opp_data['rec_yds'].mean() if 'rec_yds' in opp_data.columns else 0,
                        'avg_pass_td': opp_data['pass_tds'].mean() if 'pass_tds' in opp_data.columns else 0,
                        'avg_rush_td': opp_data['rush_tds'].mean() if 'rush_tds' in opp_data.columns else 0,
                        'avg_rec_td': opp_data['rec_tds'].mean() if 'rec_tds' in opp_data.columns else 0,
                        'avg_snap_pct': opp_data['snap_pct'].mean() if 'snap_pct' in opp_data.columns else 0
                    }
                    
                    # Calculate vs overall average
                    overall_avg = {
                        'avg_pass_yd': player_data['pass_yds'].mean() if 'pass_yds' in player_data.columns else 0,
                        'avg_rush_yd': player_data['rush_yds'].mean() if 'rush_yds' in player_data.columns else 0,
                        'avg_rec_yd': player_data['rec_yds'].mean() if 'rec_yds' in player_data.columns else 0,
                        'avg_pass_td': player_data['pass_tds'].mean() if 'pass_tds' in player_data.columns else 0,
                        'avg_rush_td': player_data['rush_tds'].mean() if 'rush_tds' in player_data.columns else 0,
                        'avg_rec_td': player_data['rec_tds'].mean() if 'rec_tds' in player_data.columns else 0,
                        'avg_snap_pct': player_data['snap_pct'].mean() if 'snap_pct' in player_data.columns else 0
                    }
                    
                    # Calculate opponent-specific multipliers
                    multipliers = {}
                    for stat in ['avg_pass_yd', 'avg_rush_yd', 'avg_rec_yd', 'avg_pass_td', 'avg_rush_td', 'avg_rec_td', 'avg_snap_pct']:
                        opp_val = opp_performance[stat]
                        overall_val = overall_avg[stat]
                        
                        if overall_val > 0:
                            multiplier = opp_val / overall_val
                            # Clamp to reasonable range
                            multiplier = max(0.3, min(3.0, multiplier))
                            multipliers[stat] = multiplier
                        else:
                            multipliers[stat] = 1.0
                    
                    opponent_stats[opponent] = {
                        'performance': opp_performance,
                        'multipliers': multipliers
                    }
            
            if opponent_stats:
                opponent_matchups[player] = opponent_stats
        
        print(f"Analyzed matchups for {len(opponent_matchups)} players")
        return opponent_matchups
    
    def apply_historical_insights(self, projections: pd.DataFrame, 
                                 trajectories: Dict, team_evolution: Dict, 
                                 opponent_matchups: Dict) -> pd.DataFrame:
        """Apply historical insights to projections"""
        print("Applying historical insights to projections...")
        
        enhanced_projections = projections.copy()
        
        # Add historical insight columns
        enhanced_projections['trajectory_direction'] = 'stable'
        enhanced_projections['trajectory_multiplier'] = 1.0
        enhanced_projections['opponent_multiplier'] = 1.0
        enhanced_projections['team_evolution_multiplier'] = 1.0
        
        for idx, row in enhanced_projections.iterrows():
            player_name = row.get('name', '')
            team = row.get('team', '')
            opponent = row.get('next_op', '')
            
            if not player_name:
                continue
            
            # Apply player trajectory insights
            if player_name in trajectories:
                trajectory = trajectories[player_name]
                direction = trajectory['direction']
                trends = trajectory['trends']
                
                enhanced_projections.loc[idx, 'trajectory_direction'] = direction
                
                # Apply trajectory multipliers to relevant stats
                for stat, trend in trends.items():
                    if trend != 0:
                        # Convert trend to multiplier (trend of 0.2 = 1.2x multiplier)
                        multiplier = 1 + (trend * 0.5)  # Scale down the impact
                        multiplier = max(0.7, min(1.3, multiplier))  # Clamp to reasonable range
                        
                        proj_col = f'proj_{stat}'
                        if proj_col in enhanced_projections.columns:
                            enhanced_projections.loc[idx, proj_col] *= multiplier
                
                # Overall trajectory multiplier
                overall_trend = np.mean(list(trends.values()))
                trajectory_multiplier = 1 + (overall_trend * 0.3)
                trajectory_multiplier = max(0.8, min(1.2, trajectory_multiplier))
                enhanced_projections.loc[idx, 'trajectory_multiplier'] = trajectory_multiplier
            
            # Apply opponent-specific matchups
            if player_name in opponent_matchups and opponent in opponent_matchups[player_name]:
                opp_data = opponent_matchups[player_name][opponent]
                multipliers = opp_data['multipliers']
                
                # Apply opponent-specific multipliers
                for stat, multiplier in multipliers.items():
                    proj_stat = stat.replace('avg_', 'proj_')
                    if proj_stat in enhanced_projections.columns:
                        enhanced_projections.loc[idx, proj_stat] *= multiplier
                
                # Overall opponent multiplier
                overall_opp_multiplier = np.mean(list(multipliers.values()))
                enhanced_projections.loc[idx, 'opponent_multiplier'] = overall_opp_multiplier
            
            # Apply team evolution insights
            if team in team_evolution:
                evolution = team_evolution[team]
                evolution_trends = evolution['evolution_trends']
                
                # Apply team evolution to relevant stats
                for stat, evolution_rate in evolution_trends.items():
                    proj_stat = stat.replace('avg_', 'proj_')
                    if proj_stat in enhanced_projections.columns:
                        multiplier = 1 + (evolution_rate * 0.2)  # Scale down impact
                        multiplier = max(0.9, min(1.1, multiplier))
                        enhanced_projections.loc[idx, proj_stat] *= multiplier
                
                # Overall team evolution multiplier
                overall_evolution = np.mean(list(evolution_trends.values()))
                team_multiplier = 1 + (overall_evolution * 0.1)
                team_multiplier = max(0.95, min(1.05, team_multiplier))
                enhanced_projections.loc[idx, 'team_evolution_multiplier'] = team_multiplier
        
        print(f"Applied historical insights to {len(enhanced_projections)} players")
        return enhanced_projections
    
    def run_historical_analysis(self, projections: pd.DataFrame) -> pd.DataFrame:
        """Run complete historical analysis and apply insights"""
        print("Starting multi-season historical analysis...")
        
        # Load historical data
        historical_data = self.load_multi_season_data()
        
        # Analyze player trajectories
        trajectories = self.analyze_player_trajectories(historical_data)
        
        # Analyze team evolution
        team_evolution = self.analyze_team_evolution(historical_data)
        
        # Analyze opponent matchups
        opponent_matchups = self.analyze_opponent_matchups(historical_data)
        
        # Apply insights to projections
        enhanced_projections = self.apply_historical_insights(
            projections, trajectories, team_evolution, opponent_matchups
        )
        
        print("Multi-season historical analysis complete")
        return enhanced_projections
