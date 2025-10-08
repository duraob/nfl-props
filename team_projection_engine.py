"""
Team Projection Engine - NFL Team-Level Predictions

Generates team-level projections using the same methodology as the player projection engine:
- 10-game sample with time decay
- Schedule strength analysis
- Opponent strength analysis
- Monte Carlo simulations for team outcomes

This module provides team-level win/loss predictions, standings, and playoff probabilities.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime as dt


class TeamDataLoader:
    """
    Team-level data loader that mirrors the projection engine's data loading logic.
    
    Key responsibilities:
    1. Load team data using 10-game sample + time decay
    2. Apply time weighting to team performance
    3. Create team performance baselines
    """
    
    def __init__(self):
        self.team_data: pd.DataFrame = pd.DataFrame()
        
    def load_time_weighted_team_data(self, week_2025_file: str, weeks_2024_file: str, 
                                   target_weeks_2024: list, projection_week: int, 
                                   decay_coefficient: float = 0.7) -> pd.DataFrame:
        """
        Load team data using the same 10-game sample + time decay approach as player engine.
        
        Args:
            week_2025_file: Path to 2025 season data
            weeks_2024_file: Path to 2024 season data
            target_weeks_2024: List of 2024 weeks to include
            projection_week: Current week for projections
            decay_coefficient: Time decay coefficient (same as player engine)
            
        Returns:
            DataFrame with time-weighted team data
        """
        print(f"Loading team data for projection week {projection_week}...")
        
        # Load 2025 data
        df_2025 = pd.read_csv(week_2025_file) if os.path.exists(week_2025_file) else pd.DataFrame()
        
        # Load 2024 data
        df_2024 = pd.read_csv(weeks_2024_file) if os.path.exists(weeks_2024_file) else pd.DataFrame()
        
        # Determine data selection based on projection week (SAME LOGIC AS PLAYER ENGINE)
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
        
        # Apply time weights (SAME DECAY LOGIC AS PLAYER ENGINE)
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
            df_combined = pd.DataFrame()
        
        print(f"Loaded team data: {len(df_combined)} team-game records")
        return df_combined
    
    
    def extract_team_performance_baseline(self, df_team_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract team performance baselines using time-weighted data.
        
        Args:
            df_team_data: Time-weighted team data
            
        Returns:
            DataFrame with team performance baselines
        """
        print("Extracting team performance baselines...")
        
        # Check if time_weight column exists, if not use equal weights
        if 'time_weight' not in df_team_data.columns:
            df_team_data['time_weight'] = 1.0
        
        # Group by team and calculate weighted averages
        team_stats = df_team_data.groupby('team').apply(
            lambda group: pd.Series({
                'points_per_game': np.average(group['team_score'], weights=group['time_weight']),
                'points_allowed_per_game': np.average(group['opp_score'], weights=group['time_weight']),
                'pass_yards_per_game': np.average(group['pass_yds'], weights=group['time_weight']),
                'rush_yards_per_game': np.average(group['rush_yds'], weights=group['time_weight']),
                'total_yards_per_game': np.average(group['pass_yds'] + group['rush_yds'], weights=group['time_weight']),
                'pass_tds_per_game': np.average(group['pass_tds'], weights=group['time_weight']),
                'rush_tds_per_game': np.average(group['rush_tds'], weights=group['time_weight']),
                'turnovers_per_game': np.average(group['pass_int'] + group['fumbles'], weights=group['time_weight']),
                'sacks_per_game': np.average(group['sacks'], weights=group['time_weight']),
                'games_played': len(group['week'].unique()),
                'opps': list(group['opponent'].unique()),  # Add opponents list
                'home_away_split': self._calculate_home_away_split(group)
            })
        ).reset_index()
        
        print(f"Extracted baselines for {len(team_stats)} teams")
        return team_stats
    
    def _calculate_home_away_split(self, team_group: pd.DataFrame) -> Dict:
        """
        Calculate home/away performance split for a team.
        
        Args:
            team_group: Team's game data
            
        Returns:
            Dictionary with home/away performance metrics
        """
        home_data = team_group[team_group['home_away'] == 'home']
        away_data = team_group[team_group['home_away'] == 'away']
        
        if home_data.empty or away_data.empty:
            return {'home_advantage': 0.0, 'home_points': 0.0, 'away_points': 0.0}
        
        home_points = np.average(home_data['team_score'], weights=home_data['time_weight'])
        away_points = np.average(away_data['team_score'], weights=away_data['time_weight'])
        
        return {
            'home_advantage': home_points - away_points,
            'home_points': home_points,
            'away_points': away_points
        }


class TeamScheduleAnalyzer:
    """
    Team schedule strength analyzer (mirrors player schedule analyzer).
    
    Key responsibilities:
    1. Calculate team schedule strength ratios
    2. Normalize team statistics against league averages
    3. Apply schedule strength adjustments to projections
    """
    
    def __init__(self):
        self.team_schedule_strength: Dict[str, Dict] = {}
        self.league_averages: Dict[str, float] = {}
        
    def calculate_team_schedule_strength(self, df_team: pd.DataFrame, schedule_file: str) -> pd.DataFrame:
        """
        Calculate team schedule strength using same methodology as player engine.
        
        Args:
            df_team: Team statistics DataFrame
            schedule_file: Path to schedule file
            
        Returns:
            DataFrame with schedule strength ratios
        """
        print("Calculating team schedule strength...")
        
        # Load schedule
        schedule = pd.read_csv(schedule_file)
        
        # Calculate league averages
        self.league_averages = self._calculate_league_averages(df_team)
        
        # Calculate team ratios to league average
        df_team_ratios = self._calculate_team_ratios(df_team)
        
        # Calculate schedule strength for each team
        df_schedule_strength = self._calculate_schedule_strength(df_team_ratios, schedule)
        
        # Reset index to get team column back
        df_schedule_strength.reset_index(inplace=True)
        
        print(f"Calculated schedule strength for {len(df_schedule_strength)} teams")
        return df_schedule_strength
    
    def _calculate_league_averages(self, df_team: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate league averages for team statistics.
        
        Args:
            df_team: Team statistics DataFrame
            
        Returns:
            Dictionary with league averages
        """
        return {
            'points_per_game': df_team['points_per_game'].mean(),
            'points_allowed_per_game': df_team['points_allowed_per_game'].mean(),
            'pass_yards_per_game': df_team['pass_yards_per_game'].mean(),
            'rush_yards_per_game': df_team['rush_yards_per_game'].mean(),
            'total_yards_per_game': df_team['total_yards_per_game'].mean()
        }
    
    def _calculate_team_ratios(self, df_team: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate team ratios to league average.
        
        Args:
            df_team: Team statistics DataFrame
            
        Returns:
            DataFrame with team ratios
        """
        df_ratios = df_team.copy()
        
        # Set team as index for proper lookup
        df_ratios.set_index('team', inplace=True)
        
        # Calculate ratios to league average
        for stat, league_avg in self.league_averages.items():
            if league_avg > 0:
                df_ratios[f'{stat}_ratio'] = df_ratios[stat] / league_avg
            else:
                df_ratios[f'{stat}_ratio'] = 1.0
        
        return df_ratios
    
    def _calculate_schedule_strength(self, df_team_ratios: pd.DataFrame, 
                                    schedule: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate schedule strength for each team using same logic as player engine.
        
        Args:
            df_team_ratios: Team ratios DataFrame
            schedule: Schedule DataFrame
            
        Returns:
            DataFrame with schedule strength
        """
        df_schedule = df_team_ratios.copy()
        
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


class TeamVarianceAnalyzer:
    """
    Team variance analyzer (mirrors player variance analyzer).
    
    Key responsibilities:
    1. Analyze historical team performance variance
    2. Calculate team-specific volatility
    3. Generate variance adjustments for projections
    """
    
    def __init__(self):
        self.team_variance: Dict[str, Dict] = {}
        
    def analyze_team_variance(self, df_team_data: pd.DataFrame, 
                            df_team_projections: pd.DataFrame) -> Dict:
        """
        Analyze team performance variance using same logic as player engine.
        
        Args:
            df_team_data: Historical team data
            df_team_projections: Team projections DataFrame
            
        Returns:
            Dictionary with team variance metrics
        """
        print("Analyzing team performance variance...")
        
        team_variance = {}
        for team in df_team_projections['team'].unique():
            team_games = df_team_data[df_team_data['team'] == team]
            
            if team_games.empty:
                continue
                
            # Calculate variance for key metrics
            variance_metrics = {
                'points_variance': team_games['team_score'].var(),
                'points_allowed_variance': team_games['opp_score'].var(),
                'pass_yards_variance': team_games['pass_yds'].var(),
                'rush_yards_variance': team_games['rush_yds'].var(),
                'total_yards_variance': (team_games['pass_yds'] + team_games['rush_yds']).var()
            }
            
            # Calculate coefficient of variation (CV = std/mean)
            team_proj = df_team_projections[df_team_projections['team'] == team].iloc[0]
            cv_metrics = {}
            
            for metric, variance in variance_metrics.items():
                if variance > 0:
                    base_metric = metric.replace('_variance', '_per_game')
                    if base_metric in team_proj:
                        mean_val = team_proj[base_metric]
                        if mean_val > 0:
                            cv_metrics[metric.replace('_variance', '_cv')] = np.sqrt(variance) / mean_val
            
            team_variance[team] = {
                'variance_metrics': variance_metrics,
                'cv_metrics': cv_metrics,
                'volatility_score': np.mean(list(cv_metrics.values())) if cv_metrics else 1.0
            }
        
        print(f"Calculated variance for {len(team_variance)} teams")
        return team_variance


class TeamProbabilityEngine:
    """
    Team probability engine (mirrors player probability engine).
    
    Key responsibilities:
    1. Generate probability distributions for team outcomes
    2. Calculate confidence intervals
    3. Create risk assessments
    """
    
    def __init__(self):
        self.probability_distributions: Dict[str, Dict] = {}
        
    def generate_team_probability_distributions(self, simulation_results: Dict) -> Dict:
        """
        Generate probability distributions for team outcomes.
        
        Args:
            simulation_results: Team simulation results
            
        Returns:
            Dictionary with probability distributions
        """
        print("Generating team probability distributions...")
        
        probability_distributions = {}
        
        for team, team_sims in simulation_results.items():
            if 'game_simulations' not in team_sims:
                continue
                
            # Calculate win probability
            total_wins = sum(sim['wins'] for sim in team_sims['game_simulations'])
            total_simulations = len(team_sims['game_simulations']) * 1000  # Assuming 1000 sims per game
            
            win_probability = total_wins / total_simulations if total_simulations > 0 else 0
            
            # Calculate confidence intervals
            points_scored = [sim['points_scored'] for sim in team_sims['game_simulations']]
            points_allowed = [sim['points_allowed'] for sim in team_sims['game_simulations']]
            
            probability_distributions[team] = {
                'win_probability': win_probability,
                'points_scored_mean': np.mean(points_scored),
                'points_scored_std': np.std(points_scored),
                'points_allowed_mean': np.mean(points_allowed),
                'points_allowed_std': np.std(points_allowed),
                'confidence_intervals': self._calculate_confidence_intervals(points_scored, points_allowed)
            }
        
        print(f"Generated probability distributions for {len(probability_distributions)} teams")
        return probability_distributions
    
    def _calculate_confidence_intervals(self, points_scored: List[float], 
                                     points_allowed: List[float]) -> Dict:
        """
        Calculate confidence intervals for team statistics.
        
        Args:
            points_scored: List of points scored values
            points_allowed: List of points allowed values
            
        Returns:
            Dictionary with confidence intervals
        """
        if not points_scored or not points_allowed:
            return {}
        
        # Calculate 95% confidence intervals
        scored_ci = np.percentile(points_scored, [2.5, 97.5])
        allowed_ci = np.percentile(points_allowed, [2.5, 97.5])
        
        return {
            'points_scored_95ci': scored_ci,
            'points_allowed_95ci': allowed_ci,
            'points_scored_range': scored_ci[1] - scored_ci[0],
            'points_allowed_range': allowed_ci[1] - allowed_ci[0]
        }


class TeamSimulationEngine:
    """
    Team simulation engine (mirrors player simulation engine).
    
    Key responsibilities:
    1. Run Monte Carlo simulations for team outcomes
    2. Generate probability distributions
    3. Calculate confidence intervals
    """
    
    def __init__(self, n_simulations: int = 1000, random_seed: int = 42):
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def run_team_simulations(self, df_team_projections: pd.DataFrame, 
                           team_variance: Dict, 
                           game_predictions: pd.DataFrame) -> Dict:
        """
        Run Monte Carlo simulations for team outcomes.
        
        Args:
            df_team_projections: Team projections DataFrame
            team_variance: Team variance metrics
            game_predictions: Game predictions DataFrame
            
        Returns:
            Dictionary with simulation results
        """
        print(f"Running {self.n_simulations:,} Monte Carlo simulations for team outcomes...")
        
        simulation_results = {}
        
        for team in df_team_projections['team'].unique():
            team_proj = df_team_projections[df_team_projections['team'] == team].iloc[0]
            team_var = team_variance.get(team, {})
            
            # Get team's remaining games
            team_games = game_predictions[
                (game_predictions['home_team'] == team) | 
                (game_predictions['away_team'] == team)
            ]
            
            if team_games.empty:
                continue
            
            # Run simulations for each game
            game_simulations = []
            for _, game in team_games.iterrows():
                is_home = game['home_team'] == team
                opponent = game['away_team'] if is_home else game['home_team']
                
                # Get opponent projections
                opp_proj = df_team_projections[df_team_projections['team'] == opponent]
                if opp_proj.empty:
                    continue
                opp_proj = opp_proj.iloc[0]
                
                # Simulate game outcomes
                game_sims = self._simulate_game_outcome(
                    team_proj, opp_proj, is_home, team_var
                )
                game_simulations.append(game_sims)
            
            # Aggregate results
            if game_simulations:
                simulation_results[team] = {
                    'game_simulations': game_simulations,
                    'total_wins': sum(sim['wins'] for sim in game_simulations),
                    'total_losses': sum(sim['losses'] for sim in game_simulations),
                    'avg_points_scored': np.mean([sim['points_scored'] for sim in game_simulations]),
                    'avg_points_allowed': np.mean([sim['points_allowed'] for sim in game_simulations])
                }
        
        print(f"Completed simulations for {len(simulation_results)} teams")
        return simulation_results
    
    def _simulate_game_outcome(self, team_proj: pd.Series, opp_proj: pd.Series, 
                             is_home: bool, team_var: Dict) -> Dict:
        """
        Simulate individual game outcome.
        
        Args:
            team_proj: Team projections
            opp_proj: Opponent projections
            is_home: Whether team is home
            team_var: Team variance metrics
            
        Returns:
            Dictionary with simulation results
        """
        # Base projections
        team_points = team_proj.get('points_per_game', 0)
        opp_points = opp_proj.get('points_per_game', 0)
        
        # Add home field advantage (data-driven)
        home_advantage = self._calculate_home_advantage(team_proj)
        if is_home:
            team_points += home_advantage
        else:
            opp_points += home_advantage
        
        # Apply variance
        team_volatility = team_var.get('volatility_score', 1.0)
        opp_volatility = 1.0  # Simplified for now
        
        # Generate random outcomes
        team_scores = np.random.normal(team_points, team_points * team_volatility * 0.1, self.n_simulations)
        opp_scores = np.random.normal(opp_points, opp_points * opp_volatility * 0.1, self.n_simulations)
        
        # Calculate wins/losses
        wins = np.sum(team_scores > opp_scores)
        losses = self.n_simulations - wins
        
        return {
            'wins': wins,
            'losses': losses,
            'points_scored': np.mean(team_scores),
            'points_allowed': np.mean(opp_scores),
            'win_probability': wins / self.n_simulations
        }
    
    def _calculate_home_advantage(self, team_proj: pd.Series) -> float:
        """
        Calculate home field advantage based on team data.
        
        Args:
            team_proj: Team projections
            
        Returns:
            Home field advantage in points
        """
        # Use home/away split if available
        if 'home_away_split' in team_proj and isinstance(team_proj['home_away_split'], dict):
            return team_proj['home_away_split'].get('home_advantage', 2.5)
        
        # Default home advantage
        return 2.5


class TeamOpponentAnalyzer:
    """
    Team opponent strength analyzer (mirrors player opponent analyzer).
    
    Key responsibilities:
    1. Analyze opponent defensive/offensive strength
    2. Calculate opponent adjustment multipliers
    3. Apply opponent adjustments to team projections
    """
    
    def __init__(self):
        self.opponent_adjustments: Dict[str, Dict] = {}
        
    def analyze_team_opponent_strength(self, df_team: pd.DataFrame, 
                                     df_opponents: pd.DataFrame, 
                                     league_averages: Dict) -> Dict:
        """
        Analyze team opponent strength using same logic as player engine.
        
        Args:
            df_team: Team statistics DataFrame
            df_opponents: Opponent statistics DataFrame
            league_averages: League averages from schedule analyzer
            
        Returns:
            Dictionary with opponent adjustments
        """
        print("Analyzing team opponent strength...")
        
        # Calculate opponent defensive strength
        opponent_defensive_strength = self._calculate_opponent_defensive_strength(df_opponents)
        
        # Create adjustment multipliers based on opponent strength
        adjustments = {}
        for team in df_team['team'].unique():
            # Get team's next opponent if available
            next_opponent = None
            if 'next_op' in df_team.columns:
                team_data = df_team[df_team['team'] == team]
                if not team_data.empty:
                    next_opponent = team_data.iloc[0].get('next_op')
            
            if next_opponent and next_opponent in opponent_defensive_strength:
                # Calculate adjustments based on opponent defensive strength
                opponent_def = opponent_defensive_strength[next_opponent]
                adjustments[team] = self._calculate_team_opponent_adjustments(opponent_def, league_averages)
            else:
                # Default neutral adjustments
                adjustments[team] = {
                    'offensive_multiplier': 1.0,
                    'defensive_multiplier': 1.0,
                    'home_advantage': 0.0
                }
        
        print(f"Calculated opponent adjustments for {len(adjustments)} teams")
        return adjustments
    
    def _calculate_team_opponent_adjustments(self, opponent_def: Dict, league_averages: Dict) -> Dict:
        """
        Calculate team adjustments based on opponent defensive strength.
        
        Args:
            opponent_def: Opponent defensive ratings
            league_averages: League averages from schedule analyzer
            
        Returns:
            Dictionary with adjustment multipliers
        """
        # Get opponent ratings
        pass_def = opponent_def['pass_defense']
        rush_def = opponent_def['rush_defense']
        rec_def = opponent_def['rec_defense']
        
        # Use actual league averages
        league_avg_pass = league_averages.get('pass_yards_per_game', 200)
        league_avg_rush = league_averages.get('rush_yards_per_game', 100)
        league_avg_rec = league_averages.get('total_yards_per_game', 150)
        
        # Calculate multipliers
        pass_multiplier = self._calculate_multiplier(pass_def, league_avg_pass)
        rush_multiplier = self._calculate_multiplier(rush_def, league_avg_rush)
        rec_multiplier = self._calculate_multiplier(rec_def, league_avg_rec)
        
        # Calculate overall offensive multiplier
        offensive_multiplier = (pass_multiplier + rush_multiplier + rec_multiplier) / 3
        defensive_multiplier = 1.0 / offensive_multiplier
        
        return {
            'offensive_multiplier': offensive_multiplier,
            'defensive_multiplier': defensive_multiplier,
            'home_advantage': 2.5,
            'pass_multiplier': pass_multiplier,
            'rush_multiplier': rush_multiplier,
            'rec_multiplier': rec_multiplier
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
    
    def _calculate_opponent_defensive_strength(self, df_opponents: pd.DataFrame) -> Dict:
        """
        Calculate opponent defensive strength using same logic as player engine.
        
        Args:
            df_opponents: Opponent statistics DataFrame
            
        Returns:
            Dictionary with defensive strength metrics
        """
        opponent_ratings = {}
        
        for team in df_opponents['team'].unique():
            team_data = df_opponents[df_opponents['team'] == team].iloc[0]
            
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
    


class TeamProjectionEngine:
    """
    Main team projection engine orchestrating the entire pipeline.
    
    This is the central class that coordinates all components of the
    team-level projection system.
    """
    
    def __init__(self, n_simulations: int = 1000, random_seed: int = 42):
        """
        Initialize the team projection engine.
        
        Args:
            n_simulations: Number of Monte Carlo simulations to run
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        
        # Initialize components
        self.team_data_loader = TeamDataLoader()
        self.team_schedule_analyzer = TeamScheduleAnalyzer()
        self.team_opponent_analyzer = TeamOpponentAnalyzer()
        self.team_variance_analyzer = TeamVarianceAnalyzer()
        self.team_simulation_engine = TeamSimulationEngine(n_simulations, random_seed)
        self.team_probability_engine = TeamProbabilityEngine()
        
        # Results storage
        self.team_baselines: pd.DataFrame = pd.DataFrame()
        self.team_projections: pd.DataFrame = pd.DataFrame()
        self.game_predictions: pd.DataFrame = pd.DataFrame()
    
    def run_team_projections(self, week_2025_file: str, weeks_2024_file: str,
                           target_weeks_2024: List[int], projection_week: int,
                           schedule_file: str, decay_coefficient: float = 0.7) -> Dict:
        """
        Run team projections using same pipeline as player engine.
        
        Args:
            week_2025_file: Path to 2025 season data
            weeks_2024_file: Path to 2024 season data
            target_weeks_2024: List of 2024 weeks to include
            projection_week: Current week for projections
            schedule_file: Path to schedule file
            decay_coefficient: Time decay coefficient
            
        Returns:
            Dictionary with team projection results
        """
        start_time = dt.now()
        print(f"\nTeam Projection Engine Start - {start_time}")
        print("=" * 60)
        
        try:
            # Step 1: Load and filter team data (10-game sample + time decay)
            print("\nStep 1: Loading and filtering team data...")
            df_team_data = self.team_data_loader.load_time_weighted_team_data(
                week_2025_file, weeks_2024_file, target_weeks_2024, 
                projection_week, decay_coefficient
            )
            
            # Step 2: Extract team performance baselines
            print("\nStep 2: Extracting team performance baselines...")
            self.team_baselines = self.team_data_loader.extract_team_performance_baseline(df_team_data)
            
            # Step 3: Schedule strength analysis
            print("\nStep 3: Analyzing team schedule strength...")
            df_team_with_schedule = self.team_schedule_analyzer.calculate_team_schedule_strength(
                self.team_baselines, schedule_file
            )
            
            # Step 4: Opponent analysis
            print("\nStep 4: Analyzing team opponent strength...")
            opponent_adjustments = self.team_opponent_analyzer.analyze_team_opponent_strength(
                df_team_with_schedule, df_team_data, self.team_schedule_analyzer.league_averages
            )
            
            # Step 5: Generate team projections
            print("\nStep 5: Generating team projections...")
            self.team_projections = self._generate_team_projections(
                df_team_with_schedule, opponent_adjustments
            )
            
            # Step 6: Analyze team variance
            print("\nStep 6: Analyzing team performance variance...")
            team_variance = self.team_variance_analyzer.analyze_team_variance(
                df_team_data, self.team_projections
            )
            
            # Step 7: Predict game outcomes
            print("\nStep 7: Predicting game outcomes...")
            self.game_predictions = self._predict_game_outcomes(
                self.team_projections, schedule_file, projection_week
            )
            
            # Step 8: Run Monte Carlo simulations
            print("\nStep 8: Running Monte Carlo simulations...")
            simulation_results = self.team_simulation_engine.run_team_simulations(
                self.team_projections, team_variance, self.game_predictions
            )
            
            # Step 9: Generate probability distributions
            print("\nStep 9: Generating probability distributions...")
            probability_distributions = self.team_probability_engine.generate_team_probability_distributions(
                simulation_results
            )
            
            # Step 10: Generate enhanced team totals
            print("\nStep 10: Generating enhanced team totals...")
            enhanced_totals = self._generate_enhanced_team_totals(
                self.team_projections, self.game_predictions, projection_week, simulation_results, probability_distributions
            )
            
            end_time = dt.now()
            print(f"\nTeam Projection Engine Complete - {end_time}")
            print(f"Total Runtime: {end_time - start_time}")
            
            return {
                'team_baselines': self.team_baselines,
                'team_projections': self.team_projections,
                'game_predictions': self.game_predictions,
                'enhanced_totals': enhanced_totals
            }
            
        except Exception as e:
            print(f"Error in team projection engine: {e}")
            return {'error': str(e)}
    
    def _generate_team_projections(self, df_team: pd.DataFrame, 
                                 opponent_adjustments: Dict) -> pd.DataFrame:
        """
        Generate team projections based on baselines and adjustments.
        
        Args:
            df_team: Team statistics DataFrame
            opponent_adjustments: Opponent adjustment multipliers
            
        Returns:
            DataFrame with team projections
        """
        print("Generating team projections...")
        
        # Start with team baselines
        team_projections = df_team.copy()
        
        # Apply opponent adjustments
        for idx, team_row in team_projections.iterrows():
            team = team_row['team']
            adjustments = opponent_adjustments.get(team, {})
            
            # Apply adjustments to projections
            for stat in ['points_per_game', 'pass_yards_per_game', 'rush_yards_per_game', 'total_yards_per_game']:
                if stat in team_projections.columns:
                    # Use specific multipliers if available, otherwise use overall offensive multiplier
                    if stat == 'pass_yards_per_game' and 'pass_multiplier' in adjustments:
                        multiplier = adjustments['pass_multiplier']
                    elif stat == 'rush_yards_per_game' and 'rush_multiplier' in adjustments:
                        multiplier = adjustments['rush_multiplier']
                    elif stat == 'total_yards_per_game' and 'rec_multiplier' in adjustments:
                        multiplier = adjustments['rec_multiplier']
                    else:
                        multiplier = adjustments.get('offensive_multiplier', 1.0)
                    
                    team_projections.loc[idx, f'proj_{stat}'] = team_row[stat] * multiplier
            
            # Apply defensive adjustments
            if 'points_allowed_per_game' in team_projections.columns:
                defensive_multiplier = adjustments.get('defensive_multiplier', 1.0)
                team_projections.loc[idx, 'proj_points_allowed_per_game'] = team_row['points_allowed_per_game'] * defensive_multiplier
        
        # Apply regression to mean
        team_projections = self._apply_regression_to_mean(team_projections)
        
        # Validate and cap projections
        team_projections = self._validate_and_cap_projections(team_projections)
        
        print(f"Generated projections for {len(team_projections)} teams")
        return team_projections
    
    def _predict_game_outcomes(self, team_projections: pd.DataFrame, 
                             schedule_file: str, projection_week: int) -> pd.DataFrame:
        """
        Predict individual game outcomes using team projections.
        
        Args:
            team_projections: Team projections DataFrame
            schedule_file: Path to schedule file
            projection_week: Current week for projections
            
        Returns:
            DataFrame with game predictions
        """
        print("Predicting game outcomes...")
        
        schedule = pd.read_csv(schedule_file)
        
        # Filter to remaining games
        remaining_games = schedule[schedule['Round Number'] >= projection_week]
        
        game_predictions = []
        for _, game in remaining_games.iterrows():
            home_team = game['Home Team']
            away_team = game['Away Team']
            
            # Get team projections
            home_proj = team_projections[team_projections['team'] == home_team]
            away_proj = team_projections[team_projections['team'] == away_team]
            
            if home_proj.empty or away_proj.empty:
                continue
            
            home_proj = home_proj.iloc[0]
            away_proj = away_proj.iloc[0]
            
            # Predict game outcome
            predicted_winner = self._predict_game_winner(home_proj, away_proj)
            
            game_predictions.append({
                'week': game['Round Number'],
                'home_team': home_team,
                'away_team': away_team,
                'predicted_winner': predicted_winner,
                'home_score_proj': home_proj.get('points_per_game', 0),
                'away_score_proj': away_proj.get('points_per_game', 0)
            })
        
        print(f"Predicted {len(game_predictions)} remaining games")
        return pd.DataFrame(game_predictions)
    
    def _predict_game_winner(self, home_proj: pd.Series, away_proj: pd.Series) -> str:
        """
        Predict game winner based on team projections.
        
        Args:
            home_proj: Home team projections
            away_proj: Away team projections
            
        Returns:
            Predicted winner team
        """
        # Simple prediction based on points per game
        home_points = home_proj.get('points_per_game', 0)
        away_points = away_proj.get('points_per_game', 0)
        
        # Add home field advantage (typically 2-3 points)
        home_advantage = 2.5
        home_points += home_advantage
        
        return home_proj['team'] if home_points > away_points else away_proj['team']
    
    def _generate_enhanced_team_totals(self, team_projections: pd.DataFrame, 
                                     game_predictions: pd.DataFrame, 
                                     projection_week: int, 
                                     simulation_results: Dict,
                                     probability_distributions: Dict) -> pd.DataFrame:
        """
        Generate enhanced team season totals with wins/losses/standings.
        
        Args:
            team_projections: Team projections DataFrame
            game_predictions: Game predictions DataFrame
            projection_week: Current week for projections
            
        Returns:
            DataFrame with enhanced team totals
        """
        print("Generating enhanced team totals...")
        
        team_totals = []
        
        for team in team_projections['team'].unique():
            team_proj = team_projections[team_projections['team'] == team].iloc[0]
            team_sims = simulation_results.get(team, {})
            
            # Use simulation results if available, otherwise fallback to simple calculation
            if team_sims:
                proj_wins = team_sims.get('total_wins', 0)
                proj_losses = team_sims.get('total_losses', 0)
                points_scored = team_sims.get('avg_points_scored', team_proj.get('points_per_game', 0))
                points_allowed = team_sims.get('avg_points_allowed', team_proj.get('points_allowed_per_game', 0))
            else:
                proj_wins, proj_losses = self._calculate_projected_record(team, game_predictions)
                points_scored = team_proj.get('points_per_game', 0)
                points_allowed = team_proj.get('points_allowed_per_game', 0)
            
            team_totals.append({
                'team': team,
                'projected_wins': proj_wins,
                'projected_losses': proj_losses,
                'points_scored_avg': points_scored,
                'points_allowed_avg': points_allowed,
                'offensive_yards_avg': team_proj.get('total_yards_per_game', 0),
                'conference': self._get_team_conference(team),
                'division': self._get_team_division(team)
            })
        
        print(f"Generated enhanced totals for {len(team_totals)} teams")
        return pd.DataFrame(team_totals)
    
    def _calculate_projected_record(self, team: str, game_predictions: pd.DataFrame) -> Tuple[int, int]:
        """
        Calculate projected wins/losses for a team.
        
        Args:
            team: Team abbreviation
            game_predictions: Game predictions DataFrame
            
        Returns:
            Tuple of (projected_wins, projected_losses)
        """
        team_games = game_predictions[
            (game_predictions['home_team'] == team) | 
            (game_predictions['away_team'] == team)
        ]
        
        wins = 0
        losses = 0
        
        for _, game in team_games.iterrows():
            if game['predicted_winner'] == team:
                wins += 1
            else:
                losses += 1
        
        return wins, losses
    
    def _get_team_conference(self, team: str) -> str:
        """
        Get team conference from data.
        
        Args:
            team: Team abbreviation
            
        Returns:
            Conference (AFC/NFC)
        """
        # Load from roster file if available
        try:
            import pandas as pd
            roster = pd.read_excel("data/roster.xlsx")
            if 'team' in roster.columns and 'conference' in roster.columns:
                team_data = roster[roster['team'] == team]
                if not team_data.empty:
                    return team_data.iloc[0]['conference']
        except:
            pass
        
        # Fallback to hardcoded mapping
        afc_teams = ['BUF', 'MIA', 'NE', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT', 
                    'HOU', 'IND', 'JAX', 'TEN', 'DEN', 'KC', 'LV', 'LAC']
        return 'AFC' if team in afc_teams else 'NFC'
    
    def _get_team_division(self, team: str) -> str:
        """
        Get team division from data.
        
        Args:
            team: Team abbreviation
            
        Returns:
            Division name
        """
        # Load from roster file if available
        try:
            import pandas as pd
            roster = pd.read_excel("data/roster.xlsx")
            if 'team' in roster.columns and 'division' in roster.columns:
                team_data = roster[roster['team'] == team]
                if not team_data.empty:
                    return team_data.iloc[0]['division']
        except:
            pass
        
        # Fallback to hardcoded mapping
        divisions = {
            'BUF': 'AFC East', 'MIA': 'AFC East', 'NE': 'AFC East', 'NYJ': 'AFC East',
            'BAL': 'AFC North', 'CIN': 'AFC North', 'CLE': 'AFC North', 'PIT': 'AFC North',
            'HOU': 'AFC South', 'IND': 'AFC South', 'JAX': 'AFC South', 'TEN': 'AFC South',
            'DEN': 'AFC West', 'KC': 'AFC West', 'LV': 'AFC West', 'LAC': 'AFC West',
            'DAL': 'NFC East', 'NYG': 'NFC East', 'PHI': 'NFC East', 'WAS': 'NFC East',
            'CHI': 'NFC North', 'DET': 'NFC North', 'GB': 'NFC North', 'MIN': 'NFC North',
            'ATL': 'NFC South', 'CAR': 'NFC South', 'NO': 'NFC South', 'TB': 'NFC South',
            'ARI': 'NFC West', 'LAR': 'NFC West', 'SF': 'NFC West', 'SEA': 'NFC West'
        }
        return divisions.get(team, 'Unknown')
    
    def _apply_regression_to_mean(self, df_team_projections: pd.DataFrame) -> pd.DataFrame:
        """
        Apply regression to mean to prevent overfitting (mirrors player engine).
        
        Args:
            df_team_projections: Team projections DataFrame
            
        Returns:
            DataFrame with regressed projections
        """
        print("Applying regression to mean for team projections...")
        
        regressed_projections = df_team_projections.copy()
        
        # Calculate league averages
        league_averages = {
            'points_per_game': df_team_projections['points_per_game'].mean(),
            'points_allowed_per_game': df_team_projections['points_allowed_per_game'].mean(),
            'pass_yards_per_game': df_team_projections['pass_yards_per_game'].mean(),
            'rush_yards_per_game': df_team_projections['rush_yards_per_game'].mean(),
            'total_yards_per_game': df_team_projections['total_yards_per_game'].mean()
        }
        
        # Apply regression to projection columns
        projection_columns = [col for col in df_team_projections.columns if col.startswith('proj_')]
        
        for col in projection_columns:
            if col in regressed_projections.columns:
                base_stat = col.replace('proj_', '')
                league_avg = league_averages.get(base_stat, 0)
                
                if league_avg > 0:
                    # Apply regression factor (0.8 = 80% projection, 20% league average)
                    regression_factor = 0.8
                    regressed_projections[col] = regressed_projections.apply(
                        lambda row: row[col] * regression_factor + league_avg * (1 - regression_factor), 
                        axis=1
                    )
        
        return regressed_projections
    
    def _validate_and_cap_projections(self, df_team_projections: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and cap team projections to realistic ranges (mirrors player engine).
        
        Args:
            df_team_projections: Team projections DataFrame
            
        Returns:
            DataFrame with validated projections
        """
        print("Validating and capping team projections...")
        
        validated_projections = df_team_projections.copy()
        
        # Define realistic ranges for team statistics
        stat_ranges = {
            'proj_points_per_game': (0, 50),  # 0-50 points per game
            'proj_points_allowed_per_game': (0, 50),  # 0-50 points allowed
            'proj_pass_yards_per_game': (0, 500),  # 0-500 passing yards
            'proj_rush_yards_per_game': (0, 300),  # 0-300 rushing yards
            'proj_total_yards_per_game': (0, 600)  # 0-600 total yards
        }
        
        # Apply caps to projection columns
        for col, (min_val, max_val) in stat_ranges.items():
            if col in validated_projections.columns:
                validated_projections[col] = validated_projections[col].clip(min_val, max_val)
        
        # Ensure non-negative values
        projection_columns = [col for col in validated_projections.columns if col.startswith('proj_')]
        for col in projection_columns:
            if col in validated_projections.columns:
                validated_projections[col] = validated_projections[col].clip(lower=0)
        
        return validated_projections


def run_team_projections(week_2025_file: str = "data/game_data/game_data_2025.csv",
                        weeks_2024_file: str = "data/game_data/game_data_2024.csv",
                        target_weeks_2024: List[int] = [9, 10, 11, 12, 13, 14, 15, 16, 17],
                        projection_week: int = 5,
                        schedule_file: str = "data/nfl-2025-EasternStandardTime.csv",
                        n_simulations: int = 1000) -> Dict:
    """
    Run the team projection engine.
    
    Args:
        week_2025_file: Path to 2025 season data
        weeks_2024_file: Path to 2024 season data
        target_weeks_2024: List of 2024 weeks to include
        projection_week: Current week for projections
        schedule_file: Path to schedule file
        n_simulations: Number of Monte Carlo simulations
        
    Returns:
        Dictionary with team projection results
    """
    # Initialize team projection engine
    engine = TeamProjectionEngine(n_simulations=n_simulations)
    
    # Run team projections
    results = engine.run_team_projections(
        week_2025_file=week_2025_file,
        weeks_2024_file=weeks_2024_file,
        target_weeks_2024=target_weeks_2024,
        projection_week=projection_week,
        schedule_file=schedule_file
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    results = run_team_projections()
    
    if 'error' not in results:
        print("\n📊 Team Projection Results:")
        print("=" * 50)
        print(f"Teams analyzed: {len(results['team_baselines'])}")
        print(f"Games predicted: {len(results['game_predictions'])}")
        print(f"Enhanced totals: {len(results['enhanced_totals'])}")
        
        # Display sample results
        if not results['enhanced_totals'].empty:
            print("\nTop 5 Teams by Projected Wins:")
            top_teams = results['enhanced_totals'].nlargest(5, 'projected_wins')
            print(top_teams[['team', 'projected_wins', 'projected_losses', 'points_scored_avg']])
    else:
        print(f"Error: {results['error']}")
