"""
Main Projection Engine - Optimized Modular Architecture

Orchestrates the entire projection pipeline:
1. Data loading and filtering
2. Schedule strength analysis
3. Opponent analysis
4. Base projections
5. Variance analysis
6. Monte Carlo simulations
7. Probability distributions

This is the main entry point for the optimized projection system.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os

# Import core modules
from .core.data_loader import DataLoader
from .core.schedule_analyzer import ScheduleAnalyzer
from .core.opponent_analyzer import OpponentAnalyzer
from .core.base_projections import BaseProjections
from .ml.variance_analyzer import VarianceAnalyzer
from .ml.simulation_engine import SimulationEngine
from .ml.probability_engine import ProbabilityEngine


class ProjectionEngine:
    """
    Main projection engine orchestrating the entire pipeline.
    
    This is the central class that coordinates all components of the
    optimized projection system.
    """
    
    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        """
        Initialize the projection engine.
        
        Args:
            n_simulations: Number of Monte Carlo simulations to run
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        
        # Initialize components
        self.data_loader = DataLoader(roster_file="data/roster.xlsx")
        self.schedule_analyzer = ScheduleAnalyzer()
        self.opponent_analyzer = OpponentAnalyzer()
        self.base_projections = BaseProjections()
        self.variance_analyzer = VarianceAnalyzer()
        self.simulation_engine = SimulationEngine(n_simulations, random_seed)
        self.probability_engine = ProbabilityEngine()
        
        # Results storage
        self.results: Dict = {}
        
    def run_projections(self, week_2025_file: str, weeks_2024_file: str,
                       target_weeks_2024: List[int], projection_week: int,
                       schedule_file: Optional[str] = None,
                       roster_file: str = "data/roster.xlsx",
                       decay_coefficient: float = 0.7) -> Dict:
        """
        Run the complete projection pipeline.
        
        Args:
            week_2025_file: Path to 2025 season data
            weeks_2024_file: Path to 2024 season data
            target_weeks_2024: List of 2024 weeks to include
            projection_week: Current week for projections
            schedule_file: Path to schedule file (optional)
            roster_file: Path to roster file
            decay_coefficient: Time decay coefficient
            
        Returns:
            Dictionary with all projection results
        """
        start_time = datetime.now()
        print(f"🚀 Starting NFL Projection Engine - Week {projection_week}")
        print(f"📊 Running {self.n_simulations:,} Monte Carlo simulations")
        print(f"⏰ Started at {start_time}")
        print("=" * 60)
        
        try:
            # Step 1: Load and filter data
            print("\n📁 Step 1: Loading and filtering data...")
            df_game_data = self._load_and_filter_data(
                week_2025_file, weeks_2024_file, target_weeks_2024, 
                projection_week, decay_coefficient, roster_file
            )
            
            # Step 2: Create team and player datasets
            print("\n🏈 Step 2: Creating team and player datasets...")
            df_team, df_players = self._create_datasets(df_game_data, schedule_file, projection_week)
            
            # Step 3: Schedule strength analysis
            print("\n📊 Step 3: Analyzing schedule strength...")
            df_team_with_schedule = self.schedule_analyzer.calculate_schedule_strength(df_team)
            
            # Step 4: Opponent analysis
            print("\n🎯 Step 4: Analyzing opponent strength...")
            opponent_adjustments = self.opponent_analyzer.analyze_opponent_strength(
                df_team_with_schedule, df_players
            )
            
            # Step 5: Base projections
            print("\n📈 Step 5: Generating base projections...")
            df_base_projections = self.base_projections.generate_base_projections(
                df_players, df_team_with_schedule, opponent_adjustments
            )
            
            # Step 6: Variance analysis
            print("\n📊 Step 6: Analyzing historical variance...")
            variance_analysis = self.variance_analyzer.analyze_historical_variance(
                df_game_data, df_base_projections
            )
            
            # Step 7: Monte Carlo simulations
            print("\n🎲 Step 7: Running Monte Carlo simulations...")
            simulation_results = self.simulation_engine.run_simulations(
                df_base_projections, variance_analysis, opponent_adjustments
            )
            
            # Step 8: Generate final results
            print("\n📋 Step 8: Compiling final results...")
            final_results = self._compile_final_results(
                df_base_projections, simulation_results, df_team_with_schedule
            )
            
            # Step 9: Export results
            print("\n💾 Step 9: Exporting results...")
            self._export_results(final_results, projection_week)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "=" * 60)
            print(f"✅ Projection Engine Complete!")
            print(f"⏰ Duration: {duration}")
            print(f"📊 Players analyzed: {len(df_base_projections)}")
            print(f"🎲 Simulations run: {self.n_simulations:,}")
            print(f"📈 Projections generated: {len(final_results['base_projections'])}")
            print("=" * 60)
            
            self.results = final_results
            return final_results
            
        except Exception as e:
            print(f"\n❌ Error in projection engine: {e}")
            raise
    
    def _load_and_filter_data(self, week_2025_file: str, weeks_2024_file: str,
                            target_weeks_2024: List[int], projection_week: int,
                            decay_coefficient: float, roster_file: str) -> pd.DataFrame:
        """Load and filter game data."""
        # Load active roster and create mappings
        self.data_loader.load_active_roster()
        self.data_loader.create_player_team_mapping()
        
        # Load time-weighted data
        df_game_data = self.data_loader.load_time_weighted_data(
            week_2025_file, weeks_2024_file, target_weeks_2024, 
            projection_week, decay_coefficient
        )
        
        # Filter to active players only
        df_filtered = self.data_loader.filter_active_players(df_game_data)
        
        # Apply snap count filtering to only include players with significant recent snaps
        df_filtered = self.data_loader.filter_by_snap_count(df_filtered, snap_threshold=20.0)
        
        print(f"✅ Loaded {len(df_filtered)} game records for {df_filtered['player'].nunique()} active players")
        return df_filtered
    
    def _create_datasets(self, df_game_data: pd.DataFrame, schedule_file: Optional[str],
                        projection_week: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create team and player datasets."""
        # Create team dataset
        df_team = self._create_team_dataset(df_game_data, schedule_file, projection_week)
        
        # Create player dataset
        df_players = self._create_player_dataset(df_game_data)
        
        print(f"✅ Created datasets: {len(df_team)} teams, {len(df_players)} players")
        return df_team, df_players
    
    def _create_team_dataset(self, df_game_data: pd.DataFrame, schedule_file: Optional[str],
                           projection_week: int) -> pd.DataFrame:
        """Create team dataset from game data."""
        print("Creating team dataset from game data...")
        
        # Build team opponents schedule
        team_schedule = self._build_team_opponents_schedule(df_game_data)
        
        # Build team statistics
        team_stats = self._build_team_statistics(df_game_data)
        
        # Aggregate team stats across all games
        team_aggregated = team_stats.groupby('team').agg({
            'pass_cmp': 'sum', 'pass_att': 'sum', 'pass_yds': 'sum', 'pass_tds': 'sum', 'pass_int': 'sum', 'sacks': 'sum',
            'rush_att': 'sum', 'rush_yds': 'sum', 'rush_tds': 'sum', 'targets': 'sum', 'receptions': 'sum', 
            'rec_yds': 'sum', 'rec_tds': 'sum', 'fumbles': 'sum',
            'def_pass_cmp': 'sum', 'def_pass_att': 'sum', 'def_pass_yd': 'sum', 'def_pass_td': 'sum', 
            'def_int': 'sum', 'def_sacks': 'sum', 'def_rush_att': 'sum', 'def_rush_yd': 'sum', 
            'def_rush_td': 'sum', 'def_targets': 'sum', 'def_rec': 'sum', 'def_rec_yd': 'sum', 
            'def_rec_td': 'sum', 'def_fum': 'sum'
        }).reset_index()
        
        # Rename columns to match expected format
        team_aggregated = team_aggregated.rename(columns={
            'pass_yds': 'pass_yd', 'pass_tds': 'pass_td', 'rush_yds': 'rush_yd', 'rush_tds': 'rush_td',
            'rec_yds': 'rec_yd', 'rec_tds': 'rec_td', 'receptions': 'rec', 'fumbles': 'off_fum'
        })
        
        # Merge with team schedule
        team_complete = pd.merge(team_aggregated, team_schedule, on='team', how='left')
        
        # Add next opponent info if schedule is available
        if schedule_file and os.path.exists(schedule_file):
            df_schedule = pd.read_csv(schedule_file)
            df_schedule['Round Number'] = pd.to_numeric(df_schedule['Round Number'])
            
            # Add next game info
            next_game_info = team_complete['team'].apply(
                lambda team: self._determine_next_game_info(team, projection_week, df_schedule)
            )
            
            team_complete['next_op'] = next_game_info.apply(lambda x: x['opponent'])
            team_complete['next_game_location'] = next_game_info.apply(lambda x: x['location'])
            team_complete['next_game_week'] = next_game_info.apply(lambda x: x['week'])
        else:
            team_complete['next_op'] = None
            team_complete['next_game_location'] = None
            team_complete['next_game_week'] = None
        
        # Clean up data
        team_complete = team_complete[team_complete['team'].notna()]
        team_complete = team_complete[team_complete['team'] != '']
        
        print(f"Created team dataset with {len(team_complete)} teams")
        return team_complete
    
    def _create_player_dataset(self, df_game_data: pd.DataFrame) -> pd.DataFrame:
        """Create player dataset from game data."""
        print("Creating player dataset from game data...")
        
        # Filter to active players only
        df_filtered = df_game_data[df_game_data['player'].isin(self.data_loader.active_roster)].copy()
        
        if df_filtered.empty:
            print("No active players found in game data")
            return pd.DataFrame()
        
        # Apply time-weighted aggregation
        stat_columns = ['pass_cmp', 'pass_att', 'pass_yds', 'pass_tds', 'pass_int', 'sacks',
                       'rush_att', 'rush_yds', 'rush_tds', 'targets', 'receptions', 'rec_yds', 'rec_tds', 'fumbles']
        
        def weighted_average(group, stat_col, weight_col):
            if group[weight_col].sum() == 0:
                return 0
            return np.average(group[stat_col], weights=group[weight_col])
        
        # Group by player and calculate weighted averages
        def calculate_player_stats(group):
            return pd.Series({
                'pass_cmp': weighted_average(group, 'pass_cmp', 'time_weight'),
                'pass_att': weighted_average(group, 'pass_att', 'time_weight'),
                'pass_yds': weighted_average(group, 'pass_yds', 'time_weight'),
                'pass_tds': weighted_average(group, 'pass_tds', 'time_weight'),
                'pass_int': weighted_average(group, 'pass_int', 'time_weight'),
                'sacks': weighted_average(group, 'sacks', 'time_weight'),
                'rush_att': weighted_average(group, 'rush_att', 'time_weight'),
                'rush_yds': weighted_average(group, 'rush_yds', 'time_weight'),
                'rush_tds': weighted_average(group, 'rush_tds', 'time_weight'),
                'targets': weighted_average(group, 'targets', 'time_weight'),
                'receptions': weighted_average(group, 'receptions', 'time_weight'),
                'rec_yds': weighted_average(group, 'rec_yds', 'time_weight'),
                'rec_tds': weighted_average(group, 'rec_tds', 'time_weight'),
                'fumbles': weighted_average(group, 'fumbles', 'time_weight')
            })
        
        player_stats = df_filtered.groupby('player').apply(calculate_player_stats).reset_index()
        
        # Rename columns to match expected format
        player_stats = player_stats.rename(columns={
            'pass_yds': 'pass_yd', 'pass_tds': 'pass_td', 'rush_yds': 'rush_yd', 'rush_tds': 'rush_td',
            'rec_yds': 'rec_yd', 'rec_tds': 'rec_td', 'receptions': 'rec', 'fumbles': 'fum',
            'targets': 'tar'
        })
        
        # Count games played
        games_played = df_filtered.groupby('player')['week'].nunique().reset_index()
        games_played.columns = ['player', 'g']
        
        # Merge stats with games played
        player_complete = pd.merge(player_stats, games_played, on='player', how='left')
        
        # Add current team from mapping
        player_complete['team'] = player_complete['player'].map(self.data_loader.player_team_mapping)
        
        # Remove players without current team mapping
        player_complete = player_complete.dropna(subset=['team'])
        
        # Rename player column to 'name'
        player_complete.rename(columns={'player': 'name'}, inplace=True)
        
        # Clean up data
        player_complete = player_complete[player_complete['team'].notna()]
        player_complete = player_complete[player_complete['team'] != '']
        
        print(f"Created player dataset with {len(player_complete)} players")
        return player_complete
    
    def _build_team_opponents_schedule(self, df_game_data: pd.DataFrame) -> pd.DataFrame:
        """Build team schedule mapping from player-level game data."""
        # Get unique team-game combinations
        team_schedule = df_game_data[['year', 'week', 'home_team', 'away_team', 'team', 'opponent']].drop_duplicates()
        
        # Group by team and aggregate opponents
        team_opponents = team_schedule.groupby('team').agg({
            'opponent': lambda x: str(list(x)),  # String representation for CSV compatibility
            'week': 'count'  # Number of games played
        }).rename(columns={'opponent': 'opps', 'week': 'games'})
        
        # Ensure opps is never empty
        team_opponents['opps'] = team_opponents['opps'].apply(
            lambda x: "[]" if pd.isna(x) or x == "0.0" or x == 0.0 else x
        )
        
        return team_opponents
    
    def _build_team_statistics(self, df_game_data: pd.DataFrame) -> pd.DataFrame:
        """Build team statistics from player-level game data."""
        # Apply proper time-weighted aggregation
        stat_columns = ['pass_cmp', 'pass_att', 'pass_yds', 'pass_tds', 'pass_int', 'sacks',
                       'rush_att', 'rush_yds', 'rush_tds', 'targets', 'receptions', 'rec_yds', 'rec_tds', 'fumbles']
        
        def weighted_average(group, stat_col, weight_col):
            if group[weight_col].sum() == 0:
                return 0
            return np.average(group[stat_col], weights=group[weight_col])
        
        # Group by team, opponent, week and calculate weighted averages for offensive stats
        team_stats = df_game_data.groupby(['team', 'opponent', 'week']).apply(
            lambda group: pd.Series({
                'pass_cmp': weighted_average(group, 'pass_cmp', 'time_weight'),
                'pass_att': weighted_average(group, 'pass_att', 'time_weight'),
                'pass_yds': weighted_average(group, 'pass_yds', 'time_weight'),
                'pass_tds': weighted_average(group, 'pass_tds', 'time_weight'),
                'pass_int': weighted_average(group, 'pass_int', 'time_weight'),
                'sacks': weighted_average(group, 'sacks', 'time_weight'),
                'rush_att': weighted_average(group, 'rush_att', 'time_weight'),
                'rush_yds': weighted_average(group, 'rush_yds', 'time_weight'),
                'rush_tds': weighted_average(group, 'rush_tds', 'time_weight'),
                'targets': weighted_average(group, 'targets', 'time_weight'),
                'receptions': weighted_average(group, 'receptions', 'time_weight'),
                'rec_yds': weighted_average(group, 'rec_yds', 'time_weight'),
                'rec_tds': weighted_average(group, 'rec_tds', 'time_weight'),
                'fumbles': weighted_average(group, 'fumbles', 'time_weight')
            })
        ).reset_index()
        
        # Calculate defensive stats by aggregating opponent's offensive performance
        def calculate_opponent_stats(group):
            return pd.Series({
                'pass_cmp': weighted_average(group, 'pass_cmp', 'time_weight'),
                'pass_att': weighted_average(group, 'pass_att', 'time_weight'),
                'pass_yds': weighted_average(group, 'pass_yds', 'time_weight'),
                'pass_tds': weighted_average(group, 'pass_tds', 'time_weight'),
                'pass_int': weighted_average(group, 'pass_int', 'time_weight'),
                'sacks': weighted_average(group, 'sacks', 'time_weight'),
                'rush_att': weighted_average(group, 'rush_att', 'time_weight'),
                'rush_yds': weighted_average(group, 'rush_yds', 'time_weight'),
                'rush_tds': weighted_average(group, 'rush_tds', 'time_weight'),
                'targets': weighted_average(group, 'targets', 'time_weight'),
                'receptions': weighted_average(group, 'receptions', 'time_weight'),
                'rec_yds': weighted_average(group, 'rec_yds', 'time_weight'),
                'rec_tds': weighted_average(group, 'rec_tds', 'time_weight'),
                'fumbles': weighted_average(group, 'fumbles', 'time_weight')
            })
        
        opponent_offensive_stats = df_game_data.groupby(['opponent', 'team', 'week']).apply(calculate_opponent_stats).reset_index()
        
        # Map opponent offensive stats to defensive stats
        def_col_mapping = {
            'pass_cmp': 'def_pass_cmp', 'pass_att': 'def_pass_att', 'pass_yds': 'def_pass_yd',
            'pass_tds': 'def_pass_td', 'pass_int': 'def_int', 'sacks': 'def_sacks',
            'rush_att': 'def_rush_att', 'rush_yds': 'def_rush_yd', 'rush_tds': 'def_rush_td',
            'targets': 'def_targets', 'receptions': 'def_rec', 'rec_yds': 'def_rec_yd',
            'rec_tds': 'def_rec_td', 'fumbles': 'def_fum'
        }
        
        # Rename opponent offensive columns to defensive columns
        opponent_offensive_stats.columns = ['team', 'opponent', 'week'] + [def_col_mapping[col] for col in opponent_offensive_stats.columns[3:]]
        
        # Merge offensive and defensive stats
        team_complete = pd.merge(team_stats, opponent_offensive_stats, on=['team', 'opponent', 'week'], how='outer')
        
        return team_complete
    
    def _determine_next_game_info(self, team: str, current_week: int, df_schedule: pd.DataFrame) -> dict:
        """Determine next game information for a team."""
        # Find next game for this team (week > current_week)
        future_games = df_schedule[df_schedule['Round Number'] > current_week]
        
        # Find games where this team plays
        team_games = future_games[
            (future_games['Away Team'] == team) | 
            (future_games['Home Team'] == team)
        ]
        
        if team_games.empty:
            return {
                'opponent': None,
                'location': 'BYE',
                'week': None
            }
        
        # Get the next game (lowest week number)
        next_game = team_games.loc[team_games['Round Number'].idxmin()]
        
        # Determine opponent and location
        if next_game['Away Team'] == team:
            return {
                'opponent': next_game['Home Team'],
                'location': 'AWAY',
                'week': next_game['Round Number']
            }
        else:
            return {
                'opponent': next_game['Away Team'],
                'location': 'HOME',
                'week': next_game['Round Number']
            }
    
    def _compile_final_results(self, df_base_projections: pd.DataFrame,
                              simulation_results: Dict, df_schedule_strength: pd.DataFrame) -> Dict:
        """Compile final results from all components."""
        return {
            'base_projections': df_base_projections,
            'simulation_results': simulation_results,
            'schedule_strength': df_schedule_strength,
            'metadata': {
                'n_simulations': self.n_simulations,
                'random_seed': self.random_seed,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _export_results(self, results: Dict, projection_week: int):
        """Export results to files."""
        # Create projections directory
        os.makedirs("data/projections", exist_ok=True)
        
        # Export base projections
        if 'base_projections' in results and not results['base_projections'].empty:
            projections_file = f"data/projections/nfl25_proj_week{projection_week}.csv"
            results['base_projections'].to_csv(projections_file, index="name")
            print(f"✅ Base projections saved to {projections_file}")
        
        # Export schedule strength data
        if 'schedule_strength' in results and not results['schedule_strength'].empty:
            schedule_file = f"data/projections/schedule_strength_week{projection_week}.csv"
            results['schedule_strength'].to_csv(schedule_file, index="team")
            print(f"✅ Schedule strength saved to {schedule_file}")
        
        # Export Monte Carlo results
        if 'simulation_results' in results and results['simulation_results']:
            monte_carlo_file = f"data/projections/monte_carlo_week{projection_week}.json"
            with open(monte_carlo_file, 'w') as f:
                json.dump(results['simulation_results'], f, indent=2, default=str)
            print(f"✅ Monte Carlo results saved to {monte_carlo_file}")
    


def run_optimized_projections(week_2025_file: str = "data/game_data/game_data_2025.csv",
                             weeks_2024_file: str = "data/game_data/game_data_2024.csv",
                             target_weeks_2024: List[int] = [9, 10, 11, 12, 13, 14, 15, 16, 17],
                             projection_week: int = 1,
                             schedule_file: str = "data/nfl-2025-EasternStandardTime.csv",
                             n_simulations: int = 10000) -> Dict:
    """
    Run the optimized projection engine.
    
    Args:
        week_2025_file: Path to 2025 season data
        weeks_2024_file: Path to 2024 season data
        target_weeks_2024: List of 2024 weeks to include
        projection_week: Current week for projections
        schedule_file: Path to schedule file
        n_simulations: Number of Monte Carlo simulations
        
    Returns:
        Dictionary with projection results
    """
    # Initialize projection engine
    engine = ProjectionEngine(n_simulations=n_simulations)
    
    # Run projections
    results = engine.run_projections(
        week_2025_file=week_2025_file,
        weeks_2024_file=weeks_2024_file,
        target_weeks_2024=target_weeks_2024,
        projection_week=projection_week,
        schedule_file=schedule_file
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    results = run_optimized_projections(projection_week=1, n_simulations=10000)
    print("Projection engine completed successfully!")
