"""
NFL Projection Backtester Module

Simulates projection runs using historical data to evaluate accuracy.
Generates CSV reports for baseline comparison and improvement tracking.

This module allows testing of the projection engine against historical
data to establish accuracy baselines and track improvements over time.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import new projection engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projection_engine import ProjectionEngine
from backtest_config import get_backtest_config

class NFLBacktester:
    """
    NFL Projection Backtester for evaluating projection accuracy.
    
    This class simulates projection runs using historical data to:
    - Generate projections for past weeks
    - Compare against actual results
    - Calculate accuracy metrics
    - Generate CSV reports for analysis
    """
    
    def __init__(self, test_season: int, reference_season: int, decay_coefficient: float = 0.7, backtest_scenario: str = "historical"):
        """
        Initialize backtester for specific seasons.
        
        Args:
            test_season: Season to test projections against
            reference_season: Season to use as reference data
            decay_coefficient: Exponential decay coefficient for time weighting
            backtest_scenario: Configuration scenario - "historical" or "current"
        """
        self.test_season = test_season
        self.reference_season = reference_season
        self.decay_coefficient = decay_coefficient
        self.config = get_backtest_config(backtest_scenario)
        self.results = []
        
        # Validate data availability
        self._validate_data_availability()
    
    def _validate_data_availability(self):
        """Validate that required data files exist."""
        required_files = [
            f'data/game_data/game_data_{self.test_season}.csv',
            f'data/game_data/game_data_{self.reference_season}.csv',
            f'data/schedule_{self.test_season}.csv',
            f'data/roster_{self.test_season}.csv'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
    
    def load_historical_data(self, season: int, week: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load historical data for backtesting.
        
        Args:
            season: Season year
            week: Week number
            
        Returns:
            Tuple of (game_data, schedule, roster)
        """
        # Load game data
        game_data = pd.read_csv(f'data/game_data/game_data_{season}.csv')
        
        # Load historical schedule
        schedule = pd.read_csv(f'data/schedule_{season}.csv')
        
        # Load historical roster
        roster = pd.read_csv(f'data/roster_{season}.csv')
        
        return game_data, schedule, roster
    
    def generate_historical_projections(self, week: int) -> pd.DataFrame:
        """
        Generate projections for a specific week using historical data.
        
        Args:
            week: Week number to generate projections for
            
        Returns:
            DataFrame with projections
        """
        print(f"  Generating projections for Week {week}...")
        
        try:
            # Initialize the new projection engine
            engine = ProjectionEngine(n_simulations=1000)  # Use fewer simulations for backtesting
            
            # Determine which season to use for current year data
            current_season = self.test_season
            reference_season = self.reference_season
            
            # Generate projections using the new engine
            results = engine.run_projections(
                week_2025_file=f'data/game_data/game_data_{current_season}.csv',
                weeks_2024_file=f'data/game_data/game_data_{reference_season}.csv',
                target_weeks_2024=[9, 10, 11, 12, 13, 14, 15, 16, 17],
                projection_week=week,
                schedule_file=f'data/nfl-2025-EasternStandardTime.csv',  # Use the correct schedule format
                roster_file="data/roster.xlsx"  # Use the correct roster format
            )
            
            if results and 'base_projections' in results:
                return results['base_projections']
            else:
                print(f"    Warning: No projections generated for Week {week}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"    Error generating projections for Week {week}: {e}")
            return pd.DataFrame()
    
    def get_actual_results(self, week: int) -> pd.DataFrame:
        """
        Get actual results for a specific week.
        
        Args:
            week: Week number
            
        Returns:
            DataFrame with actual player stats
        """
        try:
            # Load game data for the test season
            df = pd.read_csv(f'data/game_data/game_data_{self.test_season}.csv')
            
            # Filter to specific week
            week_data = df[df['week'] == week].copy()
            
            if week_data.empty:
                print(f"    Warning: No data found for Week {week}")
                return pd.DataFrame()
            
            # Aggregate player stats for the week
            actuals = week_data.groupby('player').agg({
                'pass_yds': 'sum',
                'pass_tds': 'sum',
                'pass_int': 'sum',
                'rush_yds': 'sum',
                'rush_tds': 'sum',
                'rec_yds': 'sum',
                'rec_tds': 'sum',
                'receptions': 'sum',
                'team': 'first',
                'pos': 'first'
            }).reset_index()
            
            return actuals
            
        except Exception as e:
            print(f"    Error getting actual results for Week {week}: {e}")
            return pd.DataFrame()
    
    def calculate_monte_carlo_accuracy(self, projections: pd.DataFrame, actuals: pd.DataFrame, confidence_level=0.5) -> Dict:
        """
        Calculate Monte Carlo accuracy metrics for uncertainty quantification.
        
        Args:
            projections: DataFrame with Monte Carlo enhanced projections
            actuals: DataFrame with actual results
            confidence_level: Confidence level for interval testing (default 0.5)
            
        Returns:
            Dictionary with Monte Carlo accuracy metrics
        """
        if projections.empty or actuals.empty:
            return {'error': 'Empty projections or actuals data'}
        
        try:
            # Merge projections with actuals
            if 'name' in projections.columns and 'player' in actuals.columns:
                merged = pd.merge(projections, actuals, left_on='name', right_on='player', how='inner', suffixes=('_proj', '_actual'))
            elif 'player' in projections.columns and 'player' in actuals.columns:
                merged = pd.merge(projections, actuals, on='player', how='inner', suffixes=('_proj', '_actual'))
            else:
                return {'error': 'Cannot match player columns between projections and actuals'}
            
            if merged.empty:
                return {'error': 'No matching players found between projections and actuals'}
            
            # Calculate Monte Carlo accuracy metrics
            mc_metrics = {}
            
            # Calculate confidence interval coverage for each stat
            for stat in ['pass_yd', 'rush_yd', 'rec_yd']:
                if f'{stat}_p25' in merged.columns and f'{stat}_p75' in merged.columns:
                    # Check if actual values fall within confidence intervals
                    lower_bound = merged[f'{stat}_p25']
                    upper_bound = merged[f'{stat}_p75']
                    actual_values = merged[f'{stat}_actual']
                    
                    # Calculate coverage rate
                    within_bounds = (actual_values >= lower_bound) & (actual_values <= upper_bound)
                    coverage_rate = within_bounds.mean()
                    
                    mc_metrics[f'{stat}_coverage'] = coverage_rate
                    
                    # Calculate interval width (uncertainty quantification)
                    interval_width = (upper_bound - lower_bound).mean()
                    mc_metrics[f'{stat}_interval_width'] = interval_width
                    
                    # Calculate calibration (how well the model quantifies uncertainty)
                    expected_coverage = confidence_level
                    calibration_error = abs(coverage_rate - expected_coverage)
                    mc_metrics[f'{stat}_calibration_error'] = calibration_error
            
            return mc_metrics
            
        except Exception as e:
            return {'error': f'Error calculating Monte Carlo accuracy: {e}'}

    def calculate_accuracy(self, projections: pd.DataFrame, actuals: pd.DataFrame) -> Dict:
        """
        Calculate accuracy metrics for projections vs actuals.
        Includes both overall and position-specific metrics.
        
        Args:
            projections: DataFrame with projections
            actuals: DataFrame with actual results
            
        Returns:
            Dictionary with accuracy metrics including position breakdowns
        """
        if projections.empty or actuals.empty:
            return {'error': 'Empty projections or actuals data'}
        
        try:
            # Merge projections with actuals
            # Handle different column names for player identification
            if 'name' in projections.columns and 'player' in actuals.columns:
                merged = pd.merge(projections, actuals, left_on='name', right_on='player', how='inner', suffixes=('_proj', '_actual'))
            elif 'player' in projections.columns and 'player' in actuals.columns:
                merged = pd.merge(projections, actuals, on='player', how='inner', suffixes=('_proj', '_actual'))
            else:
                return {'error': 'Cannot match player columns between projections and actuals'}
            
            if merged.empty:
                return {'error': 'No matching players found between projections and actuals'}
            
            # Calculate accuracy metrics
            metrics = {}
            
            # Passing yards accuracy (new system uses 'pass_yd' instead of 'proj_pass_yd')
            if 'pass_yd' in merged.columns and 'pass_yds' in merged.columns:
                pass_yd_mae = np.mean(np.abs(merged['pass_yd'] - merged['pass_yds']))
                pass_yd_mape = np.mean(np.abs((merged['pass_yd'] - merged['pass_yds']) / merged['pass_yds'].replace(0, 1))) * 100
                metrics['pass_yd_mae'] = pass_yd_mae
                metrics['pass_yd_mape'] = pass_yd_mape
            
            # Rushing yards accuracy (new system uses 'rush_yd' instead of 'proj_rush_yd')
            if 'rush_yd' in merged.columns and 'rush_yds' in merged.columns:
                rush_yd_mae = np.mean(np.abs(merged['rush_yd'] - merged['rush_yds']))
                rush_yd_mape = np.mean(np.abs((merged['rush_yd'] - merged['rush_yds']) / merged['rush_yds'].replace(0, 1))) * 100
                metrics['rush_yd_mae'] = rush_yd_mae
                metrics['rush_yd_mape'] = rush_yd_mape
            
            # Receiving yards accuracy (new system uses 'rec_yd' instead of 'proj_rec_yd')
            if 'rec_yd' in merged.columns and 'rec_yds' in merged.columns:
                rec_yd_mae = np.mean(np.abs(merged['rec_yd'] - merged['rec_yds']))
                rec_yd_mape = np.mean(np.abs((merged['rec_yd'] - merged['rec_yds']) / merged['rec_yds'].replace(0, 1))) * 100
                metrics['rec_yd_mae'] = rec_yd_mae
                metrics['rec_yd_mape'] = rec_yd_mape
            
            # Passing TDs accuracy (new system uses 'pass_td' instead of 'proj_pass_td')
            if 'pass_td' in merged.columns and 'pass_tds' in merged.columns:
                pass_td_mae = np.mean(np.abs(merged['pass_td'] - merged['pass_tds']))
                metrics['pass_td_mae'] = pass_td_mae
            
            # Rushing TDs accuracy (new system uses 'rush_td' instead of 'proj_rush_td')
            if 'rush_td' in merged.columns and 'rush_tds' in merged.columns:
                rush_td_mae = np.mean(np.abs(merged['rush_td'] - merged['rush_tds']))
                metrics['rush_td_mae'] = rush_td_mae
            
            # Receiving TDs accuracy (new system uses 'rec_td' instead of 'proj_rec_td')
            if 'rec_td' in merged.columns and 'rec_tds' in merged.columns:
                rec_td_mae = np.mean(np.abs(merged['rec_td'] - merged['rec_tds']))
                metrics['rec_td_mae'] = rec_td_mae
            
            # Overall accuracy metrics
            metrics['total_players'] = len(merged)
            metrics['matching_players'] = len(merged)
            
            # Position-specific accuracy analysis
            if 'pos' in merged.columns:
                position_metrics = self._calculate_position_accuracy(merged)
                metrics.update(position_metrics)
            
            return metrics
            
        except Exception as e:
            return {'error': f'Error calculating accuracy: {e}'}
    
    def _calculate_position_accuracy(self, merged: pd.DataFrame) -> Dict:
        """
        Calculate position-specific accuracy metrics.
        
        Args:
            merged: DataFrame with merged projections and actuals
            
        Returns:
            Dictionary with position-specific metrics
        """
        position_metrics = {}
        
        # Get unique positions
        positions = merged['pos'].unique()
        
        for pos in positions:
            pos_data = merged[merged['pos'] == pos]
            if len(pos_data) == 0:
                continue
                
            # Position-specific metrics
            pos_key = f"{pos.lower()}_"
            
            # Player count for this position
            position_metrics[f"{pos_key}players"] = len(pos_data)
            
            # Passing yards (for QBs) - updated for new column names
            if pos == 'QB' and 'pass_yd' in pos_data.columns and 'pass_yds' in pos_data.columns:
                qb_pass_yd_mae = np.mean(np.abs(pos_data['pass_yd'] - pos_data['pass_yds']))
                qb_pass_yd_mape = np.mean(np.abs((pos_data['pass_yd'] - pos_data['pass_yds']) / pos_data['pass_yds'].replace(0, 1))) * 100
                position_metrics[f"{pos_key}pass_yd_mae"] = qb_pass_yd_mae
                position_metrics[f"{pos_key}pass_yd_mape"] = qb_pass_yd_mape
                
                if 'pass_td' in pos_data.columns and 'pass_tds' in pos_data.columns:
                    qb_pass_td_mae = np.mean(np.abs(pos_data['pass_td'] - pos_data['pass_tds']))
                    position_metrics[f"{pos_key}pass_td_mae"] = qb_pass_td_mae
            
            # Rushing yards (for RBs and QBs) - updated for new column names
            if pos in ['RB', 'QB'] and 'rush_yd' in pos_data.columns and 'rush_yds' in pos_data.columns:
                rush_yd_mae = np.mean(np.abs(pos_data['rush_yd'] - pos_data['rush_yds']))
                rush_yd_mape = np.mean(np.abs((pos_data['rush_yd'] - pos_data['rush_yds']) / pos_data['rush_yds'].replace(0, 1))) * 100
                position_metrics[f"{pos_key}rush_yd_mae"] = rush_yd_mae
                position_metrics[f"{pos_key}rush_yd_mape"] = rush_yd_mape
                
                if 'rush_td' in pos_data.columns and 'rush_tds' in pos_data.columns:
                    rush_td_mae = np.mean(np.abs(pos_data['rush_td'] - pos_data['rush_tds']))
                    position_metrics[f"{pos_key}rush_td_mae"] = rush_td_mae
            
            # Receiving yards (for WRs, TEs, RBs) - updated for new column names
            if pos in ['WR', 'TE', 'RB'] and 'rec_yd' in pos_data.columns and 'rec_yds' in pos_data.columns:
                rec_yd_mae = np.mean(np.abs(pos_data['rec_yd'] - pos_data['rec_yds']))
                rec_yd_mape = np.mean(np.abs((pos_data['rec_yd'] - pos_data['rec_yds']) / pos_data['rec_yds'].replace(0, 1))) * 100
                position_metrics[f"{pos_key}rec_yd_mae"] = rec_yd_mae
                position_metrics[f"{pos_key}rec_yd_mape"] = rec_yd_mape
                
                if 'rec_td' in pos_data.columns and 'rec_tds' in pos_data.columns:
                    rec_td_mae = np.mean(np.abs(pos_data['rec_td'] - pos_data['rec_tds']))
                    position_metrics[f"{pos_key}rec_td_mae"] = rec_td_mae
        
        return position_metrics
    
    def run_backtest(self, start_week: int = 1, end_week: int = 17) -> List[Dict]:
        """
        Run backtesting for specified weeks.
        
        Args:
            start_week: Starting week number
            end_week: Ending week number
            
        Returns:
            List of accuracy results for each week
        """
        print(f"Running backtest for {self.test_season} using {self.reference_season} reference data")
        print(f"Testing weeks {start_week} to {end_week}")
        print("=" * 60)
        
        results = []
        
        for week in range(start_week, end_week + 1):
            print(f"\nProcessing Week {week}...")
            
            try:
                # Generate projections
                projections = self.generate_historical_projections(week)
                
                if projections.empty:
                    print(f"  No projections generated for Week {week}")
                    continue
                
                # Get actual results
                actuals = self.get_actual_results(week)
                
                if actuals.empty:
                    print(f"  No actual results found for Week {week}")
                    continue
                
                # Calculate accuracy
                accuracy = self.calculate_accuracy(projections, actuals)
                accuracy['week'] = week
                accuracy['season'] = self.test_season
                accuracy['reference_season'] = self.reference_season
                
                # Calculate Monte Carlo accuracy if enhanced projections are available
                mc_accuracy = self.calculate_monte_carlo_accuracy(projections, actuals)
                if 'error' not in mc_accuracy:
                    # Add Monte Carlo metrics to accuracy results
                    for key, value in mc_accuracy.items():
                        accuracy[f'mc_{key}'] = value
                
                results.append(accuracy)
                
                # Print summary
                if 'error' not in accuracy:
                    print(f"  Week {week} accuracy:")
                    for key, value in accuracy.items():
                        if key not in ['week', 'season', 'reference_season', 'total_players', 'matching_players']:
                            print(f"    {key}: {value:.2f}")
                else:
                    print(f"  Week {week} error: {accuracy['error']}")
                
            except Exception as e:
                print(f"  Error processing Week {week}: {e}")
                continue
        
        self.results = results
        print(f"\nBacktest complete! Processed {len(results)} weeks")
        return results
    
    def generate_csv_report(self, output_file: str = None) -> str:
        """
        Generate CSV report of backtest results.
        
        Args:
            output_file: Output file path (optional)
            
        Returns:
            Path to generated CSV file
        """
        if not self.results:
            print("No results to report. Run backtest first.")
            return None
        
        try:
            # Create DataFrame from results
            df = pd.DataFrame(self.results)
            
            # Generate filename if not provided
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"../backtest_results/backtest_results_{self.test_season}_{timestamp}.csv"
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            print(f"Backtest results saved to: {output_file}")
            
            # Print summary statistics
            self._print_summary_statistics(df)
            
            return output_file
            
        except Exception as e:
            print(f"Error generating CSV report: {e}")
            return None
    
    def _print_summary_statistics(self, df: pd.DataFrame):
        """Print summary statistics for the backtest results."""
        print("\nSummary Statistics:")
        print("=" * 40)
        
        # Calculate overall averages
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['week', 'season', 'reference_season', 'total_players', 'matching_players']]
        
        if len(numeric_columns) > 0:
            print("Overall Averages:")
            for col in numeric_columns:
                if col in df.columns:
                    avg_value = df[col].mean()
                    print(f"  {col}: {avg_value:.2f}")
        
        # Find best and worst weeks
        if 'pass_yd_mae' in df.columns:
            best_week = df.loc[df['pass_yd_mae'].idxmin(), 'week']
            worst_week = df.loc[df['pass_yd_mae'].idxmax(), 'week']
            print(f"\nBest Week: {best_week} (lowest pass_yd_mae)")
            print(f"Worst Week: {worst_week} (highest pass_yd_mae)")
        
        print(f"\nTotal weeks processed: {len(df)}")

def main():
    """Main function for running backtests."""
    if len(sys.argv) < 3:
        print("Usage: python backtester.py <test_season> <reference_season> [start_week] [end_week] [decay_coefficient] [scenario]")
        print("Example: python backtester.py 2023 2022 1 17 0.9 historical")
        print("Example: python backtester.py 2023 2022 5 10 0.85 current")
        sys.exit(1)
    
    test_season = int(sys.argv[1])
    reference_season = int(sys.argv[2])
    start_week = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    end_week = int(sys.argv[4]) if len(sys.argv) > 4 else 17
    decay_coefficient = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
    backtest_scenario = sys.argv[6] if len(sys.argv) > 6 else "historical"
    
    print("NFL Projection Backtester")
    print("=" * 40)
    print(f"Test Season: {test_season}")
    print(f"Reference Season: {reference_season}")
    print(f"Week Range: {start_week}-{end_week}")
    print(f"Scenario: {backtest_scenario}")
    print()
    
    try:
        # Create backtester
        backtester = NFLBacktester(test_season, reference_season, decay_coefficient, backtest_scenario)
        
        # Run backtest
        results = backtester.run_backtest(start_week, end_week)
        
        if results:
            # Generate CSV report
            output_file = backtester.generate_csv_report()
            print(f"\nBacktest complete! Results saved to: {output_file}")
        else:
            print("\nNo results generated. Check data availability and try again.")
            
    except Exception as e:
        print(f"Error running backtest: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
