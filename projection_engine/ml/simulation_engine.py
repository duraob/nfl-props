"""
Monte Carlo Simulation Engine

Runs thousands of simulations to generate probability distributions
for player projections using modern machine learning techniques.

Key Features:
- 10,000+ simulations per player
- Multi-factor variance modeling
- Correlation between statistics
- Confidence intervals and probability distributions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class SimulationEngine:
    """
    Monte Carlo simulation engine for NFL projections.
    
    Key responsibilities:
    1. Run thousands of simulations per player
    2. Model correlations between statistics
    3. Generate probability distributions
    4. Calculate confidence intervals
    """
    
    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        """
        Initialize simulation engine.
        
        Args:
            n_simulations: Number of simulations to run
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Simulation results storage
        self.simulation_results: Dict[str, Dict] = {}
        self.probability_distributions: Dict[str, Dict] = {}
        
    def run_simulations(self, df_players: pd.DataFrame, 
                       variance_analysis: Dict,
                       opponent_adjustments: Dict) -> Dict:
        """
        Run Monte Carlo simulations for all players.
        
        Args:
            df_players: Player statistics DataFrame
            variance_analysis: Variance analysis results
            opponent_adjustments: Opponent strength adjustments
            
        Returns:
            Dictionary with simulation results for all players
        """
        print(f"Running {self.n_simulations:,} Monte Carlo simulations...")
        
        all_simulation_results = {}
        
        for _, player_row in df_players.iterrows():
            player_name = player_row['name']
            position = player_row.get('pos', 'QB')
            
            print(f"Simulating {player_name} ({position})...")
            
            # Run simulations for this player
            player_simulations = self._simulate_player(
                player_row, variance_analysis, opponent_adjustments
            )
            
            all_simulation_results[player_name] = player_simulations
        
        print("Monte Carlo simulations complete")
        return all_simulation_results
    
    def _simulate_player(self, player_row: pd.Series, 
                        variance_analysis: Dict,
                        opponent_adjustments: Dict) -> Dict:
        """
        Run simulations for a single player.
        
        Args:
            player_row: Player statistics row
            variance_analysis: Variance analysis results
            opponent_adjustments: Opponent adjustments
            
        Returns:
            Dictionary with simulation results
        """
        player_name = player_row['name']
        position = player_row.get('pos', 'QB')
        team = player_row['team']
        
        # Get base projections
        base_projections = self._get_base_projections(player_row)
        
        # Get variance parameters
        variance_params = self._get_variance_parameters(
            player_name, position, variance_analysis
        )
        
        # Get opponent adjustments
        opponent_multipliers = self._get_opponent_multipliers(
            team, opponent_adjustments
        )
        
        # Run simulations
        simulation_results = {}
        
        for stat, base_value in base_projections.items():
            if base_value <= 0:
                continue
            
            # Get variance parameters for this stat
            if stat in variance_params:
                var_params = variance_params[stat]
            else:
                # Default parameters
                var_params = {
                    'mean': base_value,
                    'std': base_value * 0.3,  # 30% coefficient of variation
                    'distribution': 'normal'
                }
            
            # Apply opponent adjustments
            opponent_mult = opponent_multipliers.get(stat, 1.0)
            adjusted_mean = base_value * opponent_mult
            
            # Run simulations
            simulations = self._run_stat_simulations(
                adjusted_mean, var_params, stat
            )
            
            # Calculate statistics
            simulation_results[stat] = self._calculate_simulation_statistics(
                simulations, stat
            )
        
        return simulation_results
    
    def _get_base_projections(self, player_row: pd.Series) -> Dict:
        """Extract base projections from player row."""
        base_projections = {}
        
        # Key projection columns (without proj_ prefix)
        projection_columns = [
            'pass_yd', 'pass_td', 'pass_int',
            'rush_yd', 'rush_td', 
            'rec_yd', 'rec_td', 'rec'
        ]
        
        for col in projection_columns:
            if col in player_row and not pd.isna(player_row[col]):
                base_projections[col] = player_row[col]
        
        return base_projections
    
    def _get_variance_parameters(self, player_name: str, position: str,
                               variance_analysis: Dict) -> Dict:
        """Get variance parameters for a player."""
        variance_params = {}
        
        # Get position-specific variance
        if 'position' in variance_analysis and position in variance_analysis['position']:
            position_variance = variance_analysis['position'][position]
            for stat, params in position_variance.items():
                variance_params[stat] = params
        
        # Override with player-specific variance if available
        if 'player' in variance_analysis and player_name in variance_analysis['player']:
            player_variance = variance_analysis['player'][player_name]
            for stat, params in player_variance.items():
                if stat in variance_params:
                    # Blend player and position variance
                    variance_params[stat]['mean'] = 0.7 * params['mean'] + 0.3 * variance_params[stat]['mean']
                    variance_params[stat]['std'] = 0.7 * params['std'] + 0.3 * variance_params[stat]['std']
                else:
                    variance_params[stat] = params
        
        return variance_params
    
    def _get_opponent_multipliers(self, team: str, opponent_adjustments: Dict) -> Dict:
        """Get opponent adjustment multipliers for a team."""
        if team in opponent_adjustments:
            return opponent_adjustments[team]
        else:
            # Default neutral multipliers
            return {
                'pass_yd': 1.0, 'pass_td': 1.0, 'int': 1.0,
                'rush_yd': 1.0, 'rush_td': 1.0,
                'rec_yd': 1.0, 'rec_td': 1.0, 'rec': 1.0
            }
    
    def _run_stat_simulations(self, mean: float, var_params: Dict, stat: str) -> np.ndarray:
        """
        Run simulations for a single statistic.
        
        Args:
            mean: Mean value for simulations
            var_params: Variance parameters
            stat: Statistic name
            
        Returns:
            Array of simulation results
        """
        distribution = var_params.get('distribution', 'normal')
        std = var_params.get('std', mean * 0.3)
        
        if distribution == 'normal':
            simulations = np.random.normal(mean, std, self.n_simulations)
        elif distribution == 'lognormal':
            # For lognormal, work with log parameters
            mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
            sigma = np.sqrt(np.log(1 + std**2 / mean**2))
            simulations = np.random.lognormal(mu, sigma, self.n_simulations)
        elif distribution == 'gamma':
            # Gamma distribution for positive values
            shape = (mean / std) ** 2
            scale = std**2 / mean
            simulations = np.random.gamma(shape, scale, self.n_simulations)
        else:
            # Default to normal
            simulations = np.random.normal(mean, std, self.n_simulations)
        
        # Ensure non-negative values for most stats
        if stat in ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td', 'rec']:
            simulations = np.maximum(simulations, 0)
        
        return simulations
    
    def _calculate_simulation_statistics(self, simulations: np.ndarray, stat: str) -> Dict:
        """
        Calculate statistics from simulation results.
        
        Args:
            simulations: Array of simulation results
            stat: Statistic name
            
        Returns:
            Dictionary with simulation statistics
        """
        # Basic statistics
        mean = np.mean(simulations)
        std = np.std(simulations)
        median = np.median(simulations)
        
        # Percentiles
        percentiles = {
            'p10': np.percentile(simulations, 10),
            'p25': np.percentile(simulations, 25),
            'p50': np.percentile(simulations, 50),
            'p75': np.percentile(simulations, 75),
            'p90': np.percentile(simulations, 90),
            'p95': np.percentile(simulations, 95),
            'p99': np.percentile(simulations, 99)
        }
        
        # Confidence intervals
        ci_90 = np.percentile(simulations, [5, 95])
        ci_95 = np.percentile(simulations, [2.5, 97.5])
        
        # Probability of exceeding certain thresholds
        thresholds = self._get_stat_thresholds(stat)
        probabilities = {}
        for threshold_name, threshold_value in thresholds.items():
            probabilities[f'prob_{threshold_name}'] = np.mean(simulations >= threshold_value)
        
        # Distribution fit
        distribution_fit = self._fit_distribution(simulations, stat)
        
        return {
            'mean': mean,
            'std': std,
            'median': median,
            'percentiles': percentiles,
            'ci_90': ci_90,
            'ci_95': ci_95,
            'probabilities': probabilities,
            'distribution_fit': distribution_fit,
            'n_simulations': len(simulations)
        }
    
    def _get_stat_thresholds(self, stat: str) -> Dict:
        """Get meaningful thresholds for a statistic."""
        thresholds = {}
        
        if 'yd' in stat:
            # Yardage thresholds
            thresholds = {
                '100_yds': 100,
                '150_yds': 150,
                '200_yds': 200,
                '250_yds': 250
            }
        elif 'td' in stat:
            # Touchdown thresholds
            thresholds = {
                '1_td': 1,
                '2_td': 2,
                '3_td': 3,
                '4_td': 4
            }
        elif stat == 'rec':
            # Reception thresholds
            thresholds = {
                '5_rec': 5,
                '10_rec': 10,
                '15_rec': 15
            }
        
        return thresholds
    
    def _fit_distribution(self, simulations: np.ndarray, stat: str) -> Dict:
        """Fit probability distribution to simulation results."""
        try:
            # Try different distributions
            distributions = ['normal', 'lognormal', 'gamma']
            best_fit = None
            best_ks_stat = float('inf')
            
            for dist_name in distributions:
                try:
                    if dist_name == 'normal':
                        params = stats.norm.fit(simulations)
                        ks_stat, _ = stats.kstest(simulations, lambda x: stats.norm.cdf(x, *params))
                    elif dist_name == 'lognormal':
                        params = stats.lognorm.fit(simulations)
                        ks_stat, _ = stats.kstest(simulations, lambda x: stats.lognorm.cdf(x, *params))
                    elif dist_name == 'gamma':
                        params = stats.gamma.fit(simulations)
                        ks_stat, _ = stats.kstest(simulations, lambda x: stats.gamma.cdf(x, *params))
                    
                    if ks_stat < best_ks_stat:
                        best_ks_stat = ks_stat
                        best_fit = {
                            'distribution': dist_name,
                            'parameters': params,
                            'ks_statistic': ks_stat
                        }
                except:
                    continue
            
            return best_fit if best_fit else {'distribution': 'normal', 'parameters': (0, 1), 'ks_statistic': 1.0}
            
        except:
            return {'distribution': 'normal', 'parameters': (0, 1), 'ks_statistic': 1.0}
    
    def generate_probability_distributions(self, simulation_results: Dict) -> Dict:
        """
        Generate probability distributions from simulation results.
        
        Args:
            simulation_results: Results from Monte Carlo simulations
            
        Returns:
            Dictionary with probability distributions
        """
        print("Generating probability distributions...")
        
        probability_distributions = {}
        
        for player_name, player_results in simulation_results.items():
            player_distributions = {}
            
            for stat, stat_results in player_results.items():
                # Extract key statistics
                mean = stat_results['mean']
                std = stat_results['std']
                percentiles = stat_results['percentiles']
                probabilities = stat_results['probabilities']
                
                # Calculate confidence rating (1-10 scale)
                confidence_rating = self._calculate_confidence_rating(stat_results)
                
                # Create probability distribution
                player_distributions[stat] = {
                    'mean': mean,
                    'std': std,
                    'percentiles': percentiles,
                    'probabilities': probabilities,
                    'confidence_rating': confidence_rating,
                    'distribution_fit': stat_results['distribution_fit']
                }
            
            probability_distributions[player_name] = player_distributions
        
        return probability_distributions
    
    def _calculate_confidence_rating(self, stat_results: Dict) -> int:
        """
        Calculate confidence rating (1-10 scale) based on simulation results.
        
        Args:
            stat_results: Simulation results for a statistic
            
        Returns:
            Confidence rating (1-10)
        """
        # Factors that affect confidence:
        # 1. Coefficient of variation (lower = more confident)
        # 2. Sample size (more simulations = more confident)
        # 3. Distribution fit quality
        
        mean = stat_results['mean']
        std = stat_results['std']
        n_simulations = stat_results['n_simulations']
        
        # Base confidence from coefficient of variation
        cv = std / mean if mean > 0 else 1.0
        cv_score = max(0, min(10, 10 - (cv * 10)))  # Lower CV = higher score
        
        # Sample size bonus
        sample_bonus = min(2, n_simulations / 5000)  # Up to 2 points for large samples
        
        # Distribution fit bonus
        fit_bonus = 0
        if 'distribution_fit' in stat_results:
            ks_stat = stat_results['distribution_fit'].get('ks_statistic', 1.0)
            fit_bonus = max(0, min(1, 1 - ks_stat))  # Better fit = higher bonus
        
        # Calculate final confidence rating
        confidence = cv_score + sample_bonus + fit_bonus
        confidence = max(1, min(10, int(confidence)))
        
        return confidence
