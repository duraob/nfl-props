"""
Probability Distribution Engine

Generates probability distributions and confidence ratings from Monte Carlo simulations.
Provides sophisticated probability analysis for NFL projections.

Key Features:
- Multiple distribution types (normal, lognormal, gamma, beta)
- Confidence intervals and percentiles
- Risk assessment and upside/downside scenarios
- Probability of exceeding thresholds
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ProbabilityEngine:
    """
    Generates probability distributions and confidence ratings.
    
    Key responsibilities:
    1. Create probability distributions from simulation results
    2. Calculate confidence ratings (1-10 scale)
    3. Generate risk assessments
    4. Provide upside/downside scenarios
    """
    
    def __init__(self):
        self.distribution_types = ['normal', 'lognormal', 'gamma', 'beta']
        self.confidence_thresholds = {
            'very_low': (1, 3),
            'low': (3, 5),
            'medium': (5, 7),
            'high': (7, 9),
            'very_high': (9, 10)
        }
        
    def generate_probability_distributions(self, simulation_results: Dict) -> Dict:
        """
        Generate comprehensive probability distributions from simulation results.
        
        Args:
            simulation_results: Results from Monte Carlo simulations
            
        Returns:
            Dictionary with probability distributions for all players
        """
        print("Generating probability distributions...")
        
        probability_distributions = {}
        
        for player_name, player_results in simulation_results.items():
            print(f"Processing probability distributions for {player_name}...")
            
            player_distributions = {}
            
            for stat, stat_results in player_results.items():
                # Generate probability distribution
                prob_dist = self._create_probability_distribution(stat_results, stat)
                player_distributions[stat] = prob_dist
            
            probability_distributions[player_name] = player_distributions
        
        print("Probability distributions generated")
        return probability_distributions
    
    def _create_probability_distribution(self, stat_results: Dict, stat: str) -> Dict:
        """
        Create probability distribution for a single statistic.
        
        Args:
            stat_results: Simulation results for a statistic
            stat: Statistic name
            
        Returns:
            Dictionary with probability distribution
        """
        # Extract key statistics
        mean = stat_results['mean']
        std = stat_results['std']
        percentiles = stat_results['percentiles']
        probabilities = stat_results['probabilities']
        distribution_fit = stat_results['distribution_fit']
        
        # Calculate confidence rating
        confidence_rating = self._calculate_confidence_rating(stat_results)
        
        # Generate risk assessment
        risk_assessment = self._assess_risk(stat_results, stat)
        
        # Calculate upside/downside scenarios
        upside_downside = self._calculate_upside_downside(stat_results, stat)
        
        # Create probability distribution
        prob_distribution = {
            'mean': mean,
            'std': std,
            'percentiles': percentiles,
            'probabilities': probabilities,
            'confidence_rating': confidence_rating,
            'confidence_level': self._get_confidence_level(confidence_rating),
            'risk_assessment': risk_assessment,
            'upside_downside': upside_downside,
            'distribution_fit': distribution_fit,
            'statistic': stat
        }
        
        return prob_distribution
    
    def _calculate_confidence_rating(self, stat_results: Dict) -> int:
        """
        Calculate confidence rating (1-10 scale).
        
        Args:
            stat_results: Simulation results
            
        Returns:
            Confidence rating (1-10)
        """
        mean = stat_results['mean']
        std = stat_results['std']
        n_simulations = stat_results['n_simulations']
        
        # Base confidence from coefficient of variation
        cv = std / mean if mean > 0 else 1.0
        cv_score = max(0, min(8, 8 - (cv * 8)))  # Lower CV = higher score
        
        # Sample size bonus
        sample_bonus = min(1, n_simulations / 10000)  # Bonus for large samples
        
        # Distribution fit quality
        fit_bonus = 0
        if 'distribution_fit' in stat_results:
            ks_stat = stat_results['distribution_fit'].get('ks_statistic', 1.0)
            fit_bonus = max(0, min(1, 1 - ks_stat))
        
        # Calculate final rating
        confidence = cv_score + sample_bonus + fit_bonus
        return max(1, min(10, int(confidence)))
    
    def _get_confidence_level(self, confidence_rating: int) -> str:
        """Get confidence level description."""
        for level, (min_val, max_val) in self.confidence_thresholds.items():
            if min_val <= confidence_rating < max_val:
                return level
        return 'very_high' if confidence_rating >= 9 else 'very_low'
    
    def _assess_risk(self, stat_results: Dict, stat: str) -> Dict:
        """
        Assess risk factors for a statistic.
        
        Args:
            stat_results: Simulation results
            stat: Statistic name
            
        Returns:
            Risk assessment dictionary
        """
        mean = stat_results['mean']
        std = stat_results['std']
        percentiles = stat_results['percentiles']
        
        # Calculate risk metrics
        cv = std / mean if mean > 0 else 1.0
        
        # Volatility risk (coefficient of variation)
        if cv < 0.2:
            volatility_risk = 'low'
        elif cv < 0.4:
            volatility_risk = 'medium'
        else:
            volatility_risk = 'high'
        
        # Downside risk (probability of very low performance)
        p10 = percentiles['p10']
        downside_risk = 'low' if p10 > mean * 0.5 else 'high'
        
        # Upside potential (probability of exceeding 150% of mean)
        upside_threshold = mean * 1.5
        upside_prob = np.mean(np.array(stat_results.get('simulations', [mean])) >= upside_threshold)
        
        if upside_prob > 0.3:
            upside_potential = 'high'
        elif upside_prob > 0.15:
            upside_potential = 'medium'
        else:
            upside_potential = 'low'
        
        return {
            'volatility_risk': volatility_risk,
            'downside_risk': downside_risk,
            'upside_potential': upside_potential,
            'coefficient_of_variation': cv,
            'upside_probability': upside_prob
        }
    
    def _calculate_upside_downside(self, stat_results: Dict, stat: str) -> Dict:
        """
        Calculate upside and downside scenarios.
        
        Args:
            stat_results: Simulation results
            stat: Statistic name
            
        Returns:
            Upside/downside scenarios
        """
        mean = stat_results['mean']
        percentiles = stat_results['percentiles']
        
        # Define scenarios based on percentiles
        scenarios = {
            'worst_case': percentiles['p5'],  # 5th percentile
            'downside': percentiles['p25'],   # 25th percentile
            'baseline': percentiles['p50'],   # 50th percentile (median)
            'upside': percentiles['p75'],     # 75th percentile
            'best_case': percentiles['p95']   # 95th percentile
        }
        
        # Calculate scenario probabilities
        scenario_probabilities = {
            'worst_case_prob': 0.05,  # 5% chance
            'downside_prob': 0.25,    # 25% chance
            'baseline_prob': 0.50,    # 50% chance
            'upside_prob': 0.25,      # 25% chance
            'best_case_prob': 0.05    # 5% chance
        }
        
        # Calculate expected values for each scenario
        expected_values = {}
        for scenario, value in scenarios.items():
            expected_values[f'{scenario}_value'] = value
            expected_values[f'{scenario}_vs_mean'] = (value - mean) / mean if mean > 0 else 0
        
        return {
            'scenarios': scenarios,
            'probabilities': scenario_probabilities,
            'expected_values': expected_values
        }
    
    def create_confidence_matrix(self, probability_distributions: Dict) -> pd.DataFrame:
        """
        Create confidence matrix for all players and statistics.
        
        Args:
            probability_distributions: Probability distributions for all players
            
        Returns:
            DataFrame with confidence ratings
        """
        confidence_data = []
        
        for player_name, player_dists in probability_distributions.items():
            for stat, prob_dist in player_dists.items():
                confidence_data.append({
                    'player': player_name,
                    'statistic': stat,
                    'confidence_rating': prob_dist['confidence_rating'],
                    'confidence_level': prob_dist['confidence_level'],
                    'mean': prob_dist['mean'],
                    'std': prob_dist['std'],
                    'cv': prob_dist['std'] / prob_dist['mean'] if prob_dist['mean'] > 0 else 0
                })
        
        return pd.DataFrame(confidence_data)
    
    def generate_risk_report(self, probability_distributions: Dict) -> Dict:
        """
        Generate comprehensive risk report.
        
        Args:
            probability_distributions: Probability distributions for all players
            
        Returns:
            Risk report dictionary
        """
        print("Generating risk report...")
        
        risk_report = {
            'high_risk_players': [],
            'low_confidence_projections': [],
            'volatile_statistics': [],
            'summary': {}
        }
        
        # Analyze each player
        for player_name, player_dists in probability_distributions.items():
            player_risks = []
            player_confidences = []
            
            for stat, prob_dist in player_dists.items():
                # Check for high risk
                risk_assessment = prob_dist['risk_assessment']
                if risk_assessment['volatility_risk'] == 'high':
                    player_risks.append(stat)
                
                # Check for low confidence
                if prob_dist['confidence_rating'] < 5:
                    player_confidences.append(stat)
            
            # Add to risk report if applicable
            if player_risks:
                risk_report['high_risk_players'].append({
                    'player': player_name,
                    'risky_stats': player_risks
                })
            
            if player_confidences:
                risk_report['low_confidence_projections'].append({
                    'player': player_name,
                    'low_confidence_stats': player_confidences
                })
        
        # Generate summary statistics
        all_confidences = []
        all_cvs = []
        
        for player_dists in probability_distributions.values():
            for prob_dist in player_dists.values():
                all_confidences.append(prob_dist['confidence_rating'])
                all_cvs.append(prob_dist['std'] / prob_dist['mean'] if prob_dist['mean'] > 0 else 0)
        
        risk_report['summary'] = {
            'average_confidence': np.mean(all_confidences),
            'average_cv': np.mean(all_cvs),
            'total_projections': len(all_confidences),
            'high_confidence_count': sum(1 for c in all_confidences if c >= 7),
            'low_confidence_count': sum(1 for c in all_confidences if c < 5)
        }
        
        return risk_report
    
    def export_probability_distributions(self, probability_distributions: Dict, 
                                       output_file: str = "data/probability_distributions.json"):
        """
        Export probability distributions to JSON file.
        
        Args:
            probability_distributions: Probability distributions
            output_file: Output file path
        """
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert all numpy types
        converted_distributions = {}
        for player_name, player_dists in probability_distributions.items():
            converted_distributions[player_name] = {}
            for stat, prob_dist in player_dists.items():
                converted_distributions[player_name][stat] = {}
                for key, value in prob_dist.items():
                    converted_distributions[player_name][stat][key] = convert_numpy(value)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(converted_distributions, f, indent=2)
        
        print(f"Probability distributions exported to {output_file}")
    
    def get_player_summary(self, player_name: str, 
                          probability_distributions: Dict) -> Dict:
        """
        Get summary for a specific player.
        
        Args:
            player_name: Player name
            probability_distributions: Probability distributions
            
        Returns:
            Player summary dictionary
        """
        if player_name not in probability_distributions:
            return {}
        
        player_dists = probability_distributions[player_name]
        summary = {
            'player': player_name,
            'statistics': {},
            'overall_confidence': 0,
            'risk_level': 'medium'
        }
        
        confidences = []
        risks = []
        
        for stat, prob_dist in player_dists.items():
            summary['statistics'][stat] = {
                'mean': prob_dist['mean'],
                'confidence_rating': prob_dist['confidence_rating'],
                'confidence_level': prob_dist['confidence_level'],
                'risk_assessment': prob_dist['risk_assessment']
            }
            
            confidences.append(prob_dist['confidence_rating'])
            risks.append(prob_dist['risk_assessment']['volatility_risk'])
        
        # Calculate overall metrics
        summary['overall_confidence'] = np.mean(confidences)
        summary['risk_level'] = 'high' if 'high' in risks else 'low' if all(r == 'low' for r in risks) else 'medium'
        
        return summary
