"""
NFL Projection Engine - Optimized Modular Architecture

A sophisticated NFL projection system using machine learning,
Monte Carlo simulations, and probability distributions.

Key Features:
- Active roster and injury filtering
- Schedule strength normalization
- Opponent-specific matchup analysis
- Historical variance modeling
- Monte Carlo simulations (10,000+ runs)
- Probability distributions with confidence ratings
- Real-time learning and adaptation
"""

from .core.data_loader import DataLoader
from .core.schedule_analyzer import ScheduleAnalyzer
from .core.opponent_analyzer import OpponentAnalyzer
from .core.base_projections import BaseProjections
from .ml.variance_analyzer import VarianceAnalyzer
from .ml.simulation_engine import SimulationEngine
from .ml.probability_engine import ProbabilityEngine
from .projection_engine import ProjectionEngine

__version__ = "2.0.0"
__author__ = "NFL Projection Team"

__all__ = [
    'DataLoader',
    'ScheduleAnalyzer', 
    'OpponentAnalyzer',
    'BaseProjections',
    'VarianceAnalyzer',
    'SimulationEngine',
    'ProbabilityEngine',
    'ProjectionEngine'
]
