#!/usr/bin/env python3
"""
Backtesting Configuration Module

Provides configuration settings for different backtesting scenarios:
- Historical backtesting: Use all available players for maximum validation
- Current projections: Use current season roster for production accuracy
"""

class BacktestConfig:
    """
    Configuration class for backtesting scenarios.
    
    Attributes:
        use_roster_filter (bool): Whether to filter players by current roster
        min_games (int): Minimum games required for player inclusion
        min_snap_pct (float): Minimum snap percentage for player inclusion
        position_filter (list): Positions to include in backtesting
    """
    
    def __init__(self, scenario="historical"):
        """
        Initialize backtesting configuration.
        
        Args:
            scenario (str): Configuration scenario - "historical" or "current"
        """
        if scenario == "historical":
            self._setup_historical_config()
        elif scenario == "current":
            self._setup_current_config()
        else:
            raise ValueError("Scenario must be 'historical' or 'current'")
    
    def _setup_historical_config(self):
        """Configure for historical backtesting with maximum data coverage."""
        self.use_roster_filter = False
        self.min_games = 2  # Lower threshold for historical data
        self.min_snap_pct = 10.0  # Lower threshold for historical data
        self.position_filter = ['QB', 'RB', 'WR', 'TE']  # Fantasy relevant positions only
        self.scenario_name = "Historical Backtesting"
    
    def _setup_current_config(self):
        """Configure for current season projections with production accuracy."""
        self.use_roster_filter = True
        self.min_games = 3  # Higher threshold for current season
        self.min_snap_pct = 20.0  # Higher threshold for current season
        self.position_filter = ['QB', 'RB', 'WR', 'TE']  # Fantasy relevant positions only
        self.scenario_name = "Current Season Projections"
    
    def get_filter_description(self):
        """
        Get human-readable description of current filtering configuration.
        
        Returns:
            str: Description of filtering settings
        """
        roster_status = "Current roster only" if self.use_roster_filter else "All historical players"
        return f"{self.scenario_name}: {roster_status}, Min {self.min_games} games, Min {self.min_snap_pct}% snaps"

def get_backtest_config(scenario="historical"):
    """
    Factory function to get backtesting configuration.
    
    Args:
        scenario (str): Configuration scenario - "historical" or "current"
    
    Returns:
        BacktestConfig: Configured backtesting settings
    """
    return BacktestConfig(scenario)
