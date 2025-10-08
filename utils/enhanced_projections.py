"""
Enhanced NFL Projection Engine
Leverages all available data sources for maximum accuracy with current data.

Key Improvements:
1. Weather-based adjustments
2. Betting market intelligence
3. Injury status integration
4. Multi-season historical analysis
5. Advanced time weighting
6. Opponent-specific matchups
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import os

class EnhancedProjectionEngine:
    def __init__(self):
        self.weather_effects = self._load_weather_effects()
        self.injury_impacts = self._load_injury_impacts()
        self.betting_intelligence = {}
        
    def _load_weather_effects(self):
        """Load weather impact coefficients based on historical analysis"""
        return {
            'temperature': {
                'pass_yd': 0.02,  # 2% increase per 10 degrees
                'rush_yd': 0.01,  # 1% increase per 10 degrees
                'rec_yd': 0.015   # 1.5% increase per 10 degrees
            },
            'wind': {
                'pass_yd': -0.05,  # 5% decrease per 10 mph wind
                'pass_td': -0.08,  # 8% decrease per 10 mph wind
                'rush_yd': 0.03    # 3% increase per 10 mph wind
            },
            'humidity': {
                'pass_yd': -0.01,  # 1% decrease per 10% humidity
                'rush_yd': 0.005   # 0.5% increase per 10% humidity
            }
        }
    
    def _load_injury_impacts(self):
        """Load injury impact coefficients"""
        return {
            'Out': 0.0,
            'Injured Reserve': 0.0,
            'Questionable': 0.7,  # 30% reduction
            'Doubtful': 0.4,     # 60% reduction
            'Probable': 0.9      # 10% reduction
        }
    
    def extract_weather_data(self, weather_string):
        """Extract weather data from string format"""
        if pd.isna(weather_string) or weather_string == '':
            return {'temp': 70, 'humidity': 50, 'wind': 0}
        
        # Extract temperature
        temp_match = re.search(r'(\d+) degrees', weather_string)
        temp = int(temp_match.group(1)) if temp_match else 70
        
        # Extract humidity
        humidity_match = re.search(r'humidity (\d+)%', weather_string)
        humidity = int(humidity_match.group(1)) if humidity_match else 50
        
        # Extract wind
        wind_match = re.search(r'wind (\d+) mph', weather_string)
        wind = int(wind_match.group(1)) if wind_match else 0
        
        return {'temp': temp, 'humidity': humidity, 'wind': wind}
    
    def apply_weather_adjustments(self, projections, weather_data):
        """Apply weather-based adjustments to projections"""
        if not weather_data:
            return projections
        
        adjusted = projections.copy()
        
        # Temperature adjustments
        temp_factor = (weather_data['temp'] - 70) / 10  # Normalize to 70 degrees
        for stat in ['pass_yd', 'rush_yd', 'rec_yd']:
            if stat in adjusted.columns:
                multiplier = 1 + (self.weather_effects['temperature'][stat] * temp_factor)
                adjusted[stat] *= multiplier
        
        # Wind adjustments
        wind_factor = weather_data['wind'] / 10  # Normalize to 10 mph
        for stat in ['pass_yd', 'pass_td', 'rush_yd']:
            if stat in adjusted.columns:
                multiplier = 1 + (self.weather_effects['wind'][stat] * wind_factor)
                adjusted[stat] *= multiplier
        
        # Humidity adjustments
        humidity_factor = (weather_data['humidity'] - 50) / 10  # Normalize to 50%
        for stat in ['pass_yd', 'rush_yd']:
            if stat in adjusted.columns:
                multiplier = 1 + (self.weather_effects['humidity'][stat] * humidity_factor)
                adjusted[stat] *= multiplier
        
        return adjusted
    
    def load_betting_intelligence(self, week):
        """Load betting market intelligence for the week"""
        try:
            # Load player props
            props_file = f"data/odds/week_{week:02d}/player_props_week_{week:02d}.csv"
            if os.path.exists(props_file):
                props_df = pd.read_csv(props_file)
                
                # Extract player projections from betting lines
                betting_projections = {}
                for _, row in props_df.iterrows():
                    player = row['player_name']
                    prop_type = row['prop_type']
                    point = row['point']
                    
                    if player not in betting_projections:
                        betting_projections[player] = {}
                    
                    # Map betting props to our stats
                    if 'rushing_yards' in prop_type:
                        betting_projections[player]['rush_yd'] = point
                    elif 'receiving_yards' in prop_type:
                        betting_projections[player]['rec_yd'] = point
                    elif 'passing_yards' in prop_type:
                        betting_projections[player]['pass_yd'] = point
                    elif 'rushing_tds' in prop_type:
                        betting_projections[player]['rush_td'] = point
                    elif 'receiving_tds' in prop_type:
                        betting_projections[player]['rec_td'] = point
                    elif 'passing_tds' in prop_type:
                        betting_projections[player]['pass_td'] = point
                
                return betting_projections
        except Exception as e:
            print(f"Warning: Could not load betting data: {e}")
            return {}
    
    def apply_betting_intelligence(self, projections, betting_data):
        """Apply betting market intelligence to projections"""
        if not betting_data:
            return projections
        
        adjusted = projections.copy()
        
        for idx, row in adjusted.iterrows():
            player_name = row.get('name', '')
            if player_name in betting_data:
                betting_proj = betting_data[player_name]
                
                # Blend with betting intelligence (60% our projection, 40% betting)
                for stat, betting_value in betting_proj.items():
                    if stat in adjusted.columns:
                        current_proj = row[stat]
                        blended = 0.6 * current_proj + 0.4 * betting_value
                        adjusted.loc[idx, stat] = blended
        
        return adjusted
    
    def load_injury_data(self):
        """Load current injury data"""
        try:
            injuries_df = pd.read_csv('data/injuries.csv')
            injury_dict = {}
            
            for _, row in injuries_df.iterrows():
                player_name = row['name']
                status = row['status']
                injury_dict[player_name] = status
            
            return injury_dict
        except Exception as e:
            print(f"Warning: Could not load injury data: {e}")
            return {}
    
    def apply_injury_adjustments(self, projections, injury_data):
        """Apply injury-based adjustments to projections"""
        if not injury_data:
            return projections
        
        adjusted = projections.copy()
        
        for idx, row in adjusted.iterrows():
            player_name = row.get('name', '')
            if player_name in injury_data:
                status = injury_data[player_name]
                impact_factor = self.injury_impacts.get(status, 1.0)
                
                # Apply injury impact to all projection columns
                projection_cols = [col for col in adjusted.columns if col.startswith('proj_')]
                for col in projection_cols:
                    adjusted.loc[idx, col] *= impact_factor
                
                # Add injury status to projections
                adjusted.loc[idx, 'injury_status'] = status
                adjusted.loc[idx, 'injury_impact'] = impact_factor
        
        return adjusted
    
    def calculate_opponent_specific_adjustments(self, projections, historical_data):
        """Calculate opponent-specific performance adjustments"""
        adjusted = projections.copy()
        
        for idx, row in adjusted.iterrows():
            player_name = row.get('name', '')
            team = row.get('team', '')
            opponent = row.get('next_op', '')
            
            if not opponent or opponent == 'BYE':
                continue
            
            # Get player's historical performance vs this opponent
            player_vs_opp = historical_data[
                (historical_data['player'] == player_name) & 
                (historical_data['opponent'] == opponent)
            ]
            
            if len(player_vs_opp) > 0:
                # Calculate opponent-specific multipliers
                for stat in ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td']:
                    if stat in player_vs_opp.columns:
                        # Get player's average vs this opponent vs overall average
                        opp_avg = player_vs_opp[stat].mean()
                        overall_avg = historical_data[historical_data['player'] == player_name][stat].mean()
                        
                        if overall_avg > 0:
                            multiplier = opp_avg / overall_avg
                            # Clamp multiplier to reasonable range
                            multiplier = max(0.5, min(2.0, multiplier))
                            
                            # Apply to projections
                            proj_col = f'proj_{stat}'
                            if proj_col in adjusted.columns:
                                adjusted.loc[idx, proj_col] *= multiplier
                                adjusted.loc[idx, f'opp_multiplier_{stat}'] = multiplier
        
        return adjusted
    
    def calculate_momentum_adjustments(self, projections, historical_data):
        """Calculate recent performance momentum"""
        adjusted = projections.copy()
        
        for idx, row in adjusted.iterrows():
            player_name = row.get('name', '')
            
            # Get player's last 3 games vs previous 3 games
            player_data = historical_data[historical_data['player'] == player_name].copy()
            player_data = player_data.sort_values('week', ascending=False)
            
            if len(player_data) >= 6:
                recent_3 = player_data.head(3)
                previous_3 = player_data.iloc[3:6]
                
                for stat in ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td']:
                    if stat in player_data.columns:
                        recent_avg = recent_3[stat].mean()
                        previous_avg = previous_3[stat].mean()
                        
                        if previous_avg > 0:
                            momentum_factor = recent_avg / previous_avg
                            # Clamp to reasonable range
                            momentum_factor = max(0.7, min(1.3, momentum_factor))
                            
                            # Apply momentum to projections
                            proj_col = f'proj_{stat}'
                            if proj_col in adjusted.columns:
                                adjusted.loc[idx, proj_col] *= momentum_factor
                                adjusted.loc[idx, f'momentum_{stat}'] = momentum_factor
        
        return adjusted
    
    def enhance_projections(self, base_projections, week, historical_data):
        """Apply all enhancements to base projections"""
        print("Applying enhanced projection improvements...")
        
        enhanced = base_projections.copy()
        
        # 1. Load betting intelligence
        betting_data = self.load_betting_intelligence(week)
        if betting_data:
            enhanced = self.apply_betting_intelligence(enhanced, betting_data)
            print(f"Applied betting intelligence for {len(betting_data)} players")
        
        # 2. Load and apply injury adjustments
        injury_data = self.load_injury_data()
        if injury_data:
            enhanced = self.apply_injury_adjustments(enhanced, injury_data)
            print(f"Applied injury adjustments for {len(injury_data)} players")
        
        # 3. Apply opponent-specific adjustments
        enhanced = self.calculate_opponent_specific_adjustments(enhanced, historical_data)
        print("Applied opponent-specific adjustments")
        
        # 4. Apply momentum adjustments
        enhanced = self.calculate_momentum_adjustments(enhanced, historical_data)
        print("Applied momentum adjustments")
        
        # 5. Apply weather adjustments (if weather data available)
        if 'weather' in historical_data.columns:
            # Get weather for the week
            week_weather = historical_data[historical_data['week'] == week]['weather'].iloc[0] if len(historical_data[historical_data['week'] == week]) > 0 else None
            if week_weather:
                weather_data = self.extract_weather_data(week_weather)
                enhanced = self.apply_weather_adjustments(enhanced, weather_data)
                print("Applied weather adjustments")
        
        return enhanced

def integrate_enhanced_projections():
    """Integrate enhanced projections into the main projection engine"""
    # This function would be called from projection.py
    pass
