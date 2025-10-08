"""
Injury Integration for NFL Projections

This module integrates injury data from injuries.py into the projection engine
to improve accuracy by accounting for player availability and injury impacts.

Key Features:
- Load injury data from existing injuries.py output
- Apply injury impact coefficients
- Remove injured players from projections
- Adjust projections based on injury severity
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional

class InjuryIntegration:
    def __init__(self):
        self.injury_impacts = {
            'Out': 0.0,                    # No projection
            'Injured Reserve': 0.0,        # No projection
            'Questionable': 0.7,           # 30% reduction
            'Doubtful': 0.4,               # 60% reduction
            'Probable': 0.9,               # 10% reduction
            'Active': 1.0                  # No reduction
        }
        
    def load_injury_data(self) -> pd.DataFrame:
        """Load injury data from injuries.py output"""
        try:
            injury_file = 'data/injuries.csv'
            if not os.path.exists(injury_file):
                print(f"Warning: Injury file not found at {injury_file}")
                return pd.DataFrame()
            
            injuries_df = pd.read_csv(injury_file)
            print(f"Loaded injury data for {len(injuries_df)} players")
            
            # Clean and validate data
            injuries_df = injuries_df.dropna(subset=['name', 'status'])
            injuries_df['status'] = injuries_df['status'].str.strip()
            
            return injuries_df
            
        except Exception as e:
            print(f"Error loading injury data: {e}")
            return pd.DataFrame()
    
    def create_injury_mapping(self, injuries_df: pd.DataFrame) -> Dict:
        """Create mapping of player names to injury status"""
        if injuries_df.empty:
            return {}
        
        injury_mapping = {}
        
        for _, row in injuries_df.iterrows():
            player_name = row['name'].strip()
            status = row['status'].strip()
            
            # Handle multiple entries for same player (take most recent)
            if player_name in injury_mapping:
                # Keep more severe status
                current_status = injury_mapping[player_name]
                if self._is_more_severe(status, current_status):
                    injury_mapping[player_name] = status
            else:
                injury_mapping[player_name] = status
        
        print(f"Created injury mapping for {len(injury_mapping)} players")
        return injury_mapping
    
    def _is_more_severe(self, status1: str, status2: str) -> bool:
        """Determine if status1 is more severe than status2"""
        severity_order = ['Out', 'Injured Reserve', 'Doubtful', 'Questionable', 'Probable', 'Active']
        
        try:
            idx1 = severity_order.index(status1)
            idx2 = severity_order.index(status2)
            return idx1 < idx2  # Lower index = more severe
        except ValueError:
            return False
    
    def apply_injury_adjustments(self, projections: pd.DataFrame, 
                                injury_mapping: Dict) -> pd.DataFrame:
        """Apply injury-based adjustments to projections"""
        print("Applying injury adjustments to projections...")
        
        enhanced_projections = projections.copy()
        
        # Add injury columns
        enhanced_projections['injury_status'] = 'Active'
        enhanced_projections['injury_impact'] = 1.0
        enhanced_projections['injury_adjusted'] = False
        
        players_removed = 0
        players_adjusted = 0
        
        for idx, row in enhanced_projections.iterrows():
            player_name = row.get('name', '')
            
            if not player_name or player_name not in injury_mapping:
                continue
            
            status = injury_mapping[player_name]
            impact_factor = self.injury_impacts.get(status, 1.0)
            
            # Update injury columns
            enhanced_projections.loc[idx, 'injury_status'] = status
            enhanced_projections.loc[idx, 'injury_impact'] = impact_factor
            
            # Apply injury impact
            if impact_factor == 0.0:
                # Remove player completely (Out, IR)
                projection_cols = [col for col in enhanced_projections.columns if col.startswith('proj_')]
                for col in projection_cols:
                    enhanced_projections.loc[idx, col] = 0.0
                
                enhanced_projections.loc[idx, 'injury_adjusted'] = True
                players_removed += 1
                
            elif impact_factor < 1.0:
                # Reduce projections based on injury severity
                projection_cols = [col for col in enhanced_projections.columns if col.startswith('proj_')]
                for col in projection_cols:
                    enhanced_projections.loc[idx, col] *= impact_factor
                
                enhanced_projections.loc[idx, 'injury_adjusted'] = True
                players_adjusted += 1
        
        print(f"Injury adjustments applied:")
        print(f"  - Players removed (Out/IR): {players_removed}")
        print(f"  - Players adjusted: {players_adjusted}")
        print(f"  - Total players with injury data: {len(injury_mapping)}")
        
        return enhanced_projections
    
    def filter_injured_players(self, projections: pd.DataFrame, 
                              injury_mapping: Dict) -> pd.DataFrame:
        """Filter out injured players from projections"""
        print("Filtering injured players from projections...")
        
        # Get players to remove (Out, IR)
        players_to_remove = []
        for player, status in injury_mapping.items():
            if status in ['Out', 'Injured Reserve']:
                players_to_remove.append(player)
        
        # Filter projections
        if players_to_remove:
            filtered_projections = projections[~projections['name'].isin(players_to_remove)]
            removed_count = len(projections) - len(filtered_projections)
            print(f"Removed {removed_count} injured players from projections")
            return filtered_projections
        else:
            print("No injured players to remove")
            return projections
    
    def generate_injury_report(self, projections: pd.DataFrame, 
                              injury_mapping: Dict) -> pd.DataFrame:
        """Generate detailed injury impact report"""
        print("Generating injury impact report...")
        
        injury_report = []
        
        for idx, row in projections.iterrows():
            player_name = row.get('name', '')
            
            if player_name in injury_mapping:
                status = injury_mapping[player_name]
                impact_factor = self.injury_impacts.get(status, 1.0)
                
                # Get key projection stats
                key_stats = {
                    'pass_yd': row.get('proj_pass_yd', 0),
                    'rush_yd': row.get('proj_rush_yd', 0),
                    'rec_yd': row.get('proj_rec_yd', 0),
                    'pass_td': row.get('proj_pass_td', 0),
                    'rush_td': row.get('proj_rush_td', 0),
                    'rec_td': row.get('proj_rec_td', 0)
                }
                
                injury_report.append({
                    'player': player_name,
                    'team': row.get('team', ''),
                    'injury_status': status,
                    'impact_factor': impact_factor,
                    'original_pass_yd': key_stats['pass_yd'],
                    'original_rush_yd': key_stats['rush_yd'],
                    'original_rec_yd': key_stats['rec_yd'],
                    'adjusted_pass_yd': key_stats['pass_yd'] * impact_factor,
                    'adjusted_rush_yd': key_stats['rush_yd'] * impact_factor,
                    'adjusted_rec_yd': key_stats['rec_yd'] * impact_factor,
                    'impact_description': self._get_impact_description(status, impact_factor)
                })
        
        if injury_report:
            report_df = pd.DataFrame(injury_report)
            
            # Save report
            report_file = 'data/injury_impact_report.csv'
            report_df.to_csv(report_file, index=False)
            print(f"Injury impact report saved to {report_file}")
            
            # Print summary
            status_counts = report_df['injury_status'].value_counts()
            print(f"Injury status summary:")
            for status, count in status_counts.items():
                print(f"  {status}: {count} players")
            
            return report_df
        else:
            print("No injury data found for current projections")
            return pd.DataFrame()
    
    def _get_impact_description(self, status: str, impact_factor: float) -> str:
        """Get human-readable impact description"""
        if impact_factor == 0.0:
            return "Player removed from projections"
        elif impact_factor < 0.5:
            return f"Major reduction ({int((1-impact_factor)*100)}% decrease)"
        elif impact_factor < 0.8:
            return f"Moderate reduction ({int((1-impact_factor)*100)}% decrease)"
        elif impact_factor < 1.0:
            return f"Minor reduction ({int((1-impact_factor)*100)}% decrease)"
        else:
            return "No impact"
    
    def run_injury_integration(self, projections: pd.DataFrame) -> pd.DataFrame:
        """Run complete injury integration process"""
        print("Starting injury integration...")
        
        # Load injury data
        injuries_df = self.load_injury_data()
        if injuries_df.empty:
            print("No injury data available, returning original projections")
            return projections
        
        # Create injury mapping
        injury_mapping = self.create_injury_mapping(injuries_df)
        if not injury_mapping:
            print("No injury mapping created, returning original projections")
            return projections
        
        # Apply injury adjustments
        enhanced_projections = self.apply_injury_adjustments(projections, injury_mapping)
        
        # Generate injury report
        self.generate_injury_report(enhanced_projections, injury_mapping)
        
        print("Injury integration complete")
        return enhanced_projections
