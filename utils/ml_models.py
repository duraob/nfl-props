"""
Machine Learning Models for NFL Projection Engine

This module implements advanced machine learning models to enhance projection accuracy
by reducing MAE (Mean Absolute Error) and addressing projection over-estimation issues.

Models implemented:
- Position-specific regression models
- Opponent-adjusted regression
- Time-weighted features
- Random Forest regression
- Gradient Boosting
- Neural Network regression
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class MLProjectionModels:
    """
    Machine learning models for NFL projections with position-specific optimization.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        
    def create_position_models(self, df_players, df_teams, df_game_data):
        """
        Create position-specific regression models for QB/RB/WR/TE.
        
        Args:
            df_players: Player dataset with historical stats
            df_teams: Team dataset with team stats
            df_game_data: Historical game data
            
        Returns:
            dict: Position-specific models
        """
        print("Creating position-specific regression models...")
        
        positions = ['QB', 'RB', 'WR', 'TE']
        position_models = {}
        
        for position in positions:
            print(f"  Training {position} model...")
            
            # Filter players by position
            position_players = df_players[df_players.get('position', '') == position].copy()
            
            if len(position_players) < 10:  # Need minimum samples
                print(f"    Insufficient {position} players ({len(position_players)}), skipping...")
                continue
            
            # Create features for this position
            features = self._create_position_features(position_players, df_teams, df_game_data)
            
            if features is None or len(features) < 5:
                print(f"    Insufficient features for {position}, skipping...")
                continue
            
            # Train model for each stat category
            position_model = {}
            stat_categories = ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td']
            
            for stat in stat_categories:
                if stat in features.columns:
                    try:
                        # Prepare data
                        X = features.drop(columns=[stat, 'player', 'team']).select_dtypes(include=[np.number])
                        y = features[stat]
                        
                        # Remove rows with missing target values
                        valid_mask = ~y.isna()
                        X = X[valid_mask]
                        y = y[valid_mask]
                        
                        if len(X) < 5:
                            continue
                        
                        # Train model
                        model = Ridge(alpha=1.0)
                        model.fit(X, y)
                        
                        # Store model and feature names
                        position_model[stat] = {
                            'model': model,
                            'features': X.columns.tolist(),
                            'r2_score': model.score(X, y)
                        }
                        
                    except Exception as e:
                        print(f"    Error training {position} {stat} model: {e}")
                        continue
            
            if position_model:
                position_models[position] = position_model
                print(f"    {position} model created with {len(position_model)} stat categories")
        
        self.models['position_specific'] = position_models
        return position_models
    
    def train_opponent_adjusted_model(self, df_players, df_teams, df_game_data):
        """
        Train opponent-adjusted regression model with defensive strength features.
        
        Args:
            df_players: Player dataset with historical stats
            df_teams: Team dataset with team stats
            df_game_data: Historical game data
            
        Returns:
            dict: Opponent-adjusted models
        """
        print("Training opponent-adjusted regression models...")
        
        # Calculate defensive strength metrics
        defensive_strength = self._calculate_defensive_strength(df_game_data)
        
        # Create opponent-adjusted features
        features = self._create_opponent_adjusted_features(df_players, df_teams, defensive_strength)
        
        if features is None or len(features) < 10:
            print("Insufficient data for opponent-adjusted models")
            return {}
        
        opponent_models = {}
        stat_categories = ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td']
        
        for stat in stat_categories:
            if stat in features.columns:
                try:
                    # Prepare data
                    X = features.drop(columns=[stat, 'player', 'team']).select_dtypes(include=[np.number])
                    y = features[stat]
                    
                    # Remove rows with missing target values
                    valid_mask = ~y.isna()
                    X = X[valid_mask]
                    y = y[valid_mask]
                    
                    if len(X) < 10:
                        continue
                    
                    # Train model
                    model = Ridge(alpha=1.0)
                    model.fit(X, y)
                    
                    # Store model
                    opponent_models[stat] = {
                        'model': model,
                        'features': X.columns.tolist(),
                        'r2_score': model.score(X, y)
                    }
                    
                except Exception as e:
                    print(f"Error training opponent-adjusted {stat} model: {e}")
                    continue
        
        self.models['opponent_adjusted'] = opponent_models
        return opponent_models
    
    def create_time_weighted_features(self, df_players, df_game_data):
        """
        Create time-weighted features for recent performance momentum.
        
        Args:
            df_players: Player dataset with historical stats
            df_game_data: Historical game data
            
        Returns:
            DataFrame: Enhanced features with time-weighted metrics
        """
        print("Creating time-weighted features...")
        
        enhanced_features = df_players.copy()
        
        for idx, player in df_players.iterrows():
            player_name = player.get('name', '')
            team = player.get('team', '')
            
            if not player_name or not team:
                continue
            
            # Get player's recent games
            player_games = df_game_data[
                (df_game_data['player'] == player_name) & 
                (df_game_data['team'] == team)
            ].copy()
            
            if len(player_games) < 2:
                continue
            
            # Sort by week and calculate time-weighted features
            player_games = player_games.sort_values('week')
            
            # Calculate momentum features
            momentum_features = self._calculate_momentum_features(player_games)
            
            # Add to enhanced features
            for feature, value in momentum_features.items():
                enhanced_features.loc[idx, f'tw_{feature}'] = value
        
        print(f"Enhanced {len(enhanced_features)} players with time-weighted features")
        return enhanced_features
    
    def train_random_forest_model(self, df_players, df_teams, df_game_data):
        """
        Train Random Forest regression model for non-linear pattern recognition.
        
        Args:
            df_players: Player dataset with historical stats
            df_teams: Team dataset with team stats
            df_game_data: Historical game data
            
        Returns:
            dict: Random Forest models
        """
        print("Training Random Forest regression models...")
        
        # Create comprehensive features
        features = self._create_comprehensive_features(df_players, df_teams, df_game_data)
        
        if features is None or len(features) < 20:
            print("Insufficient data for Random Forest models")
            return {}
        
        rf_models = {}
        stat_categories = ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td']
        
        for stat in stat_categories:
            if stat in features.columns:
                try:
                    # Prepare data
                    X = features.drop(columns=[stat, 'player', 'team']).select_dtypes(include=[np.number])
                    y = features[stat]
                    
                    # Remove rows with missing target values
                    valid_mask = ~y.isna()
                    X = X[valid_mask]
                    y = y[valid_mask]
                    
                    if len(X) < 20:
                        continue
                    
                    # Train Random Forest
                    rf_model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        random_state=42
                    )
                    rf_model.fit(X, y)
                    
                    # Store model and feature importance
                    rf_models[stat] = {
                        'model': rf_model,
                        'features': X.columns.tolist(),
                        'r2_score': rf_model.score(X, y),
                        'feature_importance': dict(zip(X.columns, rf_model.feature_importances_))
                    }
                    
                except Exception as e:
                    print(f"Error training Random Forest {stat} model: {e}")
                    continue
        
        self.models['random_forest'] = rf_models
        return rf_models
    
    def train_gradient_boosting_model(self, df_players, df_teams, df_game_data):
        """
        Train Gradient Boosting model for error correction.
        
        Args:
            df_players: Player dataset with historical stats
            df_teams: Team dataset with team stats
            df_game_data: Historical game data
            
        Returns:
            dict: Gradient Boosting models
        """
        print("Training Gradient Boosting models...")
        
        # Create comprehensive features
        features = self._create_comprehensive_features(df_players, df_teams, df_game_data)
        
        if features is None or len(features) < 20:
            print("Insufficient data for Gradient Boosting models")
            return {}
        
        gb_models = {}
        stat_categories = ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td']
        
        for stat in stat_categories:
            if stat in features.columns:
                try:
                    # Prepare data
                    X = features.drop(columns=[stat, 'player', 'team']).select_dtypes(include=[np.number])
                    y = features[stat]
                    
                    # Remove rows with missing target values
                    valid_mask = ~y.isna()
                    X = X[valid_mask]
                    y = y[valid_mask]
                    
                    if len(X) < 20:
                        continue
                    
                    # Train Gradient Boosting
                    gb_model = GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42
                    )
                    gb_model.fit(X, y)
                    
                    # Store model and feature importance
                    gb_models[stat] = {
                        'model': gb_model,
                        'features': X.columns.tolist(),
                        'r2_score': gb_model.score(X, y),
                        'feature_importance': dict(zip(X.columns, gb_model.feature_importances_))
                    }
                    
                except Exception as e:
                    print(f"Error training Gradient Boosting {stat} model: {e}")
                    continue
        
        self.models['gradient_boosting'] = gb_models
        return gb_models
    
    def train_neural_network_model(self, df_players, df_teams, df_game_data):
        """
        Train Neural Network model for advanced pattern recognition.
        
        Args:
            df_players: Player dataset with historical stats
            df_teams: Team dataset with team stats
            df_game_data: Historical game data
            
        Returns:
            dict: Neural Network models
        """
        print("Training Neural Network models...")
        
        # Create comprehensive features
        features = self._create_comprehensive_features(df_players, df_teams, df_game_data)
        
        if features is None or len(features) < 20:
            print("Insufficient data for Neural Network models")
            return {}
        
        nn_models = {}
        stat_categories = ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td']
        
        for stat in stat_categories:
            if stat in features.columns:
                try:
                    # Prepare data
                    X = features.drop(columns=[stat, 'player', 'team']).select_dtypes(include=[np.number])
                    y = features[stat]
                    
                    # Remove rows with missing target values
                    valid_mask = ~y.isna()
                    X = X[valid_mask]
                    y = y[valid_mask]
                    
                    if len(X) < 20:
                        continue
                    
                    # Scale features for neural network
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Train Neural Network
                    nn_model = MLPRegressor(
                        hidden_layer_sizes=(100, 50),
                        activation='relu',
                        solver='adam',
                        max_iter=500,
                        random_state=42
                    )
                    nn_model.fit(X_scaled, y)
                    
                    # Store model and scaler
                    nn_models[stat] = {
                        'model': nn_model,
                        'scaler': scaler,
                        'features': X.columns.tolist(),
                        'r2_score': nn_model.score(X_scaled, y)
                    }
                    
                except Exception as e:
                    print(f"Error training Neural Network {stat} model: {e}")
                    continue
        
        self.models['neural_network'] = nn_models
        return nn_models
    
    def _create_position_features(self, df_players, df_teams, df_game_data):
        """Create features for position-specific models."""
        try:
            features = df_players.copy()
            
            # Add basic statistical features
            stat_columns = ['pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td']
            for stat in stat_columns:
                if stat in features.columns:
                    # Add rolling averages
                    features[f'{stat}_avg_3'] = features[stat].rolling(window=3, min_periods=1).mean()
                    features[f'{stat}_avg_5'] = features[stat].rolling(window=5, min_periods=1).mean()
                    
                    # Add standard deviation
                    features[f'{stat}_std'] = features[stat].rolling(window=5, min_periods=1).std()
            
            return features
            
        except Exception as e:
            print(f"Error creating position features: {e}")
            return None
    
    def _create_opponent_adjusted_features(self, df_players, df_teams, defensive_strength):
        """Create opponent-adjusted features."""
        try:
            features = df_players.copy()
            
            # Add defensive strength features
            for team in defensive_strength.keys():
                team_mask = features['team'] == team
                if team_mask.any():
                    features.loc[team_mask, 'opp_def_strength'] = defensive_strength[team]
            
            return features
            
        except Exception as e:
            print(f"Error creating opponent-adjusted features: {e}")
            return None
    
    def _create_comprehensive_features(self, df_players, df_teams, df_game_data):
        """Create comprehensive features for advanced models."""
        try:
            features = df_players.copy()
            
            # Add time-weighted features
            features = self.create_time_weighted_features(features, df_game_data)
            
            # Add position features
            features = self._create_position_features(features, df_teams, df_game_data)
            
            return features
            
        except Exception as e:
            print(f"Error creating comprehensive features: {e}")
            return None
    
    def _calculate_defensive_strength(self, df_game_data):
        """Calculate defensive strength metrics for each team."""
        try:
            defensive_strength = {}
            
            for team in df_game_data['team'].unique():
                team_data = df_game_data[df_game_data['team'] == team]
                
                # Calculate average points allowed
                avg_points_allowed = team_data.groupby('opponent')['opp_score'].mean().mean()
                
                # Calculate average yards allowed
                avg_yards_allowed = team_data.groupby('opponent')['opp_pass_yd'].mean().mean()
                
                # Store defensive strength (lower is better)
                defensive_strength[team] = (avg_points_allowed + avg_yards_allowed) / 2
            
            return defensive_strength
            
        except Exception as e:
            print(f"Error calculating defensive strength: {e}")
            return {}
    
    def _calculate_momentum_features(self, player_games):
        """Calculate momentum features for a player."""
        try:
            momentum_features = {}
            
            if len(player_games) < 2:
                return momentum_features
            
            # Calculate recent vs older performance
            recent_games = player_games.tail(3)
            older_games = player_games.head(-3)
            
            if len(recent_games) > 0 and len(older_games) > 0:
                for stat in ['pass_yd', 'rush_yd', 'rec_yd']:
                    if stat in recent_games.columns:
                        recent_avg = recent_games[stat].mean()
                        older_avg = older_games[stat].mean()
                        momentum_features[f'{stat}_momentum'] = recent_avg - older_avg
            
            return momentum_features
            
        except Exception as e:
            print(f"Error calculating momentum features: {e}")
            return {}
    
    def predict_with_models(self, player_data, position, stat_category):
        """
        Make predictions using trained models.
        
        Args:
            player_data: Player data for prediction
            position: Player position
            stat_category: Stat category to predict
            
        Returns:
            dict: Predictions from different models
        """
        predictions = {}
        
        # Try position-specific model
        if 'position_specific' in self.models and position in self.models['position_specific']:
            if stat_category in self.models['position_specific'][position]:
                try:
                    model_info = self.models['position_specific'][position][stat_category]
                    model = model_info['model']
                    features = model_info['features']
                    
                    # Prepare features
                    X = player_data[features].values.reshape(1, -1)
                    pred = model.predict(X)[0]
                    predictions['position_specific'] = pred
                except:
                    pass
        
        # Try Random Forest model
        if 'random_forest' in self.models and stat_category in self.models['random_forest']:
            try:
                model_info = self.models['random_forest'][stat_category]
                model = model_info['model']
                features = model_info['features']
                
                # Prepare features
                X = player_data[features].values.reshape(1, -1)
                pred = model.predict(X)[0]
                predictions['random_forest'] = pred
            except:
                pass
        
        # Try Gradient Boosting model
        if 'gradient_boosting' in self.models and stat_category in self.models['gradient_boosting']:
            try:
                model_info = self.models['gradient_boosting'][stat_category]
                model = model_info['model']
                features = model_info['features']
                
                # Prepare features
                X = player_data[features].values.reshape(1, -1)
                pred = model.predict(X)[0]
                predictions['gradient_boosting'] = pred
            except:
                pass
        
        return predictions


def create_ml_models(df_players, df_teams, df_game_data):
    """
    Create and train all ML models for the projection engine.
    
    Args:
        df_players: Player dataset with historical stats
        df_teams: Team dataset with team stats
        df_game_data: Historical game data
        
    Returns:
        MLProjectionModels: Trained ML models
    """
    print("Creating ML models for NFL projection engine...")
    
    ml_models = MLProjectionModels()
    
    # Create position-specific models
    ml_models.create_position_models(df_players, df_teams, df_game_data)
    
    # Create opponent-adjusted models
    ml_models.train_opponent_adjusted_model(df_players, df_teams, df_game_data)
    
    # Create time-weighted features
    enhanced_features = ml_models.create_time_weighted_features(df_players, df_game_data)
    
    # Create advanced models
    ml_models.train_random_forest_model(df_players, df_teams, df_game_data)
    ml_models.train_gradient_boosting_model(df_players, df_teams, df_game_data)
    ml_models.train_neural_network_model(df_players, df_teams, df_game_data)
    
    print("ML models creation complete!")
    return ml_models


if __name__ == "__main__":
    # Test the ML models
    print("Testing ML models...")
    
    # Load sample data
    try:
        df_players = pd.read_csv('data/game_data/game_data_2023.csv')
        df_teams = pd.DataFrame()  # Placeholder
        df_game_data = df_players.copy()
        
        # Create ML models
        ml_models = create_ml_models(df_players, df_teams, df_game_data)
        
        print("ML models test completed successfully!")
        
    except Exception as e:
        print(f"Error testing ML models: {e}")
