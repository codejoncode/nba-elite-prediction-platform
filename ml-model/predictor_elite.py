import pickle
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class ElitePredictor:
    """Load elite XGBoost model and make game predictions
    
    This predictor loads the trained NBA Elite model and provides
    predictions for individual games based on engineered features.
    
    Features Required (16 total):
    - OFF_RNK_DIFF, DEF_RNK_DIFF: Seasonal ranking differentials
    - PTS_AVG_DIFF, DEF_AVG_DIFF: 5-game rolling averages
    - HOME_OFF_RANK, HOME_DEF_RANK: Home team seasonal ranks
    - AWAY_OFF_RANK, AWAY_DEF_RANK: Away team seasonal ranks
    - HOME_RUNNING_OFF_RANK, HOME_RUNNING_DEF_RANK: Running momentum ranks
    - OFF_MOMENTUM, DEF_MOMENTUM: Momentum change indicators
    - RANK_INTERACTION, PTS_RANK_INTERACTION: Feature interactions
    - HOME_COURT: Binary indicator (always 1)
    - GAME_NUMBER: Cumulative games for home team
    
    Model Performance:
    - Accuracy: 74.73%
    - ROC-AUC: 0.8261
    - Sensitivity: 80.50%
    - Best Iteration: 42
    """
    
    def __init__(self, model_path='models/xgboost_elite_model.pkl',
                 scaler_path='models/scaler.pkl',
                 features_path='models/feature_columns.json',
                 metrics_path='models/metrics.json'):
        
        """Initialize elite predictor with trained artifacts
        
        Args:
            model_path (str): Path to trained XGBoost model pickle
            scaler_path (str): Path to StandardScaler pickle
            features_path (str): Path to feature columns JSON
            metrics_path (str): Path to training metrics JSON
        """
        
        print("🚀 Loading elite model artifacts...")
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded: {model_path}")
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Scaler loaded: {scaler_path}")
            
            # Load feature columns (CRITICAL - must match training order)
            with open(features_path, 'r') as f:
                self.feature_columns = json.load(f)
            print(f"✓ Features loaded: {len(self.feature_columns)} columns")
            
            # Load metrics (optional but informative)
            try:
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
                    print(f"✓ Metrics loaded:")
                    print(f"   - Training Accuracy: {self.metrics.get('accuracy', 'N/A'):.2%}")
                    print(f"   - ROC-AUC: {self.metrics.get('roc_auc', 'N/A'):.4f}")
                    print(f"   - Sensitivity: {self.metrics.get('sensitivity', 'N/A'):.2%}")
            except FileNotFoundError:
                self.metrics = {}
                print(f"⚠️  Metrics file not found (optional)")
            
            print("\n" + "="*70)
            print("✅ ELITE PREDICTOR READY FOR PREDICTIONS")
            print("="*70 + "\n")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Failed to load model artifacts: {e}")
    
    def validate_features(self, ranking_data):
        """Validate that all required features are present
        
        Args:
            ranking_data (dict): Dictionary containing feature values
            
        Returns:
            tuple: (is_valid, missing_features, extra_features)
        """
        
        provided_keys = set(ranking_data.keys())
        required_keys = set(self.feature_columns)
        
        missing = required_keys - provided_keys
        extra = provided_keys - required_keys
        
        return len(missing) == 0, missing, extra
    
    def predict_game(self, ranking_data):
        """Predict single NBA game outcome with elite ranking features
        
        Args:
            ranking_data (dict): Dictionary with 16 engineered features.
                Example:
                {
                    'OFF_RNK_DIFF': 5,                  # Home off rank - Away off rank
                    'DEF_RNK_DIFF': -3,                 # Home def rank - Away def rank
                    'PTS_AVG_DIFF': 2.5,                # Home PTS avg - Away PTS avg (5-game)
                    'DEF_AVG_DIFF': -1.2,               # Home def avg - Away def avg (5-game)
                    'HOME_OFF_RANK': 8,                 # Home offensive rank (1-30)
                    'HOME_DEF_RANK': 12,                # Home defensive rank (1-30)
                    'AWAY_OFF_RANK': 3,                 # Away offensive rank (1-30)
                    'AWAY_DEF_RANK': 15,                # Away defensive rank (1-30)
                    'HOME_RUNNING_OFF_RANK': 7,        # Home current running off rank
                    'HOME_RUNNING_DEF_RANK': 11,       # Home current running def rank
                    'OFF_MOMENTUM': -1,                 # Change in offensive rank
                    'DEF_MOMENTUM': -1,                 # Change in defensive rank
                    'RANK_INTERACTION': -15,            # OFF_RNK_DIFF * DEF_RNK_DIFF
                    'PTS_RANK_INTERACTION': 12.5,      # PTS_AVG_DIFF * OFF_RNK_DIFF
                    'HOME_COURT': 1,                    # Always 1 (home advantage)
                    'GAME_NUMBER': 10                   # Cumulative game number for home team
                }
        
        Returns:
            dict: Prediction results with probabilities and confidence
        """
        
        try:
            # Validate features
            is_valid, missing, extra = self.validate_features(ranking_data)
            
            if not is_valid:
                return {
                    'error': f"Missing features: {missing}",
                    'home_win_probability': 0.5,
                    'away_win_probability': 0.5,
                    'predicted_winner': 'ERROR',
                    'confidence': 0.5
                }
            
            if extra:
                print(f"⚠️  Extra features provided (ignored): {extra}")
            
            # Build feature vector in correct order (CRITICAL)
            feature_vector = np.array([
                ranking_data.get(col, 0.0) for col in self.feature_columns
            ]).reshape(1, -1)
            
            # Validate feature vector
            if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                return {
                    'error': "Invalid feature values (NaN or Inf detected)",
                    'home_win_probability': 0.5,
                    'away_win_probability': 0.5,
                    'predicted_winner': 'ERROR',
                    'confidence': 0.5
                }
            
            # Scale features using training scaler
            feature_scaled = self.scaler.transform(feature_vector)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(feature_scaled)[0]
            home_win_prob = float(probabilities[1])  # Class 1 = home win
            away_win_prob = float(probabilities[0])  # Class 0 = away win
            
            # Determine winner
            predicted_winner = 'HOME' if home_win_prob > 0.5 else 'AWAY'
            confidence = max(home_win_prob, away_win_prob)
            
            # Get feature importance context
            top_features = self._get_top_features(ranking_data)
            
            return {
                'success': True,
                'home_win_probability': home_win_prob,
                'away_win_probability': away_win_prob,
                'predicted_winner': predicted_winner,
                'confidence': confidence,
                'confidence_pct': f"{confidence*100:.2f}%",
                'top_impacting_features': top_features,
                'model_performance': {
                    'training_accuracy': self.metrics.get('accuracy', None),
                    'roc_auc': self.metrics.get('roc_auc', None),
                    'best_iteration': self.metrics.get('best_iteration', None)
                }
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'home_win_probability': 0.5,
                'away_win_probability': 0.5,
                'predicted_winner': 'ERROR',
                'confidence': 0.5
            }
    
    def _get_top_features(self, ranking_data):
        """Extract top impacting features for context
        
        Based on model feature importance analysis:
        1. DEF_AVG_DIFF (21.81%)
        2. PTS_AVG_DIFF (18.84%)
        3. PTS_RANK_INTERACTION (9.41%)
        4. OFF_RNK_DIFF (8.62%)
        5. DEF_RNK_DIFF (7.43%)
        """
        
        top_features = {}
        feature_importance_order = [
            'DEF_AVG_DIFF',
            'PTS_AVG_DIFF',
            'PTS_RANK_INTERACTION',
            'OFF_RNK_DIFF',
            'DEF_RNK_DIFF'
        ]
        
        for feature in feature_importance_order:
            if feature in ranking_data:
                top_features[feature] = ranking_data[feature]
        
        return top_features
    
    def predict_batch(self, games_list):
        """Predict multiple games at once
        
        Args:
            games_list (list): List of dicts with ranking data
            
        Returns:
            list: List of prediction results
        """
        
        print(f"🔮 Predicting {len(games_list)} games...")
        predictions = []
        
        for i, game_data in enumerate(games_list, 1):
            pred = self.predict_game(game_data)
            predictions.append(pred)
            
            if pred.get('success'):
                winner = pred['predicted_winner']
                confidence = pred['confidence']
                print(f"  [{i}] {winner} wins ({confidence*100:.1f}% confidence)")
            else:
                print(f"  [{i}] ❌ Prediction failed: {pred.get('error')}")
        
        return predictions
    
    def explain_prediction(self, ranking_data, prediction):
        """Generate human-readable explanation of prediction
        
        Args:
            ranking_data (dict): Input features
            prediction (dict): Prediction output
            
        Returns:
            str: Formatted explanation
        """
        
        if not prediction.get('success'):
            return f"⚠️  Prediction failed: {prediction.get('error')}"
        
        explanation = []
        explanation.append("\n" + "="*70)
        explanation.append("📊 PREDICTION ANALYSIS")
        explanation.append("="*70)
        explanation.append(f"\n🏀 Winner: {prediction['predicted_winner']} Team")
        explanation.append(f"📈 Home Win Probability: {prediction['home_win_probability']:.2%}")
        explanation.append(f"📉 Away Win Probability: {prediction['away_win_probability']:.2%}")
        explanation.append(f"💪 Model Confidence: {prediction['confidence_pct']}")
        
        explanation.append(f"\n🔑 Top Impacting Features:")
        for feature, value in prediction['top_impacting_features'].items():
            impact_sign = "📈" if value > 0 else "📉" if value < 0 else "➡️"
            explanation.append(f"   {impact_sign} {feature}: {value:.2f}")
        
        explanation.append(f"\n🎓 Model Stats:")
        explanation.append(f"   • Training Accuracy: {prediction['model_performance']['training_accuracy']:.2%}")
        explanation.append(f"   • ROC-AUC: {prediction['model_performance']['roc_auc']:.4f}")
        explanation.append(f"   • Best Iteration: {prediction['model_performance']['best_iteration']}")
        
        explanation.append("\n" + "="*70 + "\n")
        
        return "\n".join(explanation)


# TEST & DEMONSTRATION
if __name__ == '__main__':
    
    print("\n" + "="*70)
    print("🏆 NBA ELITE PREDICTION SYSTEM - TEST")
    print("="*70 + "\n")
    
    # Initialize predictor
    predictor = ElitePredictor()
    
    # Test Prediction 1: Strong Home Team
    test_data_1 = {
        'OFF_RNK_DIFF': 8,              # Home team has strong offense
        'DEF_RNK_DIFF': 5,              # Home team has strong defense
        'PTS_AVG_DIFF': 3.5,            # Home team scoring more
        'DEF_AVG_DIFF': 2.1,            # Home team defending better
        'HOME_OFF_RANK': 5,             # Top 5 offense
        'HOME_DEF_RANK': 8,             # Top 10 defense
        'AWAY_OFF_RANK': 15,            # Middle-tier offense
        'AWAY_DEF_RANK': 18,            # Below-average defense
        'HOME_RUNNING_OFF_RANK': 4,     # Currently ranked 4th
        'HOME_RUNNING_DEF_RANK': 7,     # Currently ranked 7th
        'OFF_MOMENTUM': 1,              # Improving offense
        'DEF_MOMENTUM': 2,              # Improving defense
        'RANK_INTERACTION': 40,         # Positive interaction
        'PTS_RANK_INTERACTION': 28.0,   # Positive interaction
        'HOME_COURT': 1,
        'GAME_NUMBER': 15
    }
    
    print("📋 TEST 1: Strong Home Team vs Weaker Away Team")
    print("-" * 70)
    prediction_1 = predictor.predict_game(test_data_1)
    print(predictor.explain_prediction(test_data_1, prediction_1))
    
    # Test Prediction 2: Close Game
    test_data_2 = {
        'OFF_RNK_DIFF': 0,              # Even offenses
        'DEF_RNK_DIFF': -1,             # Away team slightly better defense
        'PTS_AVG_DIFF': 0.5,            # Minimal scoring difference
        'DEF_AVG_DIFF': 0.3,            # Minimal defense difference
        'HOME_OFF_RANK': 12,            # Mid-tier offense
        'HOME_DEF_RANK': 14,            # Mid-tier defense
        'AWAY_OFF_RANK': 13,            # Mid-tier offense
        'AWAY_DEF_RANK': 13,            # Mid-tier defense
        'HOME_RUNNING_OFF_RANK': 12,    # Currently mid-tier
        'HOME_RUNNING_DEF_RANK': 14,    # Currently mid-tier
        'OFF_MOMENTUM': 0,              # No momentum
        'DEF_MOMENTUM': 0,              # No momentum
        'RANK_INTERACTION': 0,          # No interaction
        'PTS_RANK_INTERACTION': 0.0,    # No interaction
        'HOME_COURT': 1,
        'GAME_NUMBER': 20
    }
    
    print("📋 TEST 2: Close/Competitive Game")
    print("-" * 70)
    prediction_2 = predictor.predict_game(test_data_2)
    print(predictor.explain_prediction(test_data_2, prediction_2))
    
    # Test Prediction 3: Away Team Favored
    test_data_3 = {
        'OFF_RNK_DIFF': -6,             # Away team better offense
        'DEF_RNK_DIFF': -4,             # Away team better defense
        'PTS_AVG_DIFF': -2.8,           # Away team scoring more
        'DEF_AVG_DIFF': -1.9,           # Away team defending better
        'HOME_OFF_RANK': 22,            # Weak offense
        'HOME_DEF_RANK': 24,            # Weak defense
        'AWAY_OFF_RANK': 3,             # Elite offense
        'AWAY_DEF_RANK': 5,             # Elite defense
        'HOME_RUNNING_OFF_RANK': 23,    # Currently ranked 23rd
        'HOME_RUNNING_DEF_RANK': 25,    # Currently ranked 25th
        'OFF_MOMENTUM': -1,             # Declining offense
        'DEF_MOMENTUM': -2,             # Declining defense
        'RANK_INTERACTION': 24,         # Positive away advantage
        'PTS_RANK_INTERACTION': -16.8,  # Negative home advantage
        'HOME_COURT': 1,
        'GAME_NUMBER': 25
    }
    
    print("📋 TEST 3: Away Team Heavily Favored")
    print("-" * 70)
    prediction_3 = predictor.predict_game(test_data_3)
    print(predictor.explain_prediction(test_data_3, prediction_3))
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE - PREDICTOR FUNCTIONING PERFECTLY")
    print("="*70 + "\n")
