from flask import Flask, request, jsonify
from predictor_elite import ElitePredictor
import json
import os
from datetime import datetime
import logging
from flask_cors import CORS


# ==================== APP INITIALIZATION ====================

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load predictor
try:
    predictor = ElitePredictor(
        model_path='models/xgboost_elite_model.pkl',
        scaler_path='models/scaler.pkl',
        features_path='models/feature_columns.json',
        metrics_path='models/metrics.json'
    )
    logger.info("✓ ElitePredictor loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to load ElitePredictor: {e}")
    predictor = None


# ==================== STARTUP INFO ====================

print("""
╔══════════════════════════════════════════════════════════════════╗
║           🏀 NBA ELITE PREDICTION API - v2.1                    ║
║           Production-Grade XGBoost Model                         ║
║           Accuracy: 74.73% | ROC-AUC: 0.8261                   ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ==================== HEALTH & STATUS ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint - verify service is running
    
    Returns:
        JSON with service status and timestamp
    """
    try:
        status = 'healthy' if predictor is not None else 'degraded'
        
        return jsonify({
            'status': status,
            'service': 'nba-elite-prediction-api',
            'version': '2.1',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': predictor is not None
        }), 200
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/status', methods=['GET'])
def status():
    """Detailed status endpoint with model information
    
    Returns:
        JSON with detailed service and model status
    """
    try:
        if predictor is None:
            return jsonify({
                'status': 'offline',
                'message': 'Model not loaded'
            }), 503
        
        return jsonify({
            'status': 'online',
            'service': 'nba-elite-prediction-api',
            'version': '2.1',
            'model': {
                'type': 'XGBoost Classifier',
                'features': len(predictor.feature_columns),
                'training_accuracy': predictor.metrics.get('accuracy'),
                'roc_auc': predictor.metrics.get('roc_auc'),
                'sensitivity': predictor.metrics.get('sensitivity'),
                'specificity': predictor.metrics.get('specificity'),
                'best_iteration': predictor.metrics.get('best_iteration'),
                'early_stopping_rounds': predictor.metrics.get('early_stopping_rounds')
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


# ==================== PREDICTION ENDPOINTS ====================

@app.route('/predict', methods=['POST'])
def predict():
    """Predict single game outcome using elite ranking features
    
    POST /predict
    Content-Type: application/json
    
    Request Body:
    {
        "home_team": "Lakers",
        "away_team": "Celtics",
        "ranking_data": {
            "OFF_RNK_DIFF": 5,              # Home off rank - Away off rank
            "DEF_RNK_DIFF": -3,             # Home def rank - Away def rank
            "PTS_AVG_DIFF": 2.5,            # Home PTS avg - Away PTS avg (5-game)
            "DEF_AVG_DIFF": -1.2,           # Home def avg - Away def avg (5-game)
            "HOME_OFF_RANK": 8,             # Home offensive rank (1-30)
            "HOME_DEF_RANK": 12,            # Home defensive rank (1-30)
            "AWAY_OFF_RANK": 3,             # Away offensive rank (1-30)
            "AWAY_DEF_RANK": 15,            # Away defensive rank (1-30)
            "HOME_RUNNING_OFF_RANK": 7,    # Home current running off rank
            "HOME_RUNNING_DEF_RANK": 11,   # Home current running def rank
            "OFF_MOMENTUM": -1,             # Change in offensive rank
            "DEF_MOMENTUM": -1,             # Change in defensive rank
            "RANK_INTERACTION": -15,       # OFF_RNK_DIFF * DEF_RNK_DIFF
            "PTS_RANK_INTERACTION": 12.5,  # PTS_AVG_DIFF * OFF_RNK_DIFF
            "HOME_COURT": 1,                # Always 1 (home advantage)
            "GAME_NUMBER": 10               # Cumulative game number for home team
        }
    }
    
    Response:
    {
        "success": true,
        "home_team": "Lakers",
        "away_team": "Celtics",
        "prediction": {
            "success": true,
            "home_win_probability": 0.7245,
            "away_win_probability": 0.2755,
            "predicted_winner": "HOME",
            "confidence": "72.45%",
            "top_impacting_features": {...},
            "model_performance": {...}
        }
    }
    """
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Predictor not loaded'
            }), 503
        
        # Parse request JSON
        data = request.json
        
        if data is None:
            return jsonify({
                'success': False,
                'error': 'Request body must be valid JSON'
            }), 400
        
        # Validate ranking_data
        if 'ranking_data' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: ranking_data',
                'required_fields': ['ranking_data'],
                'optional_fields': ['home_team', 'away_team']
            }), 400
        
        ranking_data = data['ranking_data']
        
        # Validate feature count
        if not isinstance(ranking_data, dict):
            return jsonify({
                'success': False,
                'error': 'ranking_data must be a dictionary'
            }), 400
        
        # Get prediction
        prediction = predictor.predict_game(ranking_data)
        
        if not prediction.get('success'):
            return jsonify({
                'success': False,
                'error': prediction.get('error', 'Prediction failed'),
                'details': prediction
            }), 400
        
        # Return successful prediction
        response = {
            'success': True,
            'home_team': data.get('home_team', 'HOME'),
            'away_team': data.get('away_team', 'AWAY'),
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction successful: {data.get('home_team', 'HOME')} vs {data.get('away_team', 'AWAY')}")
        return jsonify(response), 200
    
    except json.JSONDecodeError:
        logger.error("Invalid JSON in request")
        return jsonify({
            'success': False,
            'error': 'Invalid JSON in request body'
        }), 400
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Predict multiple games at once
    
    POST /predict-batch
    Content-Type: application/json
    
    Request Body:
    {
        "games": [
            {
                "home_team": "Lakers",
                "away_team": "Celtics",
                "ranking_data": {...}
            },
            {
                "home_team": "Warriors",
                "away_team": "Suns",
                "ranking_data": {...}
            }
        ]
    }
    
    Response:
    {
        "success": true,
        "total_games": 2,
        "successful_predictions": 2,
        "failed_predictions": 0,
        "predictions": [...]
    }
    """
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Predictor not loaded'
            }), 503
        
        data = request.json
        
        if data is None or 'games' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: games'
            }), 400
        
        games = data['games']
        
        if not isinstance(games, list):
            return jsonify({
                'success': False,
                'error': 'games must be a list'
            }), 400
        
        if len(games) == 0:
            return jsonify({
                'success': False,
                'error': 'games list cannot be empty'
            }), 400
        
        if len(games) > 100:
            return jsonify({
                'success': False,
                'error': 'Maximum 100 games per batch'
            }), 400
        
        # Predict all games
        predictions = []
        successful = 0
        failed = 0
        
        for game in games:
            if 'ranking_data' not in game:
                failed += 1
                predictions.append({
                    'success': False,
                    'error': 'Missing ranking_data',
                    'home_team': game.get('home_team', 'UNKNOWN'),
                    'away_team': game.get('away_team', 'UNKNOWN')
                })
                continue
            
            prediction = predictor.predict_game(game['ranking_data'])
            
            if prediction.get('success'):
                successful += 1
            else:
                failed += 1
            
            predictions.append({
                'home_team': game.get('home_team', 'HOME'),
                'away_team': game.get('away_team', 'AWAY'),
                'prediction': prediction
            })
        
        logger.info(f"Batch prediction: {successful}/{len(games)} successful")
        
        return jsonify({
            'success': True,
            'total_games': len(games),
            'successful_predictions': successful,
            'failed_predictions': failed,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== METADATA ENDPOINTS ====================

@app.route('/metrics', methods=['GET'])
def metrics():
    """Return model performance metrics
    
    Returns:
        JSON with training metrics and performance stats
    """
    try:
        if predictor is None:
            return jsonify({
                'error': 'Predictor not loaded'
            }), 503
        
        return jsonify({
            'success': True,
            'metrics': predictor.metrics,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/features', methods=['GET'])
def features():
    """Return list of required features for predictions
    
    Returns:
        JSON with feature count and list of features
    """
    try:
        if predictor is None:
            return jsonify({
                'error': 'Predictor not loaded'
            }), 503
        
        features_list = predictor.feature_columns
        
        return jsonify({
            'success': True,
            'feature_count': len(features_list),
            'features': features_list,
            'feature_descriptions': {
                'OFF_RNK_DIFF': 'Home offensive rank - Away offensive rank',
                'DEF_RNK_DIFF': 'Home defensive rank - Away defensive rank',
                'PTS_AVG_DIFF': 'Home 5-game PTS average - Away 5-game PTS average',
                'DEF_AVG_DIFF': 'Home 5-game PTS allowed average - Away 5-game PTS allowed average',
                'HOME_OFF_RANK': 'Home team seasonal offensive ranking (1-30)',
                'HOME_DEF_RANK': 'Home team seasonal defensive ranking (1-30)',
                'AWAY_OFF_RANK': 'Away team seasonal offensive ranking (1-30)',
                'AWAY_DEF_RANK': 'Away team seasonal defensive ranking (1-30)',
                'HOME_RUNNING_OFF_RANK': 'Home team current running offensive rank',
                'HOME_RUNNING_DEF_RANK': 'Home team current running defensive rank',
                'OFF_MOMENTUM': 'Change in offensive ranking (recent momentum)',
                'DEF_MOMENTUM': 'Change in defensive ranking (recent momentum)',
                'RANK_INTERACTION': 'OFF_RNK_DIFF * DEF_RNK_DIFF (multiplicative interaction)',
                'PTS_RANK_INTERACTION': 'PTS_AVG_DIFF * OFF_RNK_DIFF (scoring-rank interaction)',
                'HOME_COURT': 'Home court advantage indicator (always 1)',
                'GAME_NUMBER': 'Cumulative game number for home team'
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Features endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/info', methods=['GET'])
def info():
    """Get API information and documentation
    
    Returns:
        JSON with API details and endpoint documentation
    """
    return jsonify({
        'service': 'NBA Elite Prediction API',
        'version': '2.1',
        'description': 'Advanced XGBoost-based prediction service for NBA games',
        'model': {
            'type': 'XGBoost Classifier',
            'training_accuracy': 0.7473,
            'roc_auc': 0.8261,
            'sensitivity': 0.8050,
            'specificity': 0.6750,
            'features': 16
        },
        'endpoints': {
            'GET /health': 'Quick health check',
            'GET /status': 'Detailed service status',
            'GET /info': 'API information and documentation',
            'POST /predict': 'Predict single game outcome',
            'POST /predict-batch': 'Predict multiple games',
            'GET /metrics': 'Model performance metrics',
            'GET /features': 'Required features list'
        },
        'timestamp': datetime.now().isoformat()
    }), 200


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'path': request.path,
        'method': request.method
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'path': request.path,
        'method': request.method
    }), 405


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    logger.error(f"Server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    # Configuration from environment
    port = int(os.getenv('FLASK_PORT', 5001))
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    
    # Print startup info
    print(f"\n✓ Server Configuration:")
    print(f"  • Host: {host}")
    print(f"  • Port: {port}")
    print(f"  • Debug: {debug}")
    print(f"  • Environment: {'Development' if debug else 'Production'}")
    
    print(f"\n✓ Available Endpoints:")
    print(f"  ├─ GET  /health              Quick health check")
    print(f"  ├─ GET  /status              Detailed status")
    print(f"  ├─ GET  /info                API documentation")
    print(f"  ├─ POST /predict             Single game prediction")
    print(f"  ├─ POST /predict-batch       Batch predictions")
    print(f"  ├─ GET  /metrics             Model performance")
    print(f"  └─ GET  /features            Feature list")
    
    print(f"\n✓ Model Status:")
    print(f"  • Loaded: {predictor is not None}")
    if predictor is not None:
        print(f"  • Accuracy: 74.73%")
        print(f"  • ROC-AUC: 0.8261")
        print(f"  • Features: {len(predictor.feature_columns)}")
    
    print(f"\n" + "="*70)
    print(f"🚀 Starting NBA Elite Prediction API...")
    print(f"="*70 + "\n")
    
    # Run Flask app
    app.run(debug=debug, port=port, host=host, threaded=True)
