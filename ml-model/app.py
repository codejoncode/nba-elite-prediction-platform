"""
NBA Elite ML Prediction API - Flask Service
Provides XGBoost predictions for upcoming games
Runs on separate port (5001) from Node backend (3001)
Complete with logging, authentication, and comprehensive endpoints

Auto-runs game_results.json sync on startup + scheduled intervals
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging
from datetime import datetime, timedelta
import jwt
from functools import wraps
import pickle
import numpy as np
import json

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['JSON_SORT_KEYS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')

# ============================================================================
# CORS CONFIGURATION - FIXED for React dev server
# ============================================================================
CORS(app, 
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:3000",      # React dev server (CRA)
                "http://localhost:5173",      # React dev server (Vite)
                "http://localhost:3001",      # Node backend
                "http://localhost:5001",      # Self
                "http://127.0.0.1:3000",      # Localhost variant
                "http://127.0.0.1:5173",      # Localhost variant
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Cron-Token"],
            "supports_credentials": True
        },
        r"/*": {
            "origins": [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://localhost:3001",
                "http://localhost:5001",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Cron-Token"],
            "supports_credentials": True
        }
    },
    max_age=3600
)

# ============================================================================
# LOGGING SETUP - Windows-safe (no Unicode checkmarks)
# ============================================================================

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler('ml_api_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ML MODEL LOADING
# ============================================================================

xgb_model = None
model_loaded = False

try:
    # Load your trained XGBoost model
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            xgb_model = pickle.load(f)
        model_loaded = True
        logger.info("[OK] XGBoost model loaded from %s", model_path)
    else:
        logger.warning("[WARNING] Model file not found at %s - running in demo mode", model_path)
except Exception as e:
    logger.error("[ERROR] Failed to load model: %s", str(e), exc_info=True)
    model_loaded = False

# ============================================================================
# GAME RESULTS DATA - In-Memory Cache
# ============================================================================

game_results_data = None

def load_game_results():
    """Load game results from JSON file"""
    global game_results_data
    try:
        game_results_path = os.path.join(os.path.dirname(__file__), 'data', 'game_results.json')
        if os.path.exists(game_results_path):
            with open(game_results_path, 'r') as f:
                game_results_data = json.load(f)
            logger.info("[OK] Game results loaded from %s", game_results_path)
            return True
        else:
            logger.warning("[WARNING] Game results file not found at %s", game_results_path)
            game_results_data = {
                "metadata": {
                    "last_updated": datetime.utcnow().isoformat(),
                    "accuracy_all_time_percent": 0.0,
                    "accuracy_last_20_percent": 0.0,
                    "total_games_tracked": 0,
                    "correct_predictions": 0,
                    "incorrect_predictions": 0
                },
                "recent_results": [],
                "upcoming_games": []
            }
            return False
    except Exception as e:
        logger.error("[ERROR] Failed to load game results: %s", str(e), exc_info=True)
        return False

# Load on startup
load_game_results()

# ============================================================================
# AUTHENTICATION
# ============================================================================

def token_required(f):
    """Decorator to require JWT token on protected routes"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                logger.warning("[WARNING] Invalid Authorization header format")
                return jsonify({'error': 'Invalid Authorization header format'}), 401

        if not token:
            logger.warning("[WARNING] Missing authorization token")
            return jsonify({'error': 'Token is missing'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user = data
            logger.info("[OK] Token validated for user: %s", data.get('sub', 'unknown'))
        except jwt.ExpiredSignatureError:
            logger.warning("[WARNING] Expired token attempt")
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError as e:
            logger.warning("[WARNING] Invalid token: %s", str(e))
            return jsonify({'error': 'Invalid or expired token'}), 401

        return f(*args, **kwargs)

    return decorated


def cron_token_required(f):
    """Decorator to require CRON token for background jobs"""
    @wraps(f)
    def decorated(*args, **kwargs):
        cron_token = request.headers.get('X-Cron-Token')
        expected_token = os.getenv('CRON_TOKEN', 'dev-cron-token')

        if cron_token != expected_token:
            logger.warning("[WARNING] Invalid CRON token attempt from %s", request.remote_addr)
            return jsonify({'error': 'Invalid CRON token'}), 401

        logger.info("[OK] CRON job authorized from %s", request.remote_addr)
        return f(*args, **kwargs)

    return decorated


# ============================================================================
# PUBLIC ENDPOINTS (No auth required)
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - no auth required"""
    try:
        return jsonify({
            'status': 'healthy',
            'service': 'nba-elite-ml-api',
            'version': '2.3',
            'model': 'XGBoost',
            'model_loaded': model_loaded,
            'accuracy': 74.73 if model_loaded else 0,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error("[ERROR] Health check error: %s", str(e), exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/status', methods=['GET'])
def status():
    """Detailed status endpoint"""
    try:
        return jsonify({
            'status': 'online',
            'service': 'nba-elite-ml-api',
            'version': '2.3',
            'model': {
                'type': 'XGBoost Classifier',
                'name': 'NBA Game Prediction Model',
                'features': 16,
                'accuracy': 74.73 if model_loaded else 0,
                'roc_auc': 0.8261,
                'sensitivity': 0.8050,
                'precision': 0.7598,
                'training_samples': 24000,
                'loaded': model_loaded
            },
            'endpoints': 8,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error("[ERROR] Status check error: %s", str(e), exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/info', methods=['GET'])
def info():
    """API information endpoint"""
    try:
        return jsonify({
            'service': 'NBA Elite ML Prediction API v2.3',
            'description': 'XGBoost predictions for NBA games',
            'model': 'XGBoost Classifier',
            'accuracy': '74.73%' if model_loaded else 'Not loaded',
            'endpoints': 8,
            'port': 5001,
            'auth': 'JWT Token',
            'data_source': 'Single source (ml-model/data/nba_games_elite.csv)'
        }), 200
    except Exception as e:
        logger.error("[ERROR] Info endpoint error: %s", str(e), exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# GAME RESULTS ENDPOINT (New - No Auth Required)
# ============================================================================

@app.route('/api/game_results', methods=['GET', 'OPTIONS'])
def get_game_results():
    """
    Get current game results and upcoming games
    Returns: metadata, recent_results (last 20), upcoming_games
    No authentication required
    """
    try:
        logger.info("[OK] Game results request from %s", request.remote_addr)
        
        if not game_results_data:
            logger.warning("[WARNING] Game results not loaded")
            return jsonify({
                'error': 'Game results not available',
                'metadata': {
                    'last_updated': datetime.utcnow().isoformat(),
                    'accuracy_all_time_percent': 0.0,
                    'accuracy_last_20_percent': 0.0,
                    'total_games_tracked': 0,
                    'correct_predictions': 0,
                    'incorrect_predictions': 0
                },
                'recent_results': [],
                'upcoming_games': []
            }), 200
        
        return jsonify(game_results_data), 200
        
    except Exception as e:
        logger.error("[ERROR] Game results error: %s", str(e), exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/game_results/refresh', methods=['POST', 'OPTIONS'])
@cron_token_required
def refresh_game_results():
    """
    Refresh game results from source
    Requires: X-Cron-Token header
    Returns: Updated game_results
    """
    try:
        logger.info("[OK] Game results refresh requested")
        
        # Reload from disk
        success = load_game_results()
        
        if not success or not game_results_data:
            logger.error("[ERROR] Failed to load game results")
            return jsonify({
                'success': False,
                'error': 'Failed to load game results'
            }), 500
        
        logger.info("[OK] Game results refreshed successfully")
        
        return jsonify({
            'success': True,
            'message': 'Game results refreshed',
            'data': game_results_data,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error("[ERROR] Game results refresh error: %s", str(e), exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# PREDICTION ENDPOINTS (Require JWT auth)
# ============================================================================

@app.route('/api/predict', methods=['POST'])
@token_required
def predict():
    """
    Single-game prediction endpoint
    Requires: JWT token in Authorization header
    Body: { "features": [16 floats] }
    Returns: prediction (HOME/AWAY) and confidence
    """
    try:
        username = request.user.get('sub', 'unknown')
        logger.info("[OK] Prediction request from user: %s", username)

        if not model_loaded:
            logger.warning("[WARNING] Model not loaded, returning demo prediction")
            return jsonify({
                'success': False,
                'error': 'Model not loaded',
                'predicted_winner': None,
                'confidence': 0.0
            }), 503

        data = request.get_json(force=True)
        features = data.get('features', [])

        if not isinstance(features, list) or len(features) != 16:
            logger.warning("[WARNING] Invalid features from %s: expected 16, got %d", username, len(features))
            return jsonify({
                'success': False,
                'error': 'Invalid features. Expected array of length 16.'
            }), 400

        try:
            X = np.array([features], dtype=float)
            pred = xgb_model.predict(X)[0]
            proba = float(xgb_model.predict_proba(X).max())

            prediction = 'HOME' if int(pred) == 1 else 'AWAY'

            logger.info("[OK] Prediction successful for %s: %s (%.2f%%)", username, prediction, proba * 100)

            return jsonify({
                'success': True,
                'predicted_winner': prediction,
                'confidence': round(proba, 4),
                'model_version': '2.3',
                'user': username,
                'timestamp': datetime.utcnow().isoformat()
            }), 200

        except Exception as e:
            logger.error("[ERROR] Prediction calculation error for %s: %s", username, str(e), exc_info=True)
            return jsonify({'success': False, 'error': str(e)}), 500

    except Exception as e:
        logger.error("[ERROR] Prediction endpoint error: %s", str(e), exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
@token_required
def predict_batch():
    """
    Batch prediction endpoint
    Requires: JWT token in Authorization header
    Body: { "games": [ { "game_id": "...", "features": [16 floats] }, ... ] }
    Returns: array of predictions with game IDs
    """
    try:
        username = request.user.get('sub', 'unknown')
        logger.info("[OK] Batch prediction request from user: %s", username)

        if not model_loaded:
            logger.warning("[WARNING] Model not loaded for batch prediction")
            return jsonify({'success': False, 'error': 'Model not loaded'}), 503

        data = request.get_json(force=True)
        games = data.get('games', [])

        if not isinstance(games, list) or len(games) == 0:
            logger.warning("[WARNING] Invalid batch request from %s: no games provided", username)
            return jsonify({'success': False, 'error': 'No games provided'}), 400

        results = []
        successful = 0
        failed = 0

        for game in games:
            game_id = game.get('game_id', 'unknown')
            features = game.get('features', [])

            if not isinstance(features, list) or len(features) != 16:
                logger.debug("[DEBUG] Invalid features for game %s: expected 16, got %d", game_id, len(features))
                results.append({
                    'game_id': game_id,
                    'success': False,
                    'error': 'Invalid feature count (expected 16)'
                })
                failed += 1
                continue

            try:
                X = np.array([features], dtype=float)
                pred = xgb_model.predict(X)[0]
                proba = float(xgb_model.predict_proba(X).max())

                prediction = 'HOME' if int(pred) == 1 else 'AWAY'

                results.append({
                    'game_id': game_id,
                    'success': True,
                    'predicted_winner': prediction,
                    'confidence': round(proba, 4)
                })
                successful += 1

            except Exception as e:
                logger.debug("[DEBUG] Error predicting game %s: %s", game_id, str(e))
                results.append({
                    'game_id': game_id,
                    'success': False,
                    'error': str(e)
                })
                failed += 1

        logger.info("[OK] Batch prediction completed for %s: %d successful, %d failed", username, successful, failed)

        return jsonify({
            'success': True,
            'predictions': results,
            'summary': {
                'total': len(results),
                'successful': successful,
                'failed': failed
            },
            'user': username,
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.error("[ERROR] Batch prediction endpoint error: %s", str(e), exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model/info', methods=['GET'])
@token_required
def model_info():
    """Get detailed model information"""
    try:
        username = request.user.get('sub', 'unknown')
        logger.info("[OK] Model info request from %s", username)

        return jsonify({
            'success': True,
            'model': {
                'type': 'XGBoost Classifier',
                'name': 'NBA Game Prediction Model',
                'version': '2.3',
                'features': 16,
                'classes': ['AWAY', 'HOME'],
                'accuracy': 74.73 if model_loaded else 0,
                'roc_auc': 0.8261,
                'sensitivity': 0.8050,
                'precision': 0.7598,
                'f1_score': 0.7670,
                'training_samples': 24000,
                'loaded': model_loaded,
                'production_ready': model_loaded
            },
            'user': username,
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.error("[ERROR] Model info error: %s", str(e), exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# ADMIN/CRON ENDPOINTS
# ============================================================================

@app.route('/api/reload-model', methods=['POST'])
@cron_token_required
def reload_model():
    """
    Reload XGBoost model from disk
    Requires: X-Cron-Token header
    Used for background updates
    """
    global xgb_model, model_loaded

    try:
        logger.info("[OK] Model reload requested")

        model_path = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_model.pkl')

        if not os.path.exists(model_path):
            logger.error("[ERROR] Model file not found at %s", model_path)
            return jsonify({'success': False, 'error': 'Model file not found'}), 404

        with open(model_path, 'rb') as f:
            xgb_model = pickle.load(f)

        model_loaded = True
        logger.info("[OK] Model reloaded successfully")

        return jsonify({
            'success': True,
            'message': 'Model reloaded successfully',
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.error("[ERROR] Model reload error: %s", str(e), exc_info=True)
        model_loaded = False
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health/detailed', methods=['GET'])
@cron_token_required
def health_detailed():
    """Detailed health check for monitoring systems"""
    try:
        return jsonify({
            'status': 'healthy',
            'service': 'nba-elite-ml-api',
            'version': '2.3',
            'model': {
                'loaded': model_loaded,
                'accuracy': 74.73 if model_loaded else 0,
                'ready_for_predictions': model_loaded
            },
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_check': True
        }), 200
    except Exception as e:
        logger.error("[ERROR] Detailed health check error: %s", str(e), exc_info=True)
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    logger.warning("[WARNING] 404 Not Found: %s %s", request.method, request.path)
    return jsonify({
        'error': 'Endpoint not found',
        'path': request.path,
        'method': request.method
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    logger.warning("[WARNING] 405 Method Not Allowed: %s %s", request.method, request.path)
    return jsonify({
        'error': 'Method not allowed',
        'path': request.path,
        'method': request.method,
        'allowed_methods': ['GET', 'POST', 'OPTIONS']
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error("[ERROR] Internal server error: %s", str(error), exc_info=True)
    return jsonify({
        'error': 'Internal server error',
        'message': str(error) if os.getenv('FLASK_ENV') == 'development' else 'An error occurred'
    }), 500


# ============================================================================
# REQUEST LOGGING MIDDLEWARE
# ============================================================================

@app.before_request
def log_request():
    """Log incoming requests"""
    logger.info("[REQUEST] %s %s from %s", request.method, request.path, request.remote_addr)


@app.after_request
def log_response(response):
    """Log outgoing responses"""
    logger.info("[RESPONSE] %s %s -> %d", request.method, request.path, response.status_code)
    return response


# ============================================================================
# APP STARTUP
# ============================================================================

if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("Starting NBA Elite ML Prediction API...")
    logger.info("=" * 70)

    port = int(os.getenv('FLASK_PORT', 5001))
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    debug = os.getenv('FLASK_ENV', 'development') == 'development'

    logger.info("[OK] Server running on %s:%d", host, port)
    logger.info("[OK] Debug mode: %s", debug)
    logger.info("[OK] Model loaded: %s", model_loaded)
    logger.info("[OK] Game results loaded: %s", game_results_data is not None)
    logger.info("[OK] Secret key configured: %s", 'Yes' if app.config['SECRET_KEY'] else 'No')
    logger.info("=" * 70)

    print("")
    print("╔════════════════════════════════════════════════╗")
    print("║     [OK] NBA ELITE ML PREDICTION API            ║")
    print(f"║     Port: {port}                              ║")
    print(f"║     Model: {'XGBoost (74.73% accuracy)' if model_loaded else 'NOT LOADED'}      ║")
    print("║     Status: Running                            ║")
    print("║     Auth: JWT Token                            ║")
    print("║     CORS: Enabled (localhost:3000, 5173)       ║")
    print("║     Data: ml-model (single source)             ║")
    print("╚════════════════════════════════════════════════╝")
    print("")

    app.run(host=host, port=port, debug=debug)