import os
from dotenv import load_dotenv
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Load environment variables FIRST
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== APP INITIALIZATION ====================
app = Flask(__name__)
CORS(app)
# ✅ ADD THE SECURITY HEADERS HE
@app.after_request
def set_security_headers(response):
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin-allow-popups'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    return response

# ==================== LOAD PREDICTOR ====================
try:
    # Note: predictor_elite.py needs to exist or comment out for now
    # from predictor_elite import ElitePredictor
    # predictor = ElitePredictor(...)
    predictor = None
    logger.info("✓ Predictor initialization skipped (demo mode)")
except Exception as e:
    logger.error(f"✗ Failed to load ElitePredictor: {e}")
    predictor = None

# ==================== IMPORT AUTH ====================
from auth import (
    token_required, 
    handle_register, 
    handle_login, 
    handle_get_current_user,
    handle_logout,
    handle_change_password,
    handle_delete_account,
    generate_token,
    users_db,
    User
)

# ==================== STARTUP INFO ====================
print("""
╔══════════════════════════════════════════════════════════════════╗
║           🏀 NBA ELITE PREDICTION API - v2.2 (Google OAuth)      ║
║           Production-Grade XGBoost + Google Authentication       ║
║           Accuracy: 74.73% | ROC-AUC: 0.8261                    ║
╚══════════════════════════════════════════════════════════════════╝
""")

# ==================== AUTH ENDPOINTS ====================

@app.route('/auth/register', methods=['POST'])
def register():
    """Legacy register (optional - Google OAuth primary)"""
    try:
        data = request.json
        result, status_code = handle_register(data)
        return jsonify(result), status_code
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/auth/login', methods=['POST'])
def login():
    """Legacy login (optional - Google OAuth primary)"""
    try:
        data = request.json
        result, status_code = handle_login(data)
        return jsonify(result), status_code
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/auth/google-login', methods=['POST'])
def google_login():
    """Primary: Google OAuth login - NO passwords stored!"""
    try:
        data = request.json
        
        if not data or 'id_token' not in data:
            logger.warning("Google login attempt: Missing id_token in request")
            return jsonify({'error': 'Missing id_token'}), 400
        
        id_token = data['id_token']
        logger.info(f"Attempting Google OAuth verification...")
        
        # Import BEFORE try block - catches ImportError immediately
        import google.oauth2.id_token as google_id_token
        from google.auth.transport import requests as google_requests
        
        try:
            # Check GOOGLE_CLIENT_ID exists
            google_client_id = os.getenv('GOOGLE_CLIENT_ID')
            if not google_client_id:
                logger.error("CRITICAL: GOOGLE_CLIENT_ID not set in environment")
                return jsonify({'error': 'Server misconfiguration: Missing GOOGLE_CLIENT_ID'}), 500
            
            logger.info(f"Using Client ID: {google_client_id[:40]}...")
            
            # Verify token with Google
            request_obj = google_requests.Request()
            payload = google_id_token.verify_oauth2_token(
                id_token,
                request_obj,
                google_client_id
            )
            
            logger.info(f"✓ Google token verified successfully")
            
            # Extract user info
            email = payload.get('email')
            if not email:
                logger.error("Google payload missing email field")
                return jsonify({'error': 'Google token missing email'}), 400
            
            username = email.split('@')[0].replace('.', '_')
            name = payload.get('name', username)
            picture = payload.get('picture', '')
            
            logger.info(f"Extracted user: {username} ({email})")
            
            # Create or get user
            if username not in users_db:
                user = User(username, email, 'google-oauth-no-password')
                user.name = name
                user.picture = picture
                users_db[username] = user
                logger.info(f"✓ New Google user created: {username}")
            else:
                user = users_db[username]
                user.update_last_login()
                logger.info(f"✓ Existing user logged in: {username}")
            
            # Generate JWT token
            token = generate_token(username)
            logger.info(f"✓ JWT token generated")
            
            return jsonify({
                'success': True,
                'message': 'Google login successful',
                'user': {
                    'username': user.username,
                    'email': user.email,
                    'name': getattr(user, 'name', username),
                    'picture': getattr(user, 'picture', '')
                },
                'token': token
            }), 200
        
        except ValueError as e:
            # Token verification failed (expired, invalid signature, etc)
            logger.error(f"✗ Token verification failed: {str(e)}")
            return jsonify({'error': f'Token verification failed: {str(e)}'}), 401
        
        except Exception as e:
            # Any other error
            logger.error(f"✗ {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': f'{type(e).__name__}: {str(e)}'}), 401
    
    except Exception as e:
        logger.error(f"✗ Google login error: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/auth/me', methods=['GET'])
@token_required
def get_current_user():
    """Get current user info"""
    try:
        username = request.user['username']
        result, status_code = handle_get_current_user(username)
        return jsonify(result), status_code
    except Exception as e:
        logger.error(f"Get current user error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/auth/logout', methods=['POST'])
@token_required
def logout():
    """Logout user"""
    try:
        username = request.user['username']
        result, status_code = handle_logout(username)
        return jsonify(result), status_code
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({'error': str(e)}), 500

# ==================== HEALTH & STATUS ====================

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    status = 'healthy' if predictor is not None else 'degraded'
    return jsonify({
        'status': status,
        'service': 'nba-elite-prediction-api',
        'version': '2.2',
        'auth': 'Google OAuth',
        'model_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/status', methods=['GET'])
def status():
    """Detailed status"""
    if predictor is None:
        return jsonify({'status': 'offline', 'message': 'Model not loaded'}), 503
    
    return jsonify({
        'status': 'online',
        'service': 'nba-elite-prediction-api',
        'version': '2.2',
        'auth': 'Google OAuth Ready',
        'model': {
            'type': 'XGBoost Classifier',
            'features': 16,
            'accuracy': 0.7473,
            'roc_auc': 0.8261
        },
        'timestamp': datetime.now().isoformat()
    }), 200

# ==================== PREDICTION ENDPOINTS ====================

@app.route('/predict', methods=['POST'])
@token_required
def predict():
    """Predict single game (protected)"""
    try:
        username = request.user['username']
        data = request.json

        if predictor is None:
            return jsonify({'success': False, 'error': 'Predictor not loaded'}), 503

        if not data or 'ranking_data' not in data:
            return jsonify({'success': False, 'error': 'Missing ranking_data'}), 400
        
        # Demo response (replace with real predictor.predict_game())
        prediction = {
            'success': True,
            'home_win_probability': 0.65,
            'away_win_probability': 0.35,
            'predicted_winner': 'HOME',
            'confidence': 0.65
        }

        response = {
            'success': True,
            'home_team': data.get('home_team', 'HOME'),
            'away_team': data.get('away_team', 'AWAY'),
            'prediction': prediction,
            'user': username,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Prediction by {username}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== METADATA ENDPOINTS ====================

@app.route('/metrics', methods=['GET'])
@token_required
def metrics():
    """Model metrics (protected)"""
    return jsonify({
        'success': True,
        'metrics': {
            'accuracy': 0.7473,
            'roc_auc': 0.8261,
            'sensitivity': 0.8050,
            'specificity': 0.6750,
            'best_iteration': 42
        },
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/features', methods=['GET'])
@token_required
def features():
    """Required features for prediction"""
    return jsonify({
        'success': True,
        'feature_count': 16,
        'features': [
            'OFF_RNK_DIFF', 'DEF_RNK_DIFF', 'PTS_AVG_DIFF', 'DEF_AVG_DIFF',
            'HOME_OFF_RANK', 'HOME_DEF_RANK', 'AWAY_OFF_RANK', 'AWAY_DEF_RANK',
            'HOME_RUNNING_OFF_RANK', 'HOME_RUNNING_DEF_RANK', 'OFF_MOMENTUM',
            'DEF_MOMENTUM', 'RANK_INTERACTION', 'PTS_RANK_INTERACTION',
            'HOME_COURT', 'GAME_NUMBER'
        ],
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/info', methods=['GET'])
def info():
    """API documentation"""
    return jsonify({
        'service': 'NBA Elite Prediction API v2.2',
        'description': 'Google OAuth + XGBoost NBA predictions',
        'auth': 'Google OAuth (primary) + Legacy username/password',
        'endpoints': {
            'POST /auth/google-login': 'Google OAuth login (PRIMARY)',
            'POST /auth/login': 'Legacy username/password',
            'GET /auth/me': 'Get user info (auth required)',
            'POST /auth/logout': 'Logout (auth required)',
            'POST /predict': 'Make prediction (auth required)',
            'GET /metrics': 'Model metrics (auth required)',
            'GET /features': 'Feature list (auth required)',
            'GET /health': 'Health check',
            'GET /info': 'API info'
        }
    }), 200

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'success': False, 'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5001))
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    
    print(f"\n✓ Server Configuration:")
    print(f"  • Host: {host}")
    print(f"  • Port: {port}")
    print(f"  • Google OAuth: {'✓ Loaded' if os.getenv('GOOGLE_CLIENT_ID') else '✗ Missing .env'}")
    print(f"\n✓ Primary Endpoints:")
    print(f"  ├─ POST /auth/google-login  ← Google OAuth (PRIMARY)")
    print(f"  ├─ POST /predict           ← Auth required")
    print(f"  ├─ GET  /metrics           ← Auth required")
    print(f"  └─ GET  /health")
    
    print(f"\n" + "="*70)
    print(f"🚀 Starting NBA Elite API (Google OAuth Ready)...")
    print(f"="*70)
    
    app.run(debug=debug, port=port, host=host, threaded=True)
