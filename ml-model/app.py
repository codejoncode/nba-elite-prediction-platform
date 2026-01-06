from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging
from datetime import datetime, timedelta
import pickle
import numpy as np
import pandas as pd
import json
import requests
import traceback
from pathlib import Path

load_dotenv()

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_api_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CORS
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
PREDICTIONS_DB = DATA_DIR / 'predictions_history.json'
TEAM_STATS_DB = DATA_DIR / 'team_stats.json'

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Global vars
xgb_model = None
scaler = None
feature_columns = None
model_loaded = False
ESPN_SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'

# NBA Team abbreviation mapping
TEAM_MAPPING = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load XGBoost model, scaler, and features"""
    global xgb_model, scaler, feature_columns, model_loaded
    
    try:
        model_path = MODELS_DIR / 'xgboost_model.pkl'
        scaler_path = MODELS_DIR / 'scaler.pkl'
        features_path = MODELS_DIR / 'feature_columns.json'
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False
            
        with open(model_path, 'rb') as f:
            xgb_model = pickle.load(f)
        logger.info("✓ Model loaded")
        
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("✓ Scaler loaded")
        
        if features_path.exists():
            with open(features_path, 'r') as f:
                feature_columns = json.load(f)
            logger.info(f"✓ Features loaded ({len(feature_columns)} features)")
        
        model_loaded = True
        return True
        
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return False

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def load_predictions_db():
    """Load predictions history from JSON"""
    if PREDICTIONS_DB.exists():
        with open(PREDICTIONS_DB, 'r') as f:
            return json.load(f)
    return {'predictions': []}

def save_predictions_db(db):
    """Save predictions history to JSON"""
    with open(PREDICTIONS_DB, 'w') as f:
        json.dump(db, f, indent=2)

def load_team_stats():
    """Load team statistics"""
    if TEAM_STATS_DB.exists():
        with open(TEAM_STATS_DB, 'r') as f:
            return json.load(f)
    return {}

def save_team_stats(stats):
    """Save team statistics"""
    with open(TEAM_STATS_DB, 'w') as f:
        json.dump(stats, f, indent=2)

# ============================================================================
# FEATURE ENGINEERING FROM LIVE DATA
# ============================================================================

def get_team_abbr(team_name):
    """Convert full team name to abbreviation"""
    return TEAM_MAPPING.get(team_name, team_name[:3].upper())

def calculate_team_features(home_team, away_team, team_stats):
    """Calculate features for prediction using team statistics"""
    
    home_abbr = get_team_abbr(home_team)
    away_abbr = get_team_abbr(away_team)
    
    home_stats = team_stats.get(home_abbr, {})
    away_stats = team_stats.get(away_abbr, {})
    
    # Get stats with defaults
    home_off_rank = home_stats.get('off_rank', 15)
    home_def_rank = home_stats.get('def_rank', 15)
    away_off_rank = away_stats.get('off_rank', 15)
    away_def_rank = away_stats.get('def_rank', 15)
    
    home_pts_avg = home_stats.get('pts_avg', 110.0)
    away_pts_avg = away_stats.get('pts_avg', 110.0)
    home_pts_allowed = home_stats.get('pts_allowed_avg', 110.0)
    away_pts_allowed = away_stats.get('pts_allowed_avg', 110.0)
    
    # Calculate features
    features = {
        'OFF_RNK_DIFF': home_off_rank - away_off_rank,
        'DEF_RNK_DIFF': home_def_rank - away_def_rank,
        'PTS_AVG_DIFF': home_pts_avg - away_pts_avg,
        'DEF_AVG_DIFF': home_pts_allowed - away_pts_allowed,
        'HOME_OFF_RANK': home_off_rank,
        'HOME_DEF_RANK': home_def_rank,
        'AWAY_OFF_RANK': away_off_rank,
        'AWAY_DEF_RANK': away_def_rank,
        'HOME_RUNNING_OFF_RANK': home_stats.get('running_off_rank', home_off_rank),
        'HOME_RUNNING_DEF_RANK': home_stats.get('running_def_rank', home_def_rank),
        'OFF_MOMENTUM': home_stats.get('off_momentum', 0),
        'DEF_MOMENTUM': home_stats.get('def_momentum', 0),
        'RANK_INTERACTION': (home_off_rank - away_off_rank) * (home_def_rank - away_def_rank),
        'PTS_RANK_INTERACTION': (home_pts_avg - away_pts_avg) * (home_off_rank - away_off_rank),
        'HOME_COURT': 1,
        'GAME_NUMBER': home_stats.get('games_played', 1)
    }
    
    return features

def predict_game(home_team, away_team, team_stats):
    """Make prediction for a game"""
    if not model_loaded:
        # Fallback prediction
        return {
            'predicted_winner': home_team,
            'confidence': 0.55,
            'home_prob': 0.55,
            'away_prob': 0.45
        }
    
    try:
        features = calculate_team_features(home_team, away_team, team_stats)
        
        # Ensure correct feature order
        if feature_columns:
            feature_vector = np.array([features.get(col, 0) for col in feature_columns]).reshape(1, -1)
        else:
            feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Scale if scaler available
        if scaler:
            feature_vector = scaler.transform(feature_vector)
        
        # Predict
        probas = xgb_model.predict_proba(feature_vector)[0]
        home_prob = float(probas[1])
        away_prob = float(probas[0])
        
        predicted_winner = home_team if home_prob > 0.5 else away_team
        confidence = max(home_prob, away_prob)
        
        return {
            'predicted_winner': predicted_winner,
            'confidence': round(confidence, 3),
            'home_prob': round(home_prob, 3),
            'away_prob': round(away_prob, 3)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            'predicted_winner': home_team,
            'confidence': 0.55,
            'home_prob': 0.55,
            'away_prob': 0.45
        }

# ============================================================================
# ESPN API FUNCTIONS
# ============================================================================

def fetch_espn_games():
    """Fetch games from ESPN with date filtering"""
    try:
        logger.info("[ESPN] Fetching games...")
        
        # Get current date and next 3 days
        today = datetime.utcnow().date()
        date_range = [(today + timedelta(days=i)).strftime('%Y%m%d') for i in range(3)]
        
        all_games = []
        
        for date_str in date_range:
            url = f"{ESPN_SCOREBOARD}?dates={date_str}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', [])
                
                for event in events:
                    try:
                        status_type = event.get('status', {}).get('type', {}).get('name', '')
                        competitions = event.get('competitions', [])
                        
                        if not competitions:
                            continue
                        
                        comp = competitions[0]
                        competitors = comp.get('competitors', [])
                        
                        if len(competitors) < 2:
                            continue
                        
                        home_comp = next((c for c in competitors if c.get('homeAway') == 'home'), competitors[0])
                        away_comp = next((c for c in competitors if c.get('homeAway') == 'away'), competitors[1])
                        
                        game_id = event.get('id')
                        game_date = event.get('date', '')
                        home_team = home_comp.get('team', {}).get('displayName', '')
                        away_team = away_comp.get('team', {}).get('displayName', '')
                        home_score = int(home_comp.get('score', 0) or 0)
                        away_score = int(away_comp.get('score', 0) or 0)
                        
                        game_obj = {
                            'game_id': game_id,
                            'date': game_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'status': status_type,
                            'home_score': home_score,
                            'away_score': away_score
                        }
                        
                        all_games.append(game_obj)
                        
                    except Exception as e:
                        logger.debug(f"Parse error: {e}")
                        continue
        
        logger.info(f"✓ Fetched {len(all_games)} games")
        return all_games
        
    except Exception as e:
        logger.error(f"ESPN fetch error: {e}")
        return []

# ============================================================================
# SYNC AND UPDATE FUNCTIONS
# ============================================================================

def update_team_stats_from_games(games, team_stats):
    """Update team statistics from completed games"""
    
    for game in games:
        if game['status'] != 'STATUS_FINAL':
            continue
        
        home_team = get_team_abbr(game['home_team'])
        away_team = get_team_abbr(game['away_team'])
        home_score = game['home_score']
        away_score = game['away_score']
        
        # Initialize team stats if not exists
        if home_team not in team_stats:
            team_stats[home_team] = {
                'games_played': 0,
                'pts_avg': 110.0,
                'pts_allowed_avg': 110.0,
                'off_rank': 15,
                'def_rank': 15,
                'running_off_rank': 15,
                'running_def_rank': 15,
                'off_momentum': 0,
                'def_momentum': 0,
                'recent_scores': [],
                'recent_allowed': []
            }
        
        if away_team not in team_stats:
            team_stats[away_team] = {
                'games_played': 0,
                'pts_avg': 110.0,
                'pts_allowed_avg': 110.0,
                'off_rank': 15,
                'def_rank': 15,
                'running_off_rank': 15,
                'running_def_rank': 15,
                'off_momentum': 0,
                'def_momentum': 0,
                'recent_scores': [],
                'recent_allowed': []
            }
        
        # Update home team
        home_stats = team_stats[home_team]
        home_stats['recent_scores'].append(home_score)
        home_stats['recent_allowed'].append(away_score)
        home_stats['recent_scores'] = home_stats['recent_scores'][-10:]
        home_stats['recent_allowed'] = home_stats['recent_allowed'][-10:]
        home_stats['pts_avg'] = np.mean(home_stats['recent_scores'])
        home_stats['pts_allowed_avg'] = np.mean(home_stats['recent_allowed'])
        home_stats['games_played'] += 1
        
        # Update away team
        away_stats = team_stats[away_team]
        away_stats['recent_scores'].append(away_score)
        away_stats['recent_allowed'].append(home_score)
        away_stats['recent_scores'] = away_stats['recent_scores'][-10:]
        away_stats['recent_allowed'] = away_stats['recent_allowed'][-10:]
        away_stats['pts_avg'] = np.mean(away_stats['recent_scores'])
        away_stats['pts_allowed_avg'] = np.mean(away_stats['recent_allowed'])
        away_stats['games_played'] += 1
    
    # Recalculate rankings
    teams_list = list(team_stats.keys())
    
    # Offensive rankings (higher PPG = better rank)
    sorted_off = sorted(teams_list, key=lambda t: team_stats[t]['pts_avg'], reverse=True)
    for rank, team in enumerate(sorted_off, 1):
        team_stats[team]['off_rank'] = rank
        team_stats[team]['running_off_rank'] = rank
    
    # Defensive rankings (lower allowed = better rank)
    sorted_def = sorted(teams_list, key=lambda t: team_stats[t]['pts_allowed_avg'])
    for rank, team in enumerate(sorted_def, 1):
        team_stats[team]['def_rank'] = rank
        team_stats[team]['running_def_rank'] = rank
    
    return team_stats

def sync_predictions():
    """Main sync function - fetch games, make predictions, verify results"""
    logger.info("="*70)
    logger.info("SYNC: Starting predictions sync")
    logger.info("="*70)
    
    # Load databases
    predictions_db = load_predictions_db()
    team_stats = load_team_stats()
    
    # Fetch games
    games = fetch_espn_games()
    
    if not games:
        logger.warning("No games fetched")
        return None
    
    # Update team stats from completed games
    team_stats = update_team_stats_from_games(games, team_stats)
    save_team_stats(team_stats)
    
    # Process games
    completed_games = []
    upcoming_games = []
    
    for game in games:
        game_id = game['game_id']
        status = game['status']
        
        # Check if prediction already exists
        existing = next((p for p in predictions_db['predictions'] if p['game_id'] == game_id), None)
        
        if status == 'STATUS_FINAL':
            # Completed game
            actual_winner = game['home_team'] if game['home_score'] > game['away_score'] else game['away_team']
            
            if existing:
                # Update existing prediction with result
                existing['actual_winner'] = actual_winner
                existing['home_score'] = game['home_score']
                existing['away_score'] = game['away_score']
                existing['is_correct'] = (existing['predicted_winner'] == actual_winner)
                existing['verified_at'] = datetime.utcnow().isoformat()
                prediction = existing
            else:
                # Create new prediction (retroactive)
                pred_result = predict_game(game['home_team'], game['away_team'], team_stats)
                prediction = {
                    'game_id': game_id,
                    'date': game['date'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'predicted_winner': pred_result['predicted_winner'],
                    'confidence': pred_result['confidence'],
                    'home_prob': pred_result['home_prob'],
                    'away_prob': pred_result['away_prob'],
                    'actual_winner': actual_winner,
                    'home_score': game['home_score'],
                    'away_score': game['away_score'],
                    'is_correct': (pred_result['predicted_winner'] == actual_winner),
                    'predicted_at': datetime.utcnow().isoformat(),
                    'verified_at': datetime.utcnow().isoformat()
                }
                predictions_db['predictions'].append(prediction)
            
            completed_games.append(prediction)
            logger.info(f"{'✓' if prediction['is_correct'] else '✗'} {game['away_team']} @ {game['home_team']}: {actual_winner}")
        
        elif 'SCHEDULED' in status or 'PRE' in status:
            # Upcoming game
            if not existing:
                # Create new prediction
                pred_result = predict_game(game['home_team'], game['away_team'], team_stats)
                prediction = {
                    'game_id': game_id,
                    'date': game['date'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'predicted_winner': pred_result['predicted_winner'],
                    'confidence': pred_result['confidence'],
                    'home_prob': pred_result['home_prob'],
                    'away_prob': pred_result['away_prob'],
                    'predicted_at': datetime.utcnow().isoformat()
                }
                predictions_db['predictions'].append(prediction)
            else:
                prediction = existing
            
            upcoming_games.append(prediction)
            logger.info(f"🔮 {game['away_team']} @ {game['home_team']}: {prediction['predicted_winner']} ({prediction['confidence']:.1%})")
    
    # Save predictions database
    save_predictions_db(predictions_db)
    
    # Calculate metrics
    all_verified = [p for p in predictions_db['predictions'] if 'is_correct' in p]
    recent_20 = sorted(all_verified, key=lambda x: x.get('verified_at', ''), reverse=True)[:20]
    
    total = len(all_verified)
    correct = sum(1 for p in all_verified if p['is_correct'])
    accuracy_all = (correct / total * 100) if total > 0 else 0
    
    recent_correct = sum(1 for p in recent_20 if p['is_correct'])
    accuracy_20 = (recent_correct / len(recent_20) * 100) if len(recent_20) > 0 else 0
    
    metadata = {
        'last_updated': datetime.utcnow().isoformat(),
        'accuracy_all_time_percent': round(accuracy_all, 1),
        'accuracy_last_20_percent': round(accuracy_20, 1),
        'total_games_tracked': total,
        'correct_predictions': correct,
        'incorrect_predictions': total - correct,
        'record_wins': correct,
        'record_losses': total - correct,
        'upcoming_games_count': len(upcoming_games),
        'sync_status': 'success'
    }
    
    logger.info(f"📊 All-time: {correct}/{total} ({accuracy_all:.1f}%)")
    logger.info(f"📊 Last 20: {recent_correct}/{len(recent_20)} ({accuracy_20:.1f}%)")
    logger.info("="*70)
    
    return {
        'metadata': metadata,
        'recent_results': recent_20,
        'upcoming_games': upcoming_games,
        'all_results': all_verified
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/api/game_results', methods=['GET'])
def get_game_results():
    try:
        logger.info("[API] game_results requested")
        result = sync_predictions()
        
        if result:
            return jsonify(result), 200
        else:
            return jsonify({'error': 'Sync failed'}), 500
            
    except Exception as e:
        logger.error(f"[API] Error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/check-updates', methods=['POST'])
def check_updates():
    try:
        logger.info("[API] Manual sync requested")
        result = sync_predictions()
        
        if result:
            return jsonify({
                'success': True,
                'message': 'Sync successful',
                'data': result
            }), 200
        else:
            return jsonify({'success': False, 'error': 'Sync failed'}), 500
            
    except Exception as e:
        logger.error(f"[API] Sync error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == '__main__':
    logger.info("="*70)
    logger.info("NBA ELITE ML API - STARTING")
    logger.info("="*70)
    
    # Load model
    load_model()
    
    # Initial sync
    logger.info("Running initial sync...")
    sync_predictions()
    
    port = int(os.getenv('FLASK_PORT', 5001))
    logger.info(f"Server starting on port {port}")
    logger.info("="*70)
    
    app.run(host='0.0.0.0', port=port, debug=True)