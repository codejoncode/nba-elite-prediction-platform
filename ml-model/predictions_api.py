#!/usr/bin/env python3
"""
NBA Predictions - Update game_results.json with predictions
Integrates with ElitePredictor for actual model predictions
Syncs predictions to frontend-ready database
"""

import json
import sys
import os
from datetime import datetime
import logging

# Add ml-model to path so we can import ElitePredictor
sys.path.insert(0, os.path.dirname(__file__))
from predictor_elite import ElitePredictor

RESULTS_DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'game_results.json')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'predictions.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_db():
    """Load game results database"""
    try:
        if os.path.exists(RESULTS_DB_PATH):
            with open(RESULTS_DB_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load database: {e}")
    return None

def save_db(db):
    """Save database"""
    try:
        with open(RESULTS_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
        logger.info(f"[OK] Database saved")
        return True
    except Exception as e:
        logger.error(f"Failed to save database: {e}")
        return False

def main():
    logger.info("=" * 70)
    logger.info("NBA PREDICTIONS - ADD FORECASTS TO GAMES")
    logger.info("=" * 70)
    
    # Load ElitePredictor
    try:
        logger.info("Loading ElitePredictor model...")
        predictor = ElitePredictor()
    except Exception as e:
        logger.error(f"Failed to load ElitePredictor: {e}")
        logger.info("Cannot proceed without model. Check paths:")
        logger.info("  - models/xgboost_elite_model.pkl")
        logger.info("  - models/scaler.pkl")
        logger.info("  - models/feature_columns.json")
        logger.info("  - models/metrics.json")
        return
    
    # Load database
    db = load_db()
    if not db:
        logger.error("Database not found")
        return
    
    logger.info(f"Loaded database with {len(db['recent_results'])} results")
    logger.info(f"Loaded database with {len(db['upcoming_games'])} upcoming")
    
    # ========================================================================
    # Step 1: Update recent results with predictions + win/loss markers
    # ========================================================================
    results_updated = 0
    
    for result in db['recent_results']:
        if result['predicted']:  # Already has prediction
            continue
        
        # TODO: Extract features from game data
        # For now, using demo data - replace with real feature extraction
        demo_features = {
            'OFF_RNK_DIFF': 5,
            'DEF_RNK_DIFF': -3,
            'PTS_AVG_DIFF': 2.5,
            'DEF_AVG_DIFF': -1.2,
            'HOME_OFF_RANK': 8,
            'HOME_DEF_RANK': 12,
            'AWAY_OFF_RANK': 3,
            'AWAY_DEF_RANK': 15,
            'HOME_RUNNING_OFF_RANK': 7,
            'HOME_RUNNING_DEF_RANK': 11,
            'OFF_MOMENTUM': -1,
            'DEF_MOMENTUM': -1,
            'RANK_INTERACTION': -15,
            'PTS_RANK_INTERACTION': 12.5,
            'HOME_COURT': 1,
            'GAME_NUMBER': 10
        }
        
        # Get prediction from ElitePredictor
        prediction = predictor.predict_game(demo_features)
        
        if prediction.get('success'):
            result['predicted'] = prediction['predicted_winner']
            
            # Determine if prediction was correct
            if result['actual_winner']:
                result['result'] = 'win' if prediction['predicted_winner'] == result['actual_winner'] else 'loss'
            
            results_updated += 1
            logger.info(f"[PRED] {result['away_team']} @ {result['home_team']}: {prediction['predicted_winner']} ({prediction['confidence_pct']})")
    
    # ========================================================================
    # Step 2: Add predictions to upcoming games
    # ========================================================================
    upcoming_updated = 0
    
    for game in db['upcoming_games']:
        if game['predicted']:  # Already has prediction
            continue
        
        # TODO: Extract features for this game from ESPN/database
        # For now, using demo data
        demo_features = {
            'OFF_RNK_DIFF': 3,
            'DEF_RNK_DIFF': 2,
            'PTS_AVG_DIFF': 1.5,
            'DEF_AVG_DIFF': 0.8,
            'HOME_OFF_RANK': 10,
            'HOME_DEF_RANK': 11,
            'AWAY_OFF_RANK': 12,
            'AWAY_DEF_RANK': 14,
            'HOME_RUNNING_OFF_RANK': 9,
            'HOME_RUNNING_DEF_RANK': 12,
            'OFF_MOMENTUM': 0,
            'DEF_MOMENTUM': 1,
            'RANK_INTERACTION': 6,
            'PTS_RANK_INTERACTION': 4.5,
            'HOME_COURT': 1,
            'GAME_NUMBER': 15
        }
        
        prediction = predictor.predict_game(demo_features)
        
        if prediction.get('success'):
            game['predicted'] = prediction['predicted_winner']
            game['confidence'] = round(prediction['confidence'] * 100, 1)
            upcoming_updated += 1
            logger.info(f"[PRED] {game['away_team']} @ {game['home_team']}: {prediction['predicted_winner']} ({game['confidence']}%)")
    
    # ========================================================================
    # Step 3: Calculate accuracy stats
    # ========================================================================
    recent_with_results = [r for r in db['recent_results'] if r['result']]
    
    if recent_with_results:
        wins = sum(1 for r in recent_with_results if r['result'] == 'win')
        accuracy_last_20 = (wins / len(recent_with_results)) * 100
        db['metadata']['accuracy_last_20_percent'] = round(accuracy_last_20, 2)
        logger.info(f"[STATS] Last {len(recent_with_results)} games: {wins} correct - {accuracy_last_20:.1f}% accuracy")
    
    # All-time accuracy (from metadata)
    total = db['metadata']['total_games_tracked']
    correct = db['metadata']['correct_predictions']
    if total > 0:
        accuracy_all_time = (correct / total) * 100
        db['metadata']['accuracy_all_time_percent'] = round(accuracy_all_time, 2)
        logger.info(f"[STATS] All-time: {correct}/{total} correct - {accuracy_all_time:.1f}% accuracy")
    
    # ========================================================================
    # Step 4: Save updated database
    # ========================================================================
    db['metadata']['last_updated'] = datetime.utcnow().isoformat()
    save_db(db)
    
    logger.info(f"[OK] {results_updated} results updated with predictions")
    logger.info(f"[OK] {upcoming_updated} upcoming games updated with predictions")
    logger.info("=" * 70)

if __name__ == '__main__':
    main()