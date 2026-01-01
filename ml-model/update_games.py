#!/usr/bin/env python3
"""
NBA Game Update Script - Automated CSV Updates
Fetches latest results from ESPN API → Updates CSV → Notifies Backend
Runs via cron job (daily 2 AM recommended)
FIXED: Correctly parses ESPN API STATUS_FINAL status
"""

import json
import requests
from datetime import datetime
import os
import logging
import sys

RESULTS_DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'game_results.json')
ESPN_SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'game_updates.log'), encoding='utf-8'),
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
    
    return {
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
    }

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

def fetch_games():
    """Fetch all NBA games (completed + scheduled)"""
    try:
        logger.info("Fetching games from ESPN API...")
        response = requests.get(ESPN_SCOREBOARD, timeout=15)
        
        if response.status_code != 200:
            logger.error(f"ESPN API returned {response.status_code}")
            return [], []
        
        data = response.json()
        events = data.get('events', [])
        
        completed = []
        scheduled = []
        
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
                
                game_date = event.get('date', '')
                home_team = home_comp.get('team', {}).get('displayName', '').split()[-1]  # Last word only
                away_team = away_comp.get('team', {}).get('displayName', '').split()[-1]
                home_score = int(home_comp.get('score', '0') or 0)
                away_score = int(away_comp.get('score', '0') or 0)
                
                if status_type == 'STATUS_FINAL':
                    actual_winner = away_team if away_score > home_score else home_team
                    completed.append({
                        'date': game_date[:10],  # YYYY-MM-DD only
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'actual_winner': actual_winner,
                        'score_str': f"{away_score}-{home_score}"
                    })
                    logger.info(f"[COMPLETE] {away_team} {away_score} @ {home_team} {home_score}")
                
                elif status_type not in ['STATUS_IN_PROGRESS', 'STATUS_HALFTIME']:
                    scheduled.append({
                        'date': game_date,
                        'home_team': home_team,
                        'away_team': away_team
                    })
                    logger.info(f"[SCHEDULED] {away_team} @ {home_team}")
                
            except Exception as e:
                logger.debug(f"Error parsing event: {e}")
                continue
        
        logger.info(f"Found {len(completed)} completed, {len(scheduled)} scheduled")
        return completed, scheduled
        
    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        return [], []

def update_results(db, completed_games):
    """Update recent results (keep only last 20)"""
    if not completed_games:
        return 0
    
    updated = 0
    
    for game in completed_games:
        # Check if already in recent_results
        exists = any(
            r['date'] == game['date'] and 
            r['home_team'] == game['home_team'] and 
            r['away_team'] == game['away_team']
            for r in db['recent_results']
        )
        
        if not exists:
            db['recent_results'].append({
                'date': game['date'],
                'away_team': game['away_team'],
                'home_team': game['home_team'],
                'score': game['score_str'],
                'actual_winner': game['actual_winner'],
                'predicted': None,  # Will be filled by predictions_api.py
                'result': None      # Will be 'win' or 'loss'
            })
            updated += 1
            logger.info(f"[OK] Added result: {game['away_team']} @ {game['home_team']}")
    
    # Keep only last 20 results
    if len(db['recent_results']) > 20:
        db['recent_results'] = db['recent_results'][-20:]
        logger.info(f"[TRIM] Keeping last 20 results")
    
    return updated

def update_upcoming(db, scheduled_games):
    """Update upcoming games"""
    if not scheduled_games:
        return
    
    # Clear old upcoming games, add new ones
    db['upcoming_games'] = []
    
    for game in scheduled_games:
        db['upcoming_games'].append({
            'date': game['date'],
            'away_team': game['away_team'],
            'home_team': game['home_team'],
            'predicted': None,      # Will be filled by predictions_api.py
            'confidence': None
        })
    
    logger.info(f"[OK] Updated {len(db['upcoming_games'])} upcoming games")

def main():
    logger.info("=" * 70)
    logger.info("NBA GAME RESULTS - UPDATE")
    logger.info("=" * 70)
    
    db = load_db()
    
    completed, scheduled = fetch_games()
    
    if completed:
        updated = update_results(db, completed)
        logger.info(f"[OK] {updated} new results added")
    
    if scheduled:
        update_upcoming(db, scheduled)
    
    db['metadata']['last_updated'] = datetime.utcnow().isoformat()
    save_db(db)
    
    logger.info(f"[SUMMARY] Recent results: {len(db['recent_results'])}")
    logger.info(f"[SUMMARY] Upcoming games: {len(db['upcoming_games'])}")
    logger.info("=" * 70)

if __name__ == '__main__':
    main()