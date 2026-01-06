#!/usr/bin/env python3
"""
NBA Game Update Script - Fetches Real Games from ESPN API
Updates game_results.json with completed and upcoming games
Runs via cron job or scheduled task (daily 2 AM recommended)
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
    """Load existing game results database"""
    try:
        if os.path.exists(RESULTS_DB_PATH):
            with open(RESULTS_DB_PATH, 'r', encoding='utf-8') as f:
                db = json.load(f)
                logger.info("[DB] Loaded existing database")
                return db
    except Exception as e:
        logger.warning("[DB] Could not load database: %s", e)

    # Default empty database
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
    """Save database to file"""
    try:
        os.makedirs(os.path.dirname(RESULTS_DB_PATH), exist_ok=True)
        with open(RESULTS_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
        logger.info("[DB] Saved database with %d recent + %d upcoming", 
                   len(db.get('recent_results', [])), 
                   len(db.get('upcoming_games', [])))
        return True
    except Exception as e:
        logger.error("[DB] Failed to save database: %s", e)
        return False


def fetch_all_games():
    """Fetch all NBA games (completed + scheduled) from ESPN API"""
    try:
        logger.info("[ESPN] Fetching games from ESPN API...")
        response = requests.get(ESPN_SCOREBOARD, timeout=15)

        if response.status_code != 200:
            logger.error("[ESPN] API returned status %d", response.status_code)
            return [], []

        data = response.json()
        events = data.get('events', [])
        logger.info("[ESPN] Got %d total events", len(events))

        completed = []
        upcoming = []

        for idx, event in enumerate(events):
            try:
                # Get status
                status_obj = event.get('status', {})
                status_type = status_obj.get('type', '')

                # Get competitions
                competitions = event.get('competitions', [])
                if not competitions:
                    continue

                comp = competitions[0]
                competitors = comp.get('competitors', [])
                if len(competitors) < 2:
                    continue

                # Find home and away
                home_comp = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                away_comp = next((c for c in competitors if c.get('homeAway') == 'away'), None)

                if not home_comp or not away_comp:
                    home_comp, away_comp = competitors[1], competitors[0]

                game_date = event.get('date', '')
                home_team = home_comp.get('team', {}).get('displayName', '')
                away_team = away_comp.get('team', {}).get('displayName', '')

                # Skip invalid teams
                if not home_team or not away_team:
                    continue

                home_score = int(home_comp.get('score', 0) or 0)
                away_score = int(away_comp.get('score', 0) or 0)

                # Parse status
                if status_type == 'STATUS_FINAL':
                    # Completed game
                    actual_winner = home_team if home_score > away_score else away_team
                    completed.append({
                        'date': game_date[:10],  # YYYY-MM-DD
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'actual_winner': actual_winner,
                        'score': f"{away_score}-{home_score}",
                        'predicted': None,
                        'confidence': None,
                        'result': None
                    })
                    logger.info("[ESPN] [FINAL] %s %d @ %s %d", away_team, away_score, home_team, home_score)

                elif status_type in ['STATUS_SCHEDULED', 'STATUS_POSTPONED']:
                    # Upcoming game
                    upcoming.append({
                        'date': game_date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'predicted': None,
                        'confidence': None
                    })
                    logger.info("[ESPN] [UPCOMING] %s @ %s | %s", away_team, home_team, game_date)

            except Exception as e:
                logger.debug("[ESPN] Error parsing event %d: %s", idx, e)
                continue

        logger.info("[ESPN] Found %d completed, %d upcoming games", len(completed), len(upcoming))
        return completed, upcoming

    except requests.exceptions.Timeout:
        logger.error("[ESPN] Request timeout")
        return [], []
    except Exception as e:
        logger.error("[ESPN] Error: %s", e)
        return [], []


def merge_results(db, completed_games):
    """Merge completed games into recent_results (avoid duplicates)"""
    updated_count = 0

    if not completed_games:
        logger.info("[MERGE] No completed games to add")
        return updated_count

    existing_recent = db.get('recent_results', [])

    for game in completed_games:
        # Check if already exists
        exists = any(
            r['date'] == game['date'] and 
            r['home_team'] == game['home_team'] and 
            r['away_team'] == game['away_team']
            for r in existing_recent
        )

        if not exists:
            existing_recent.append(game)
            updated_count += 1
            logger.info("[MERGE] Added: %s @ %s (%s)", game['away_team'], game['home_team'], game['date'])

    # Keep only last 20 results
    if len(existing_recent) > 20:
        removed = len(existing_recent) - 20
        existing_recent = existing_recent[-20:]
        logger.info("[MERGE] Trimmed to last 20 results (removed %d)", removed)

    db['recent_results'] = existing_recent
    return updated_count


def update_upcoming(db, upcoming_games):
    """Update upcoming games"""
    if not upcoming_games:
        logger.info("[UPCOMING] No upcoming games")
        return

    # Replace upcoming games entirely
    db['upcoming_games'] = upcoming_games
    logger.info("[UPCOMING] Updated %d upcoming games", len(upcoming_games))


def main():
    logger.info("=" * 90)
    logger.info("NBA GAME UPDATE - ESPN API FETCH")
    logger.info("=" * 90)

    # Load existing database
    db = load_db()

    # Fetch real games from ESPN
    completed, upcoming = fetch_all_games()

    # Merge results
    if completed:
        added = merge_results(db, completed)
        logger.info("[RESULT] Added %d new completed games", added)

    # Update upcoming
    if upcoming:
        update_upcoming(db, upcoming)

    # Update metadata
    db['metadata']['last_updated'] = datetime.utcnow().isoformat()

    # Save
    if save_db(db):
        logger.info("[SUCCESS] Database saved")
    else:
        logger.error("[FAILURE] Failed to save database")
        return False

    logger.info("=" * 90)
    logger.info("[SUMMARY] Recent results: %d", len(db.get('recent_results', [])))
    logger.info("[SUMMARY] Upcoming games: %d", len(db.get('upcoming_games', [])))
    logger.info("=" * 90)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)