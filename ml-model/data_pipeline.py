"""
Data pipeline to fetch NBA schedule and results from free APIs
Uses ESPN API (free, no auth required)
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from models import db, GameSchedule, Prediction, AccuracyMetrics
import logging

logger = logging.getLogger(__name__)

class NBAPipeline:
    """Fetch NBA schedule and game results from free APIs"""
    
    ESPN_SCHEDULE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/schedule"
    ESPN_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
    
    @staticmethod
    def fetch_upcoming_games(days_ahead=30):
        """Fetch upcoming NBA games from ESPN (free API)"""
        try:
            logger.info(f"Fetching NBA schedule for next {days_ahead} days...")
            response = requests.get(NBAPipeline.ESPN_SCHEDULE_URL, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            upcoming_games = []
            cutoff_date = datetime.utcnow() + timedelta(days=days_ahead)
            
            if 'events' in data:
                for event in data['events']:
                    try:
                        game_date = datetime.fromisoformat(event['date'].replace('Z', '+00:00'))
                        
                        if game_date > datetime.utcnow() and game_date < cutoff_date:
                            competitors = event.get('competitions', [{}])[0].get('competitors', [])
                            if len(competitors) >= 2:
                                home_team = competitors[1]['team']['name']
                                away_team = competitors[0]['team']['name']
                                
                                game_id = event.get('id')
                                game_time_et = game_date.strftime('%I:%M %p ET')
                                
                                upcoming_games.append({
                                    'game_id': game_id,
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'game_date': game_date,
                                    'game_time_et': game_time_et,
                                    'arena': event.get('competitions', [{}])[0].get('venue', {}).get('fullName', 'TBD'),
                                    'season': datetime.utcnow().year,
                                    'status': 'scheduled'
                                })
                    except (KeyError, IndexError, ValueError) as e:
                        logger.warning(f"Error parsing game event: {e}")
                        continue
            
            logger.info(f"✅ Found {len(upcoming_games)} upcoming games")
            return upcoming_games
            
        except requests.RequestException as e:
            logger.error(f"❌ Error fetching schedule from ESPN: {e}")
            return []
    
    @staticmethod
    def fetch_recent_results(days_back=7):
        """Fetch recent completed games from ESPN"""
        try:
            logger.info(f"Fetching NBA results from last {days_back} days...")
            response = requests.get(NBAPipeline.ESPN_SCHEDULE_URL, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            if 'events' in data:
                for event in data['events']:
                    try:
                        game_date = datetime.fromisoformat(event['date'].replace('Z', '+00:00'))
                        
                        status = event.get('status', {}).get('type', {}).get('name', '').lower()
                        if status == 'final' and game_date > cutoff_date and game_date < datetime.utcnow():
                            competitors = event.get('competitions', [{}])[0].get('competitors', [])
                            if len(competitors) >= 2:
                                home_team = competitors[1]['team']['name']
                                away_team = competitors[0]['team']['name']
                                home_score = int(competitors[1].get('score', 0))
                                away_score = int(competitors[0].get('score', 0))
                                
                                game_id = event.get('id')
                                winner = home_team if home_score > away_score else away_team
                                
                                results.append({
                                    'game_id': game_id,
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'home_score': home_score,
                                    'away_score': away_score,
                                    'actual_winner': winner,
                                    'game_date': game_date,
                                    'status': 'final'
                                })
                    except (KeyError, IndexError, ValueError) as e:
                        logger.warning(f"Error parsing result event: {e}")
                        continue
            
            logger.info(f"✅ Found {len(results)} recent game results")
            return results
            
        except requests.RequestException as e:
            logger.error(f"❌ Error fetching results from ESPN: {e}")
            return []
    
    @staticmethod
    def update_game_schedules(db_session):
        """Fetch latest games and update database"""
        upcoming = NBAPipeline.fetch_upcoming_games(days_ahead=30)
        
        for game in upcoming:
            existing = db_session.query(GameSchedule).filter_by(
                game_id=game['game_id']
            ).first()
            
            if not existing:
                new_game = GameSchedule(**game)
                db_session.add(new_game)
                logger.info(f"Added: {game['away_team']} @ {game['home_team']} on {game['game_date']}")
        
        db_session.commit()
        logger.info("✅ Game schedule updated")
    
    @staticmethod
    def update_game_results(db_session):
        """Fetch recent results and update predictions"""
        results = NBAPipeline.fetch_recent_results(days_back=7)
        
        for result in results:
            prediction = db_session.query(Prediction).filter_by(
                game_id=result['game_id']
            ).first()
            
            if prediction:
                prediction.actual_winner = result['actual_winner']
                prediction.actual_score_home = result['home_score']
                prediction.actual_score_away = result['away_score']
                prediction.is_correct = (prediction.predicted_winner == result['actual_winner'])
                prediction.status = 'completed'
                prediction.result_updated_at = datetime.utcnow()
                
                status = "✅ CORRECT" if prediction.is_correct else "❌ INCORRECT"
                logger.info(f"{status}: {result['away_team']} @ {result['home_team']}")
        
        db_session.commit()
        logger.info("✅ Game results updated")
    
    @staticmethod
    def calculate_accuracy_metrics(db_session):
        """Calculate and cache accuracy metrics"""
        completed = db_session.query(Prediction).filter(
            Prediction.is_correct.isnot(None)
        ).all()
        
        if not completed:
            logger.warning("⚠️ No completed predictions to calculate metrics")
            return
        
        total = len(completed)
        correct = sum(1 for p in completed if p.is_correct)
        incorrect = total - correct
        accuracy = correct / total if total > 0 else 0
        
        # Get recent predictions
        recent_20 = sorted(completed, key=lambda x: x.result_updated_at, reverse=True)[:20]
        recent_10 = recent_20[:10]
        
        last_10_wins = sum(1 for p in recent_10 if p.is_correct)
        last_20_wins = sum(1 for p in recent_20 if p.is_correct)
        
        # Calculate current streak
        current_streak = 0
        for p in reversed(sorted(completed, key=lambda x: x.result_updated_at)):
            if p.is_correct:
                current_streak += 1
            else:
                break
        
        # Calculate longest streaks
        longest_win = 0
        longest_loss = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for p in sorted(completed, key=lambda x: x.result_updated_at):
            if p.is_correct:
                current_win_streak += 1
                current_loss_streak = 0
                longest_win = max(longest_win, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                longest_loss = max(longest_loss, current_loss_streak)
        
        # Calculate team accuracy
        teams = set()
        for p in completed:
            teams.add(p.home_team)
            teams.add(p.away_team)
        
        team_accuracy = {}
        for team in teams:
            team_preds = [p for p in completed if p.home_team == team or p.away_team == team]
            if team_preds:
                team_correct = sum(1 for p in team_preds if p.is_correct)
                team_accuracy[team] = team_correct / len(team_preds)
        
        best_team = max(team_accuracy, key=team_accuracy.get) if team_accuracy else None
        worst_team = min(team_accuracy, key=team_accuracy.get) if team_accuracy else None
        
        # Get or create metrics record
        metrics = db_session.query(AccuracyMetrics).first()
        if not metrics:
            metrics = AccuracyMetrics()
        
        metrics.total_predictions = total
        metrics.correct_predictions = correct
        metrics.incorrect_predictions = incorrect
        metrics.overall_accuracy = accuracy
        metrics.last_10_wins = last_10_wins
        metrics.last_10_losses = 10 - last_10_wins if len(recent_10) >= 10 else len(recent_10) - last_10_wins
        metrics.last_20_wins = last_20_wins
        metrics.last_20_losses = 20 - last_20_wins if len(recent_20) >= 20 else len(recent_20) - last_20_wins
        metrics.current_streak = current_streak if recent_10 and recent_10[0].is_correct else -current_streak
        metrics.longest_win_streak = longest_win
        metrics.longest_loss_streak = longest_loss
        metrics.best_team = best_team
        metrics.best_team_accuracy = team_accuracy.get(best_team) if best_team else None
        metrics.worst_team = worst_team
        metrics.worst_team_accuracy = team_accuracy.get(worst_team) if worst_team else None
        metrics.calculated_at = datetime.utcnow()
        
        if not db_session.query(AccuracyMetrics).first():
            db_session.add(metrics)
        
        db_session.commit()
        
        logger.info(f"✅ Metrics calculated: {accuracy*100:.2f}% accuracy ({correct}-{incorrect})")


if __name__ == '__main__':
    games = NBAPipeline.fetch_upcoming_games(days_ahead=7)
    print(f"\n📅 Upcoming Games (Next 7 Days):")
    for game in games[:5]:
        print(f"  {game['away_team']} @ {game['home_team']} | {game['game_time_et']}")
    
    results = NBAPipeline.fetch_recent_results(days_back=3)
    print(f"\n📊 Recent Results (Last 3 Days):")
    for result in results[:5]:
        print(f"  {result['away_team']} {result['away_score']} @ {result['home_team']} {result['home_score']}")