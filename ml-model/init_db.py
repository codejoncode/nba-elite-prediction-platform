"""
Initialize database and load initial data
Run this once to set up the database with all tables and sample data
"""
import os
from datetime import datetime
from app import app
from models import db, GameSchedule, Prediction, AccuracyMetrics
from data_pipeline import NBAPipeline
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """Create all tables"""
    with app.app_context():
        logger.info("🔧 Creating database tables...")
        db.create_all()
        logger.info("✅ Database tables created")
        
        # Verify tables
        tables = db.inspect(db.engine).get_table_names()
        logger.info(f"✅ Tables in database: {', '.join(tables)}")


def create_default_metrics():
    """Create default AccuracyMetrics record if missing"""
    with app.app_context():
        logger.info("📊 Checking AccuracyMetrics record...")
        
        metrics = AccuracyMetrics.query.first()
        if not metrics:
            logger.info("Creating default AccuracyMetrics record...")
            default_metrics = AccuracyMetrics()
            db.session.add(default_metrics)
            db.session.commit()
            logger.info("✅ Default AccuracyMetrics created")
        else:
            logger.info("✅ AccuracyMetrics record already exists")


def populate_schedules():
    """Load upcoming NBA games"""
    with app.app_context():
        logger.info("📅 Fetching upcoming NBA games...")
        try:
            NBAPipeline.update_game_schedules(db.session)
            logger.info("✅ Schedule updated")
        except Exception as e:
            logger.error(f"⚠️  Could not fetch schedules: {e}")
            logger.info("This is OK - games will be fetched when needed")


def populate_past_results():
    """Load recent game results"""
    with app.app_context():
        logger.info("📊 Fetching recent game results...")
        try:
            NBAPipeline.update_game_results(db.session)
            logger.info("✅ Recent results loaded")
        except Exception as e:
            logger.error(f"⚠️  Could not fetch results: {e}")
            logger.info("This is OK - results will be updated via CRON job")


def generate_sample_predictions():
    """Generate sample predictions from upcoming games"""
    with app.app_context():
        logger.info("🎯 Generating sample predictions...")
        
        try:
            games = db.session.query(GameSchedule).filter(
                GameSchedule.status == 'scheduled'
            ).limit(5).all()
            
            if not games:
                logger.info("⚠️  No upcoming games found - skipping sample predictions")
                return
            
            created_count = 0
            for game in games:
                existing = db.session.query(Prediction).filter_by(
                    game_id=game.game_id
                ).first()
                
                if not existing:
                    confidence = 0.65
                    predicted_winner = game.home_team
                    
                    # ✅ NEW FIELDS: SAVE #1 fields
                    prediction = Prediction(
                        game_id=game.game_id,
                        home_team=game.home_team,
                        away_team=game.away_team,
                        predicted_winner=predicted_winner,
                        predicted_confidence=confidence,
                        game_date=game.game_date,
                        game_time_et=game.game_time_et,
                        status='pending',
                        predicted_at=datetime.utcnow(),
                        # ✅ NEW: Track who made it and when
                        created_by_user='sample_init',
                        created_at=datetime.utcnow(),
                        # ✅ SAVE #2 fields (will be updated when game completes)
                        actual_winner=None,
                        actual_score_home=None,
                        actual_score_away=None,
                        is_correct=None,
                        result_updated_at=None
                    )
                    db.session.add(prediction)
                    created_count += 1
            
            if created_count > 0:
                db.session.commit()
                logger.info(f"✅ {created_count} sample predictions created")
            else:
                logger.info("✅ All sample predictions already exist")
        
        except Exception as e:
            logger.error(f"⚠️  Error generating sample predictions: {e}")
            logger.info("This is OK - predictions can be created manually")


def show_status():
    """Display database status"""
    with app.app_context():
        games_count = db.session.query(GameSchedule).count()
        predictions_count = db.session.query(Prediction).count()
        completed_count = db.session.query(Prediction).filter(
            Prediction.is_correct.isnot(None)
        ).count()
        
        # Get metrics
        metrics = AccuracyMetrics.query.first()
        accuracy = round(metrics.overall_accuracy * 100, 2) if metrics else 0.0
        
        print("\n" + "="*60)
        print("📊 DATABASE STATUS")
        print("="*60)
        print(f"✅ Game Schedules:       {games_count}")
        print(f"✅ Total Predictions:    {predictions_count}")
        print(f"✅ Completed Predictions: {completed_count}")
        print(f"✅ Overall Accuracy:     {accuracy}%")
        print("="*60)
        print("\n📊 DATABASE SCHEMA")
        print("="*60)
        print("✅ Predictions table has:")
        print("   - SAVE #1 fields: created_by_user, created_at")
        print("   - SAVE #2 fields: actual_score_home, actual_score_away, result_updated_at")
        print("✅ GameSchedule table: upcoming games")
        print("✅ AccuracyMetrics table: model performance")
        print("="*60 + "\n")


if __name__ == '__main__':
    print("""
    🏀 NBA Elite Predictor - Database Initialization
    ================================================
    
    This script will:
    1. Create database tables (with 6 new tracking fields)
    2. Create default AccuracyMetrics record
    3. Fetch upcoming NBA games (optional - skips if ESPN unavailable)
    4. Fetch recent game results (optional - skips if ESPN unavailable)
    5. Generate sample predictions (optional - skips if no games)
    
    Note: ESPN API is free and requires no authentication
    All steps are optional and gracefully skip on error.
    """)
    
    try:
        init_database()
        create_default_metrics()
        populate_schedules()
        populate_past_results()
        generate_sample_predictions()
        show_status()
        logger.info("✅ Database initialization complete!")
        print("\n✅ Ready to run: python app.py\n")
        
    except Exception as e:
        logger.error(f"❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()