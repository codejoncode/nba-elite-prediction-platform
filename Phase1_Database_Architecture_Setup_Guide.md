🏀 Phase 1: Database Architecture - Setup Guide
📦 What We Created
1. models.py - Database Schema
Three core tables:

predictions - Store all predictions + outcomes

game_schedules - Store upcoming games from NBA

accuracy_metrics - Cache accuracy stats for quick retrieval

2. data_pipeline.py - Free Data Sources
Fetches data from ESPN API (free, no auth):

fetch_upcoming_games() - Next 30 days of games

fetch_recent_results() - Last 7 days of results

calculate_accuracy_metrics() - Compute running accuracy

update_game_results() - Match results to predictions

3. init_db.py - Database Setup Script
One-time initialization:

Creates all tables

Populates upcoming games

Loads recent results

Generates sample predictions

4. app_database.py - New API Routes
Four new endpoints:

GET /api/games/upcoming - Next 5 games (auto-predicted)

GET /api/predictions/history - Last 20 with results

GET /api/predictions/stats - Overall metrics

POST /api/games/update-results - Daily cron job

🚀 Installation Steps
Step 1: Install Dependencies
bash
cd ml-model
pip install flask-sqlalchemy requests pandas
Step 2: Copy Files
Copy these files to ml-model/:

models.py

data_pipeline.py

init_db.py

app_database.py

Step 3: Update app.py
At the TOP of app.py, add:

python
from models import db, Prediction, GameSchedule, AccuracyMetrics
from data_pipeline import NBAPipeline
After app = Flask(__name__), add:

python
# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nba_predictor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Create tables
with app.app_context():
    db.create_all()
Then copy the new routes from app_database.py into app.py

Step 4: Initialize Database
Run this ONCE:

bash
python init_db.py
Output should look like:

text
🏀 NBA Elite Predictor - Database Initialization
================================================

📅 Fetching upcoming NBA games...
✅ Found 150 upcoming games
✅ Schedule updated

📊 Fetching recent game results...
✅ Found 45 recent game results
✅ Recent results loaded

🎯 Generating sample predictions...
✅ Sample predictions created

==================================================
📊 DATABASE STATUS
==================================================
✅ Game Schedules: 150
✅ Total Predictions: 50
✅ Completed Predictions: 45
==================================================

✅ Database initialization complete!
Step 5: Test the API
Start Flask server:

bash
python app.py
Test endpoints:

bash
# Get upcoming games
curl http://localhost:5001/api/games/upcoming

# Get prediction history
curl http://localhost:5001/api/predictions/history

# Get accuracy stats
curl http://localhost:5001/api/predictions/stats
📊 Data Flow
text
1. ESPN API (Free)
   ├── Schedule: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/schedule
   └── Updates every game

2. Flask Backend
   ├── data_pipeline.py fetches data
   └── models.py stores in SQLite

3. Database (SQLite)
   ├── game_schedules (upcoming games)
   ├── predictions (your predictions)
   └── accuracy_metrics (running stats)

4. Frontend (React)
   └── Calls /api/games/upcoming, /api/predictions/history, etc
🔄 Keeping Data Fresh
Daily Cron Job
Set up a cron job to update results daily:

bash
# Linux/Mac: Add to crontab (crontab -e)
0 2 * * * curl -H "X-Cron-Token: your-token" http://localhost:5001/api/games/update-results

# Or use APScheduler (add to app.py):
from apscheduler.schedulers.background import BackgroundScheduler

def update_results_job():
    with app.app_context():
        from data_pipeline import NBAPipeline
        NBAPipeline.update_game_results(db.session)
        NBAPipeline.calculate_accuracy_metrics(db.session)

scheduler = BackgroundScheduler()
scheduler.add_job(func=update_results_job, trigger="cron", hour=2, minute=0)
scheduler.start()
💾 Your games.csv Integration
Your existing games.csv should be loaded once to backfill historical predictions:

python
# Add to init_db.py
def load_historical_games():
    """Load games.csv for backfill"""
    import pandas as pd
    
    df = pd.read_csv('games.csv')
    
    for _, row in df.iterrows():
        game = GameSchedule(
            game_id=f"{row['Home Team']}_{row['Away Team']}_{row['Date']}",
            home_team=row['Home Team'],
            away_team=row['Away Team'],
            game_date=pd.to_datetime(row['Date']),
            status='final',  # These are historical
            home_score=row['Home Score'],
            away_score=row['Away Score'],
            season=2024  # Adjust as needed
        )
        db.session.add(game)
    
    db.session.commit()
📈 What's Next (Phase 2)
Once database is running:

Update /predict endpoint to save predictions to DB

Add scheduling API to auto-predict upcoming games

Create frontend components for stats display

Set up cron jobs for daily updates

⚠️ Troubleshooting
Error: "No module named 'models'"
→ Make sure models.py is in the same directory as app.py

Error: "Cannot open database"
→ Check file permissions in ml-model/ directory
→ Delete nba_predictor.db and run init_db.py again

Error: "ESP API timeout"
→ ESPN API is sometimes slow. Try running again
→ If persistent, check internet connection

Database grows too large?
→ Archive old predictions: DELETE FROM predictions WHERE predicted_at < DATE('now', '-1 year')

✅ Success Checklist
 All files copied to ml-model/

 app.py updated with imports and config

 pip install flask-sqlalchemy requests pandas

 python init_db.py runs successfully

 nba_predictor.db file created

 API endpoints respond with data

 Frontend can call /api/games/upcoming

 Metrics showing accuracy stats

Once this is done, you're ready for Phase 2: Backend API Enhancement! 🚀