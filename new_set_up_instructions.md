NBA Elite Prediction Platform - Setup Instructions
🎯 Overview
This system provides:

Real-time NBA game predictions using your trained XGBoost model
Persistent historical tracking of all predictions and results
Automated result verification when games complete
Live accuracy metrics (all-time and last 20 games)
Automatic updates every hour via scheduler


📋 Prerequisites

Python 3.8+
Node.js 16+ (for React frontend)
Your trained model files in models/ directory:

xgboost_model.pkl
scaler.pkl (optional but recommended)
feature_columns.json (optional but recommended)




🚀 Installation Steps
Step 1: Backend Setup
bash# Navigate to your project directory
cd nba-elite-prediction-platform

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create data directory
mkdir -p data

# Create .env file (optional)
echo "SECRET_KEY=your-secret-key-here" > .env
echo "FLASK_PORT=5001" >> .env
Step 2: Verify Model Files
Ensure you have these files in the models/ directory:
models/
├── xgboost_model.pkl          # Your trained model
├── scaler.pkl                 # Feature scaler (optional)
└── feature_columns.json       # Feature order (optional)
If you don't have scaler.pkl or feature_columns.json, the system will still work but with slightly lower accuracy.
Step 3: Initial Data Setup
bash# Run initial sync to fetch games and create databases
python app.py
This will:

Load your trained model
Fetch current NBA games from ESPN
Create data/predictions_history.json (persistent storage)
Create data/team_stats.json (team statistics)
Generate predictions for upcoming games

Step 4: Start the Scheduler (Optional but Recommended)
In a separate terminal, run:
bash# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Start scheduler
python scheduler.py
This will:

Update predictions every hour
Verify completed game results
Update accuracy metrics automatically

Step 5: Frontend Setup
bash# Navigate to frontend directory
cd frontend  # or wherever your React app is

# Install dependencies
npm install

# Start development server
npm start

📊 Data Files Explained
1. data/predictions_history.json
This is the permanent record of all predictions and results:
json{
  "predictions": [
    {
      "game_id": "401584893",
      "date": "2026-01-06T19:00:00Z",
      "home_team": "Boston Celtics",
      "away_team": "Chicago Bulls",
      "predicted_winner": "Boston Celtics",
      "confidence": 0.732,
      "home_prob": 0.732,
      "away_prob": 0.268,
      "actual_winner": "Boston Celtics",
      "home_score": 115,
      "away_score": 101,
      "is_correct": true,
      "predicted_at": "2026-01-06T10:00:00Z",
      "verified_at": "2026-01-06T22:00:00Z"
    }
  ]
}
2. data/team_stats.json
Rolling statistics for each team:
json{
  "BOS": {
    "games_played": 35,
    "pts_avg": 118.5,
    "pts_allowed_avg": 108.2,
    "off_rank": 1,
    "def_rank": 3,
    "running_off_rank": 1,
    "running_def_rank": 3,
    "off_momentum": 0,
    "def_momentum": 1,
    "recent_scores": [120, 115, 122, ...],
    "recent_allowed": [105, 98, 110, ...]
  }
}

🔄 How the System Works
Prediction Flow

Scheduler triggers (every hour) → calls /api/check-updates
ESPN API fetch → gets games for today + next 2 days
Team stats calculation → updates from completed games
Feature engineering → calculates 16 features per game
Model prediction → XGBoost generates probabilities
Save to database → stores in predictions_history.json

Result Verification Flow

Scheduler checks completed games
Compares prediction vs actual winner
Updates record with is_correct flag
Recalculates metrics:

All-time accuracy
Last 20 games accuracy
Win-loss record



Feature Engineering
The system automatically calculates these 16 features:
pythonfeatures = {
    'OFF_RNK_DIFF': home_off_rank - away_off_rank,
    'DEF_RNK_DIFF': home_def_rank - away_def_rank,
    'PTS_AVG_DIFF': home_pts_avg - away_pts_avg,
    'DEF_AVG_DIFF': home_pts_allowed - away_pts_allowed,
    'HOME_OFF_RANK': home_off_rank,
    'HOME_DEF_RANK': home_def_rank,
    'AWAY_OFF_RANK': away_off_rank,
    'AWAY_DEF_RANK': away_def_rank,
    'HOME_RUNNING_OFF_RANK': home_running_off_rank,
    'HOME_RUNNING_DEF_RANK': home_running_def_rank,
    'OFF_MOMENTUM': off_momentum,
    'DEF_MOMENTUM': def_momentum,
    'RANK_INTERACTION': off_rank_diff * def_rank_diff,
    'PTS_RANK_INTERACTION': pts_avg_diff * off_rank_diff,
    'HOME_COURT': 1,
    'GAME_NUMBER': games_played
}

🎨 Frontend Integration
Your React DashboardPage.jsx should work with minimal changes. The API returns:
javascript{
  "metadata": {
    "last_updated": "2026-01-06T15:30:00Z",
    "accuracy_all_time_percent": 68.5,
    "accuracy_last_20_percent": 75.0,
    "total_games_tracked": 127,
    "correct_predictions": 87,
    "incorrect_predictions": 40,
    "record_wins": 87,
    "record_losses": 40,
    "upcoming_games_count": 8,
    "sync_status": "success"
  },
  "recent_results": [ /* last 20 verified games */ ],
  "upcoming_games": [ /* next games with predictions */ ],
  "all_results": [ /* complete history */ ]
}

🐛 Troubleshooting
Issue: "Model not found"
Solution:
bash# Ensure model files exist
ls -la models/
# Should show: xgboost_model.pkl, scaler.pkl, feature_columns.json
Issue: "No games fetched from ESPN"
Solution:

Check internet connection
ESPN API might be rate-limiting
Try manual sync: curl -X POST http://localhost:5001/api/check-updates

Issue: "Accuracy lower than expected"
Reasons:

Feature mismatch - Real game features differ from training data
Team stats not initialized - First few predictions use defaults
Model overfitting - Training accuracy ≠ real-world accuracy

Solutions:

Let system run for 20+ games to build team stats
Retrain model on more recent data
Check ml_api_logs.log for feature values

Issue: "Predictions not saving"
Solution:
bash# Check data directory exists
mkdir -p data

# Check file permissions
chmod 755 data/

# Check logs
tail -f ml_api_logs.log

📈 Improving Accuracy
1. Collect More Data
bash# Run scheduler for 2-3 weeks
# System will build better team statistics
2. Retrain Model with Real Features
python# Extract features from predictions_history.json
# Retrain using actual game outcomes
# This creates a feedback loop for improvement
3. Add More Features
Consider adding:

Player injuries
Back-to-back games
Travel distance
Playoff implications

4. Ensemble Models
Train multiple models:

XGBoost
Random Forest
Neural Network
Average their predictions


🔒 Production Deployment
Heroku
bash# Create Procfile
echo "web: gunicorn app:app" > Procfile
echo "worker: python scheduler.py" >> Procfile

# Deploy
heroku create nba-elite-predictions
git push heroku main
Docker
dockerfileFROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5001"]

📊 Monitoring
Check Logs
bash# API logs
tail -f ml_api_logs.log

# Scheduler logs
tail -f scheduler.log
Health Check
bashcurl http://localhost:5001/health
View Database
bash# Pretty print predictions
python -m json.tool data/predictions_history.json | less

# Count predictions
cat data/predictions_history.json | jq '.predictions | length'

# Calculate accuracy
cat data/predictions_history.json | jq '[.predictions[] | select(.is_correct == true)] | length'

🎯 For Recruiters
This system demonstrates:
✅ Full-stack ML engineering

Flask REST API
React frontend
XGBoost ML model
Feature engineering

✅ Production-ready code

Error handling
Logging
Persistent storage
Automated scheduling

✅ Real-world ML application

Live data integration
Model deployment
Performance tracking
Result verification

✅ System design skills

API design
Database schema
Data pipelines
Monitoring


📞 Support
Check these files if you encounter issues:

ml_api_logs.log - API errors and predictions
scheduler.log - Automated sync status
data/predictions_history.json - All predictions
data/team_stats.json - Team statistics


🚀 Quick Start Summary
bash# 1. Install
pip install -r requirements.txt

# 2. Run API
python app.py

# 3. Run scheduler (separate terminal)
python scheduler.py

# 4. Start frontend (separate terminal)
cd frontend && npm start

# 5. Open browser
# http://localhost:3000
That's it! The system will automatically:

Fetch NBA games every hour
Generate predictions for upcoming games
Verify results when games complete
Update accuracy metrics in real-time