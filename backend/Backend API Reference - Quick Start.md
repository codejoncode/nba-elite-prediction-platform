🎯 Backend API Reference - Quick Start
🔐 Authentication Endpoints
1. Google OAuth Login (PRIMARY) ⭐
text
POST /auth/google-login
Content-Type: application/json

{
  "id_token": "google_id_token_from_frontend"
}

Response 200:
{
  "success": true,
  "user": {
    "username": "alice",
    "email": "alice@gmail.com",
    "name": "Alice Smith",
    "picture": "https://..."
  },
  "token": "eyJhbGciOiJIUzI1NiIs..."  ← Use this in future requests!
}
2. Get Current User
text
GET /auth/me
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...

Response 200:
{
  "success": true,
  "user": {
    "username": "alice",
    "email": "alice@gmail.com",
    "created_at": "2025-12-31T10:00:00"
  }
}
3. Logout
text
POST /auth/logout
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...

Response 200:
{
  "success": true,
  "message": "Logged out successfully"
}
🏀 Prediction Endpoints (All Protected)
1. Create Prediction (SAVE #1) ⭐
text
POST /api/predictions/create
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "game_id": "400588282",
  "home_team": "Los Angeles Lakers",
  "away_team": "Boston Celtics",
  "ranking_data": {
    "OFF_RNK_DIFF": 5,
    "DEF_RNK_DIFF": -3,
    "PTS_AVG_DIFF": 2.5,
    "DEF_AVG_DIFF": -1.2,
    "HOME_OFF_RANK": 8,
    "HOME_DEF_RANK": 12,
    "AWAY_OFF_RANK": 3,
    "AWAY_DEF_RANK": 15,
    "HOME_RUNNING_OFF_RANK": 7,
    "HOME_RUNNING_DEF_RANK": 11,
    "OFF_MOMENTUM": -1,
    "DEF_MOMENTUM": -1,
    "RANK_INTERACTION": -15,
    "PTS_RANK_INTERACTION": 12.5,
    "HOME_COURT": 1,
    "GAME_NUMBER": 10
  }
}

Response 201 (Created):
{
  "success": true,
  "message": "Prediction created and saved",
  "game": {
    "game_id": "400588282",
    "home_team": "Los Angeles Lakers",
    "away_team": "Boston Celtics"
  },
  "prediction": {
    "predicted_winner": "HOME",
    "home_win_probability": 0.65,
    "away_win_probability": 0.35,
    "confidence": 0.65,
    "confidence_pct": "65.00%"
  },
  "user": "alice",
  "timestamp": "2025-12-31T10:05:17"
}
2. Get Upcoming Games
text
GET /api/games/upcoming
Authorization: Bearer <jwt_token>

Response 200:
{
  "success": true,
  "games": [
    {
      "game_id": "400588282",
      "home_team": "Los Angeles Lakers",
      "away_team": "Boston Celtics",
      "game_date": "2025-12-31T22:00:00",
      "game_time_et": "08:00 PM ET",
      "arena": "Crypto.com Arena",
      "prediction": {
        "predicted_winner": "HOME",
        "confidence": 65.0
      }
    },
    ...
  ],
  "count": 5
}
3. Get Prediction History
text
GET /api/predictions/history
Authorization: Bearer <jwt_token>

Response 200:
{
  "success": true,
  "predictions": [
    {
      "game_id": "400588170",
      "home_team": "Los Angeles Lakers",
      "away_team": "Golden State Warriors",
      "predicted_winner": "HOME",
      "actual_winner": "AWAY",
      "is_correct": false,
      "confidence": 62.0,
      "result": "✗ INCORRECT",
      "result_date": "2025-12-30T03:00:00"
    },
    ...
  ],
  "count": 20,
  "accuracy": 65.0,
  "record": "13-7"
}
4. Get Accuracy Stats
text
GET /api/predictions/stats
Authorization: Bearer <jwt_token>

Response 200:
{
  "success": true,
  "stats": {
    "overall_accuracy": 74.73,
    "total_predictions": 250,
    "correct_predictions": 187,
    "incorrect_predictions": 63,
    "last_10_wins": 7,
    "last_20_wins": 14,
    "current_streak": 3,
    "longest_win_streak": 12,
    "longest_loss_streak": 4,
    "best_team": "Los Angeles Lakers",
    "best_team_accuracy": 81.25,
    "worst_team": "Detroit Pistons",
    "worst_team_accuracy": 32.14,
    "calculated_at": "2025-12-31T23:00:00"
  }
}
⚙️ Admin/CRON Endpoints
Update Game Results (CRON ONLY - Internal)
text
POST /api/games/update-results
X-Cron-Token: dev-token
Content-Type: application/json

Response 200:
{
  "success": true,
  "message": "Game results and metrics updated",
  "timestamp": "2025-12-31T23:00:00"
}

ERRORS:
- 401: Invalid X-Cron-Token
- 500: Database error
📊 Public Endpoints (No Auth)
Health Check
text
GET /health

Response 200:
{
  "status": "healthy",
  "service": "nba-elite-prediction-api",
  "version": "2.3",
  "auth": "Google OAuth + JWT",
  "model_loaded": true,
  "timestamp": "2025-12-31T10:05:00"
}
Detailed Status
text
GET /status

Response 200:
{
  "status": "online",
  "service": "nba-elite-prediction-api",
  "version": "2.3",
  "model": {
    "type": "XGBoost Classifier",
    "features": 16,
    "accuracy": 0.7473,
    "roc_auc": 0.8261,
    "sensitivity": 0.8050
  },
  "timestamp": "2025-12-31T10:05:00"
}
API Info
text
GET /info

Response 200:
{
  "service": "NBA Elite Prediction API v2.3",
  "description": "Production-grade XGBoost predictions with Google OAuth",
  "auth": "Google OAuth (primary) + JWT tokens",
  "accuracy": "74.73%",
  "roc_auc": 0.8261,
  "endpoints": { ... all endpoints listed ... }
}
🔑 HTTP Status Codes
Code	Meaning	When
200	OK	Successful GET/POST request
201	Created	Prediction successfully saved
400	Bad Request	Missing/invalid fields
401	Unauthorized	Invalid/missing JWT token
403	Forbidden	User account inactive
404	Not Found	Endpoint doesn't exist
500	Server Error	Internal error
503	Service Unavailable	Model not loaded
🛠️ Error Response Format
All errors follow this format:

json
{
  "success": false,
  "error": "Description of what went wrong",
  "user": "username" (if applicable),
  "timestamp": "2025-12-31T10:05:00"
}
Examples:

json
// Missing token
{
  "error": "Token is missing. Please provide a valid token in Authorization header.",
  "status_code": 401
}

// Invalid token
{
  "error": "Invalid or expired token",
  "status_code": 401
}

// Missing fields
{
  "error": "Missing required fields",
  "required": ["game_id", "home_team", "away_team", "ranking_data"],
  "status_code": 400
}
📋 Frontend Integration Checklist
1. Login Flow
 User clicks "Login with Google"

 Frontend requests Google ID token

 Frontend sends POST /auth/google-login with id_token

 Backend returns JWT token

 Frontend stores JWT in localStorage

2. API Calls (All protected)
 Every request includes: Authorization: Bearer <jwt_token>

 Handle 401 → redirect to login

 Handle 500 → show error message

3. Create Prediction
 Get 16 features from backend/database

 POST /api/predictions/create with features

 Show prediction to user (HOME/AWAY + confidence)

 Save prediction to localStorage (optional)

4. View Upcoming Games
 GET /api/games/upcoming on page load

 Display 5 upcoming games with predictions

 Show "No predictions yet" if none created

5. View Prediction History
 GET /api/predictions/history

 Display 20 past predictions

 Show ✓/✗ for each result

 Display overall record (13-7, etc.)

6. View Stats
 GET /api/predictions/stats

 Display 74.73% overall accuracy

 Show current streak (3 wins, etc.)

 Show best/worst team accuracy

🧪 Quick Testing (with curl)
1. Get health status
bash
curl http://localhost:5001/health
2. Google OAuth login
bash
curl -X POST http://localhost:5001/auth/google-login \
  -H "Content-Type: application/json" \
  -d '{"id_token": "your_google_token"}'
3. Get upcoming games (with token)
bash
curl http://localhost:5001/api/games/upcoming \
  -H "Authorization: Bearer your_jwt_token"
4. Create prediction (with token)
bash
curl -X POST http://localhost:5001/api/predictions/create \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{...ranking_data...}'
5. CRON job (internal only)
bash
curl -X POST http://localhost:5001/api/games/update-results \
  -H "X-Cron-Token: dev-token"
📚 Reference
Authentication: Google OAuth + JWT

Database: SQLite (dev) / PostgreSQL (production)

Predictor: XGBoost (74.73% accuracy)

Model Confidence: 0.0-1.0 (displayed as 0-100%)

Model Features: 16 (offensive/defensive rankings + momentum)

Supported Teams: All 30 NBA teams

Game Status: 'scheduled', 'completed'

Prediction Status: 'scheduled', 'completed'

Ready to build the React frontend! 🚀