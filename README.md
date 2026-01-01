# nba-elite-prediction-platform
AI powered NBA prediction using elite ranking features
# Terminal 1: ML-Model
cd ml-model && python app.py

# Terminal 2: Backend
cd backend && npm start

# Terminal 3: Ready for frontend
npm run dev

Test it works
curl http://localhost:5001/health
# Should return: {"status": "healthy", "model_loaded": true}


🔐 Authentication (Simple & Secure)
User Logs In
text
Google OAuth → JWT Token → Stored in localStorage
Every Protected Request
text
Authorization: Bearer <JWT_TOKEN>
Protected Endpoints
✅ POST /api/predictions/create - Create prediction

✅ GET /api/games/upcoming - Get upcoming games

✅ GET /api/predictions/history - Get past predictions

✅ GET /api/predictions/stats - Get accuracy metrics

Public Endpoints
✅ POST /auth/google-login - Login

✅ GET /health - Health check

✅ GET /status - Detailed status



📊 What Gets Tracked
For every prediction, track:

✅ WHO: created_by_user = "alice"

✅ WHAT: predicted_winner = "HOME", predicted_confidence = 0.85

✅ WHEN: created_at = "2025-12-31T10:05:17"

✅ GAME: game_id, home_team, away_team

✅ RESULT: actual_winner, actual_score_home, actual_score_away

✅ CORRECT: is_correct = TRUE/FALSE

✅ VERIFIED: result_updated_at, status = "completed"

✅ LOGGED: Complete audit trail in api_logs.log