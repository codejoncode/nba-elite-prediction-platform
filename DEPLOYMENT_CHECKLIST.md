╔═══════════════════════════════════════════════════════════════════════════════╗
║ 🏀 NBA ELITE PREDICTOR - FINAL DEPLOYMENT CHECKLIST ║
║ ║
║ Follow this before moving to frontend ║
╚═══════════════════════════════════════════════════════════════════════════════╝

PHASE 1: CODE UPDATES (15 minutes)
═══════════════════════════════════════════════════════════════════════════════

□ STEP 1: Replace app.py
Command: cp app_production.py app.py
Verify: File replaced, 400+ lines, has @token_required decorator
Expected: No errors on import

□ STEP 2: Update models.py
Add to Prediction class:
- created_by_user = db.Column(db.String(100))
- created_at = db.Column(db.DateTime, default=datetime.utcnow)
- predicted_confidence = db.Column(db.Float, default=0.5)
- actual_score_home = db.Column(db.Integer, nullable=True)
- actual_score_away = db.Column(db.Integer, nullable=True)
- result_updated_at = db.Column(db.DateTime, nullable=True)
Verify: All 6 fields added
Expected: models.py imports without error

□ STEP 3: Update .env (ml-model/)
Set:
FLASK_PORT=5001
FLASK_HOST=0.0.0.0
FLASK_ENV=development
GOOGLE_CLIENT_ID=YOUR_ACTUAL_GOOGLE_CLIENT_ID
SECRET_KEY=generate-new-secret-key
CRON_TOKEN=dev-token
Verify: GOOGLE_CLIENT_ID is not placeholder
Expected: Load without warnings

□ STEP 4: Update .env (backend/)
Set:
PORT=5000
NODE_ENV=development
ML_MODEL_API=http://localhost:5001
FRONTEND_URL=http://localhost:5173
CRON_TOKEN=dev-token
Verify: All values set correctly
Expected: npm start doesn't show env errors

□ STEP 5: Replace server.js (backend/)
Command: cp server_consolidated.js server.js
Verify: File replaced, has proper route mounting
Expected: npm start shows all routes

PHASE 2: DATABASE SETUP (5 minutes)
═══════════════════════════════════════════════════════════════════════════════

□ STEP 6: Backup existing database (if any)
Command: cp nba_predictor.db nba_predictor.db.backup
Verify: Backup file created
Expected: Both files exist

□ STEP 7: Delete old database
Command: rm nba_predictor.db
Verify: File deleted
Expected: File no longer exists

□ STEP 8: Recreate database with new schema
Commands:
python
from app import app, db
with app.app_context():
db.create_all()
exit()
Verify: nba_predictor.db file created
Expected: No errors, file ~10KB

□ STEP 9: Verify schema
Command: sqlite3 nba_predictor.db ".schema Prediction"
Verify: See all new columns (created_by_user, created_at, etc)
Expected: 13+ columns in Prediction table

PHASE 3: STARTUP VERIFICATION (10 minutes)
═══════════════════════════════════════════════════════════════════════════════

□ STEP 10: Start ML-Model (Terminal 1)
Command: cd ml-model && python app.py
Look for:
✓ Model loaded
✓ Database initialized
✓ Running on 0.0.0.0:5001
Verify: No errors, all green checkmarks
Expected: Server listening on port 5001

□ STEP 11: Start Backend (Terminal 2)
Command: cd backend && npm start
Look for:
✓ Server Running on port 5000
✓ Routes mounted
✓ No connection errors
Verify: Shows startup message
Expected: Server listening on port 5000

□ STEP 12: Test Health Endpoint
Command: curl http://localhost:5001/health
Verify: Response includes:
"status": "healthy"
"model_loaded": true
"database": "Connected"
Expected: 200 OK with all green indicators

□ STEP 13: Test Status Endpoint
Command: curl http://localhost:5001/status
Verify: Response includes:
"status": "online"
"accuracy": 0.7473
"roc_auc": 0.8261
Expected: 200 OK with model metrics

PHASE 4: AUTHENTICATION TESTING (10 minutes)
═══════════════════════════════════════════════════════════════════════════════

□ STEP 14: Test Google OAuth endpoint exists
Command: curl -X POST http://localhost:5001/auth/google-login
Verify: Returns error about missing id_token (not 404)
Expected: 400 Bad Request (means endpoint exists!)

□ STEP 15: Test protected route blocks unauthorized
Command: curl http://localhost:5001/api/games/upcoming
Verify: Response includes:
"error": "Token is missing"
Expected: 401 Unauthorized

□ STEP 16: Create test user
Command: (via Google OAuth in frontend later)
Or manually in Python:
python
from app import app, users_db, User
app.app_context().push()
user = User('testuser', 'test@example.com', 'password')
users_db['testuser'] = user
Verify: User created in memory
Expected: No errors

PHASE 5: PREDICTION FLOW TESTING (15 minutes)
═══════════════════════════════════════════════════════════════════════════════

□ STEP 17: Generate test JWT token
Python:
python
from app import app, generate_token
app.app_context().push()
token = generate_token('testuser')
print(token)
Copy the token value
Verify: Token is long string with dots
Expected: eyJhbGciOiJIUzI1NiIs...

□ STEP 18: Test create prediction endpoint
Command: (replace YOUR_JWT_TOKEN with actual token)
curl -X POST http://localhost:5001/api/predictions/create
-H "Authorization: Bearer YOUR_JWT_TOKEN"
-H "Content-Type: application/json"
-d '{
"game_id": "test123",
"home_team": "Lakers",
"away_team": "Celtics",
"ranking_data": {
"OFF_RNK_DIFF": 5,
"DEF_RNK_DIFF": 3,
"PTS_AVG_DIFF": 2.5,
"DEF_AVG_DIFF": 1.2,
"HOME_OFF_RANK": 8,
"HOME_DEF_RANK": 10,
"AWAY_OFF_RANK": 5,
"AWAY_DEF_RANK": 8,
"HOME_RUNNING_OFF_RANK": 7,
"HOME_RUNNING_DEF_RANK": 10,
"OFF_MOMENTUM": 1,
"DEF_MOMENTUM": 1,
"RANK_INTERACTION": 15,
"PTS_RANK_INTERACTION": 12.5,
"HOME_COURT": 1,
"GAME_NUMBER": 10
}
}'
Verify: Response 201 Created with:
"success": true
"predicted_winner": "HOME" or "AWAY"
"confidence": 0.xx
Expected: Prediction returned successfully

□ STEP 19: Verify prediction saved to database
Command: sqlite3 nba_predictor.db "SELECT * FROM predictions WHERE game_id='test123';"
Verify: See one row with:
game_id = test123
home_team = Lakers
predicted_winner = HOME (or AWAY)
created_by_user = testuser
status = scheduled
Expected: Row exists in database

□ STEP 20: Test get upcoming games
Command: (replace YOUR_JWT_TOKEN)
curl http://localhost:5001/api/games/upcoming
-H "Authorization: Bearer YOUR_JWT_TOKEN"
Verify: Response 200 OK with:
"success": true
"games": []
"count": 0
Expected: Returns empty list or existing games from database

□ STEP 21: Check logs
Command: tail -20 api_logs.log
Verify: Last 20 lines include:
- Login/auth events
- Prediction creation
- Database saves
Expected: Log entries with timestamps and messages

PHASE 6: DATA PERSISTENCE TESTING (5 minutes)
═══════════════════════════════════════════════════════════════════════════════

□ STEP 22: Restart ML-Model and verify data persists
Command: Stop (Ctrl+C) and restart ML-Model
cd ml-model && python app.py
Then query database:
sqlite3 nba_predictor.db "SELECT COUNT(*) FROM predictions;"
Verify: Same count as before (data persists)
Expected: Data survives restart

□ STEP 23: Query by username
Command: sqlite3 nba_predictor.db "SELECT game_id, created_by_user FROM predictions LIMIT 5;"
Verify: See created_by_user field populated
Expected: Shows "testuser" or actual username

□ STEP 24: Check all fields populated
Command: sqlite3 nba_predictor.db "SELECT game_id, created_by_user, created_at, predicted_confidence, status FROM predictions LIMIT 1;"
Verify: All fields have values
Expected: Example:
test123|testuser|2025-12-31 10:05:17.123456|0.65|scheduled

PHASE 7: CRON JOB TESTING (10 minutes)
═══════════════════════════════════════════════════════════════════════════════

□ STEP 25: Test CRON endpoint authentication
Command: curl -X POST http://localhost:5001/api/games/update-results
Verify: Response includes:
"error": "Unauthorized"
Expected: 401 Unauthorized (correct behavior)

□ STEP 26: Test CRON endpoint with token
Command: (with correct CRON_TOKEN)
curl -X POST http://localhost:5001/api/games/update-results
-H "X-Cron-Token: dev-token"
Verify: Response 200 OK with:
"success": true
"message": "Game results and metrics updated"
Expected: CRON job runs successfully

□ STEP 27: Check CRON logs
Command: tail -5 api_logs.log | grep CRON
Verify: See CRON job entries
Expected: "CRON: Starting..." and "CRON: Complete..."

PHASE 8: FINAL INTEGRATION CHECK (5 minutes)
═══════════════════════════════════════════════════════════════════════════════

□ STEP 28: All three services running
Check Terminal 1: ML-Model running ✓
Check Terminal 2: Backend running ✓
Expected: Both servers healthy

□ STEP 29: Test complete flow

Create prediction (Terminal 1)

Query database (Terminal 2)

Check logs (Terminal 3)
Expected: All three show related events

□ STEP 30: Load test summary
Commands:
- curl http://localhost:5001/health
- curl http://localhost:5001/status
- curl http://localhost:5001/info
Verify: All endpoints respond
Expected: All 200 OK

PHASE 9: FRONTEND READINESS (Final Check)
═══════════════════════════════════════════════════════════════════════════════

□ STEP 31: Document API endpoints for frontend
Create file: FRONTEND_ENDPOINTS.txt
Include:
- All endpoint URLs
- Required headers
- Example requests
- Expected responses
Expected: Frontend developer has clear reference

□ STEP 32: Prepare frontend environment
File: frontend/.env
Set:
REACT_APP_API_URL=http://localhost:5001
REACT_APP_GOOGLE_CLIENT_ID=YOUR_GOOGLE_ID
Expected: Frontend ready to make API calls

□ STEP 33: Create frontend troubleshooting guide
Document:
- Common errors (401, 400, 500)
- JWT token handling
- CORS setup
- Logging from frontend
Expected: Frontend developer has solutions

VERIFICATION SUMMARY
═══════════════════════════════════════════════════════════════════════════════

Total Checks: 33
Required Passes: 30+

Authentication:
□ Google OAuth endpoint exists
□ JWT generation works
□ Protected routes require token
□ 401 without token
□ User creation works

Predictions:
□ Create prediction endpoint works
□ Prediction saves to database
□ created_by_user populated
□ created_at populated
□ Status field set correctly

Data Persistence:
□ Data survives restart
□ All fields retrievable
□ Database queries work

Logging:
□ api_logs.log file exists
□ Auth events logged
□ Prediction events logged
□ Errors logged

CRON:
□ Update results endpoint works
□ CRON token validation works
□ Games update successfully

Frontend Ready:
□ All endpoints documented
□ Error handling clear
□ Examples provided
□ Google OAuth clear

═══════════════════════════════════════════════════════════════════════════════

IF ALL 33 STEPS PASS: ✅ READY FOR FRONTEND!

If any step fails:

Check error message carefully

Review relevant documentation file

Check logs (api_logs.log)

Verify environment variables

Restart services and retry

═══════════════════════════════════════════════════════════════════════════════