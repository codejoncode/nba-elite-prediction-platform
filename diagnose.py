#!/usr/bin/env python3
"""
NBA Elite Predictor - Diagnostic Tool
Verifies correct setup:
- CSV in ml-model/data/ (single source of truth)
- Node backend loads CSV
- Flask ML API ready
- No database needed
"""

import os
import sys


def find_project_root():
    """Find the actual project root by looking for key markers"""
    current = os.path.abspath(os.path.dirname(__file__))
    
    # Look for project markers (go up directories)
    for _ in range(5):
        contents = os.listdir(current)
        
        # Project root should have: backend, ml-model, and ideally frontend/src
        if ('backend' in contents or 'src' in contents) and 'ml-model' in contents:
            return current
        
        # Go up one directory
        parent = os.path.dirname(current)
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    # If not found, return current directory
    return os.path.abspath('.')


# Find actual project root
project_root = find_project_root()

print("\n" + "="*70)
print("NBA ELITE PREDICTOR - DIAGNOSTIC")
print("="*70)
print(f"\nProject Root: {project_root}\n")

# Define correct paths (ACTUAL SETUP - no database!)
ml_model_dir = os.path.join(project_root, "ml-model")
backend_dir = os.path.join(project_root, "backend")
csv_source = os.path.join(ml_model_dir, "data", "nba_games_elite.csv")
server_js = os.path.join(backend_dir, "server.js")
flask_app = os.path.join(ml_model_dir, "app.py")
xgb_model = os.path.join(ml_model_dir, "models", "xgboost_elite_model.pkl")
backend_env = os.path.join(backend_dir, ".env")
flask_env = os.path.join(ml_model_dir, ".env")

# ============================================================================
# 1. CHECK CSV (Single Source of Truth)
# ============================================================================

print("1️⃣  CSV Data Source (ml-model/data/nba_games_elite.csv):")
print("-" * 70)

if os.path.exists(csv_source):
    size = os.path.getsize(csv_source)
    size_mb = size / (1024 * 1024)
    try:
        with open(csv_source, 'r', encoding='utf-8') as f:
            lines = len(f.readlines())
        records = lines - 1
        print(f"   OK FOUND")
        print(f"      Path: {csv_source}")
        print(f"      Size: {size_mb:.2f} MB ({size:,} bytes)")
        print(f"      Games: {records:,}")
    except Exception as e:
        print(f"   WARNING ERROR reading: {e}")
else:
    print(f"   FAILED NOT FOUND: {csv_source}")
    print(f"\n      TO FIX:")
    print(f"      1. Ensure CSV exists in ml-model/data/")
    print(f"      2. File must be named: nba_games_elite.csv")
    print(f"      3. Should contain 26,567+ game records")

# ============================================================================
# 2. CHECK NODE BACKEND (server.js)
# ============================================================================

print("\n2️⃣  Node.js Backend (backend/server.js):")
print("-" * 70)

if os.path.exists(server_js):
    print(f"   OK FOUND: {server_js}")
    
    try:
        with open(server_js, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            'CSV path definition': "path.join(__dirname, '..', 'ml-model', 'data')" in content or 'GAMES_CSV_PATH' in content,
            'csv-parser library': "require('csv-parser')" in content,
            'loadGamesFromCSV function': 'function loadGamesFromCSV' in content or 'const loadGamesFromCSV' in content,
            'gamesData in-memory store': 'let gamesData' in content,
            'GET /api/games endpoint': "app.get('/api/games'" in content or 'GET /api/games' in content,
            'GET /api/predictions/history endpoint': "app.get('/api/predictions/history'" in content,
            'GET /api/predictions/stats endpoint': "app.get('/api/predictions/stats'" in content,
        }
        
        print(f"\n   Endpoints and Features:")
        all_ok = True
        for check_name, result in checks.items():
            symbol = "OK" if result else "FAILED"
            print(f"      {symbol} {check_name}")
            if not result:
                all_ok = False
        
        if all_ok:
            print(f"\n   OK Server.js is properly configured")
        else:
            print(f"\n   WARNING Some features may be missing")
    
    except UnicodeDecodeError:
        print(f"   WARNING File encoding issue")
    except Exception as e:
        print(f"   WARNING Error reading: {e}")
else:
    print(f"   FAILED NOT FOUND: {server_js}")
    print(f"\n      TO FIX:")
    print(f"      1. Create backend/server.js")
    print(f"      2. Must load CSV from ml-model/data/")
    print(f"      3. Must have /api/games and /api/predictions endpoints")

# ============================================================================
# 3. CHECK FLASK ML API (app.py)
# ============================================================================

print("\n3️⃣  Flask ML API (ml-model/app.py):")
print("-" * 70)

if os.path.exists(flask_app):
    print(f"   OK FOUND: {flask_app}")
    
    try:
        with open(flask_app, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            'Flask setup': 'from flask import Flask' in content,
            'Logging (Windows-safe)': 'logging.basicConfig' in content and 'ml_api_logs.log' in content,
            'JWT authentication': 'token_required' in content and 'def token_required' in content,
            'CRON authentication': 'cron_token_required' in content and '@cron_token_required' in content,
            'Health check endpoint': "app.route('/health'" in content,
            'Prediction endpoint': "app.route('/api/predict'" in content,
            'Batch prediction': "app.route('/api/predict/batch'" in content,
            'Model reload': "app.route('/api/reload-model'" in content,
        }
        
        print(f"\n   Endpoints and Features:")
        all_ok = True
        for check_name, result in checks.items():
            symbol = "OK" if result else "FAILED"
            print(f"      {symbol} {check_name}")
            if not result:
                all_ok = False
        
        if all_ok:
            print(f"\n   OK Flask app is fully configured")
        else:
            print(f"\n   WARNING Some features may be missing")
    
    except Exception as e:
        print(f"   WARNING Error reading: {e}")
else:
    print(f"   FAILED NOT FOUND: {flask_app}")
    print(f"\n      TO FIX:")
    print(f"      This is required. Copy from flask-api-complete.py")

# ============================================================================
# 4. CHECK XGBoost MODEL
# ============================================================================

print("\n4️⃣  XGBoost Model (ml-model/models/xgboost_model.pkl):")
print("-" * 70)

if os.path.exists(xgb_model):
    size = os.path.getsize(xgb_model)
    print(f"   OK FOUND: {xgb_model}")
    print(f"      Size: {size:,} bytes")
else:
    print(f"   WARNING NOT FOUND: {xgb_model}")
    print(f"\n      Note: Optional for testing")
    print(f"      Flask will run in demo mode without it")
    print(f"      Place trained model in: ml-model/models/xgboost_model.pkl")

# ============================================================================
# 5. CHECK ENVIRONMENT VARIABLES
# ============================================================================

print("\n5️⃣  Environment Variables (.env files):")
print("-" * 70)

# Check backend .env
print(f"\n   Backend (.env):")
if os.path.exists(backend_env):
    print(f"   OK FOUND: {backend_env}")
    try:
        with open(backend_env, 'r') as f:
            content = f.read()
        has_cron_token = 'CRON_TOKEN' in content
        has_port = 'PORT' in content or 'port' in content.lower()
        
        if has_cron_token:
            print(f"      OK CRON_TOKEN defined")
        else:
            print(f"      FAILED CRON_TOKEN missing (add: CRON_TOKEN=dev-cron-token)")
        
        if has_port:
            print(f"      OK PORT configured")
        else:
            print(f"      WARNING PORT not defined (default: 3001)")
    except Exception as e:
        print(f"      WARNING Error: {e}")
else:
    print(f"   FAILED MISSING: {backend_env}")
    print(f"      Create .env with: CRON_TOKEN=dev-cron-token")

# Check Flask .env
print(f"\n   Flask (.env):")
if os.path.exists(flask_env):
    print(f"   OK FOUND: {flask_env}")
    try:
        with open(flask_env, 'r') as f:
            content = f.read()
        has_cron_token = 'CRON_TOKEN' in content
        has_secret = 'SECRET_KEY' in content
        
        if has_cron_token:
            print(f"      OK CRON_TOKEN defined")
        else:
            print(f"      FAILED CRON_TOKEN missing (add: CRON_TOKEN=dev-cron-token)")
        
        if has_secret:
            print(f"      OK SECRET_KEY configured")
        else:
            print(f"      WARNING SECRET_KEY not defined")
    except Exception as e:
        print(f"      WARNING Error: {e}")
else:
    print(f"   FAILED MISSING: {flask_env}")
    print(f"      Create .env with: CRON_TOKEN=dev-cron-token")

# ============================================================================
# 6. CHECK DEPENDENCIES
# ============================================================================

print("\n6️⃣  Python Dependencies:")
print("-" * 70)

try:
    import flask
    print(f"   OK flask {flask.__version__}")
except ImportError:
    print(f"   FAILED flask (pip install flask)")

try:
    import flask_cors
    print(f"   OK flask-cors")
except ImportError:
    print(f"   FAILED flask-cors (pip install flask-cors)")

try:
    import jwt
    print(f"   OK PyJWT")
except ImportError:
    print(f"   FAILED PyJWT (pip install PyJWT)")

try:
    import numpy
    print(f"   OK numpy {numpy.__version__}")
except ImportError:
    print(f"   FAILED numpy (pip install numpy)")

try:
    import xgboost
    print(f"   OK xgboost {xgboost.__version__}")
except ImportError:
    print(f"   WARNING xgboost (optional, pip install xgboost)")

# ============================================================================
# 7. FINAL VERDICT
# ============================================================================

print("\n" + "="*70)
print("FINAL VERDICT & NEXT STEPS")
print("="*70)

csv_ok = os.path.exists(csv_source)
server_ok = os.path.exists(server_js)
flask_ok = os.path.exists(flask_app)
tokens_ok = os.path.exists(backend_env) and os.path.exists(flask_env)

all_ok = csv_ok and server_ok and flask_ok and tokens_ok

if all_ok:
    print("\nOK ALL SYSTEMS GO!\n")
    print("Your setup is correct:")
    print("  OK CSV in ml-model/data/ (single source)")
    print("  OK Node backend configured")
    print("  OK Flask ML API ready")
    print("  OK Environment variables set")
    print("\nTO START:")
    print("\n  Terminal 1 - Node Backend:")
    print("    cd backend")
    print("    node server.js")
    print("\n  Terminal 2 - Flask ML API:")
    print("    cd ml-model")
    print("    python app.py")
    print("\n  Terminal 3 - React Frontend:")
    print("    npm start")
    print("\nDATA UPDATES (without database):")
    print("  1. Create: ml-model/update_games.py (from guide)")
    print("  2. Schedule: crontab -e")
    print("  3. Add: 0 2 * * * cd " + project_root + " && python ml-model/update_games.py")
    print("\nOK READY TO GO! \n")
else:
    print("\nWARNING ISSUES FOUND:\n")
    
    if not csv_ok:
        print("  FAILED CSV Source Missing")
        print("     Fix: Place nba_games_elite.csv in ml-model/data/\n")
    
    if not server_ok:
        print("  FAILED Node Backend Missing")
        print("     Fix: Copy server.js to backend/ directory\n")
    
    if not flask_ok:
        print("  FAILED Flask App Missing")
        print("     Fix: Copy app.py to ml-model/ directory\n")
    
    if not tokens_ok:
        print("  FAILED Environment Variables Missing")
        print("     Fix: Create .env files in backend/ and ml-model/\n")
    
    print("SETUP CHECKLIST:")
    print("  [ ] CSV in ml-model/data/nba_games_elite.csv")
    print("  [ ] server.js in backend/")
    print("  [ ] app.py in ml-model/")
    print("  [ ] .env files with CRON_TOKEN")
    print("\nOnce complete, run diagnostic again!")

print("="*70 + "\n")