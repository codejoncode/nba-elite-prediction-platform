#!/usr/bin/env python3
"""
System Validation Script
Tests all components of the NBA prediction system
"""

import json
import pickle
import requests
from pathlib import Path
import sys

def test_model_files():
    """Test if model files exist and are loadable"""
    print("\n" + "="*70)
    print("1. TESTING MODEL FILES")
    print("="*70)
    
    models_dir = Path('models')
    required_files = ['xgboost_model.pkl']
    optional_files = ['scaler.pkl', 'feature_columns.json', 'metrics.json']
    
    all_good = True
    
    for file in required_files:
        file_path = models_dir / file
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    pickle.load(f)
                print(f"  ✓ {file} - OK")
            except Exception as e:
                print(f"  ✗ {file} - ERROR: {e}")
                all_good = False
        else:
            print(f"  ✗ {file} - NOT FOUND")
            all_good = False
    
    for file in optional_files:
        file_path = models_dir / file
        if file_path.exists():
            print(f"  ✓ {file} - OK (optional)")
        else:
            print(f"  ⚠ {file} - NOT FOUND (optional, but recommended)")
    
    return all_good

def test_data_directory():
    """Test if data directory is writable"""
    print("\n" + "="*70)
    print("2. TESTING DATA DIRECTORY")
    print("="*70)
    
    data_dir = Path('data')
    
    if not data_dir.exists():
        print(f"  ⚠ Data directory doesn't exist, creating...")
        data_dir.mkdir(exist_ok=True)
    
    # Test write permissions
    test_file = data_dir / '.test_write'
    try:
        test_file.write_text('test')
        test_file.unlink()
        print(f"  ✓ Data directory writable")
        return True
    except Exception as e:
        print(f"  ✗ Cannot write to data directory: {e}")
        return False

def test_espn_api():
    """Test ESPN API connectivity"""
    print("\n" + "="*70)
    print("3. TESTING ESPN API")
    print("="*70)
    
    url = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            events = data.get('events', [])
            print(f"  ✓ ESPN API accessible")
            print(f"  ✓ Found {len(events)} games in current response")
            return True
        else:
            print(f"  ✗ ESPN API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Cannot connect to ESPN API: {e}")
        return False

def test_flask_api():
    """Test Flask API endpoints"""
    print("\n" + "="*70)
    print("4. TESTING FLASK API")
    print("="*70)
    
    base_url = 'http://localhost:5001'
    
    # Test health endpoint
    try:
        response = requests.get(f'{base_url}/health', timeout=5)
        if response.status_code == 200:
            print(f"  ✓ Health endpoint OK")
            data = response.json()
            print(f"    - Model loaded: {data.get('model_loaded')}")
        else:
            print(f"  ✗ Health endpoint returned {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Cannot connect to Flask API: {e}")
        print(f"  ℹ Make sure Flask is running: python app.py")
        return False
    
    # Test game_results endpoint
    try:
        response = requests.get(f'{base_url}/api/game_results', timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Game results endpoint OK")
            print(f"    - Total games: {data['metadata']['total_games_tracked']}")
            print(f"    - Accuracy: {data['metadata']['accuracy_all_time_percent']}%")
            print(f"    - Recent results: {len(data['recent_results'])}")
            print(f"    - Upcoming games: {len(data['upcoming_games'])}")
            return True
        else:
            print(f"  ✗ Game results endpoint returned {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Game results endpoint error: {e}")
        return False

def test_predictions_database():
    """Test predictions database"""
    print("\n" + "="*70)
    print("5. TESTING PREDICTIONS DATABASE")
    print("="*70)
    
    db_path = Path('data/predictions_history.json')
    
    if not db_path.exists():
        print(f"  ⚠ Predictions database doesn't exist yet")
        print(f"  ℹ It will be created when you run the Flask API")
        return True
    
    try:
        with open(db_path, 'r') as f:
            db = json.load(f)
        
        predictions = db.get('predictions', [])
        verified = [p for p in predictions if 'is_correct' in p]
        upcoming = [p for p in predictions if 'is_correct' not in p]
        correct = sum(1 for p in verified if p['is_correct'])
        
        print(f"  ✓ Database loaded")
        print(f"    - Total predictions: {len(predictions)}")
        print(f"    - Verified results: {len(verified)}")
        print(f"    - Correct predictions: {correct}")
        print(f"    - Pending predictions: {len(upcoming)}")
        
        if len(verified) > 0:
            accuracy = (correct / len(verified)) * 100
            print(f"    - Accuracy: {accuracy:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading database: {e}")
        return False

def test_team_stats():
    """Test team statistics database"""
    print("\n" + "="*70)
    print("6. TESTING TEAM STATISTICS")
    print("="*70)
    
    stats_path = Path('data/team_stats.json')
    
    if not stats_path.exists():
        print(f"  ⚠ Team stats don't exist yet")
        print(f"  ℹ Run: python init_team_stats.py")
        return False
    
    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        teams_with_data = {k: v for k, v in stats.items() if v.get('games_played', 0) > 0}
        
        print(f"  ✓ Team stats loaded")
        print(f"    - Total teams: {len(stats)}")
        print(f"    - Teams with data: {len(teams_with_data)}")
        
        if teams_with_data:
            # Show top 3 offensive teams
            sorted_off = sorted(teams_with_data.items(), key=lambda x: x[1]['off_rank'])[:3]
            print(f"    - Top offensive teams:")
            for abbr, data in sorted_off:
                print(f"      {abbr}: {data['pts_avg']:.1f} PPG (Rank {data['off_rank']})")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading team stats: {e}")
        return False

def print_summary(results):
    """Print test summary"""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total = len(results)
    passed = sum(results.values())
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    print("-"*70)
    print(f"  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED! System is ready.")
        print("\nNext steps:")
        print("  1. Start Flask API: python app.py")
        print("  2. Start scheduler: python scheduler.py")
        print("  3. Start React frontend: cd frontend && npm start")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("\nRecommended fixes:")
        if not results.get('Model Files'):
            print("  • Train or copy model files to models/ directory")
        if not results.get('Team Statistics'):
            print("  • Run: python init_team_stats.py")
        if not results.get('Flask API'):
            print("  • Start Flask: python app.py")
    
    print("\n" + "="*70)

def main():
    print("\n" + "="*70)
    print("NBA ELITE PREDICTION SYSTEM - VALIDATION")
    print("="*70)
    
    results = {
        'Model Files': test_model_files(),
        'Data Directory': test_data_directory(),
        'ESPN API': test_espn_api(),
        'Team Statistics': test_team_stats(),
        'Predictions Database': test_predictions_database(),
        'Flask API': test_flask_api(),
    }
    
    print_summary(results)
    
    # Return exit code
    return 0 if all(results.values()) else 1

if __name__ == '__main__':
    sys.exit(main())