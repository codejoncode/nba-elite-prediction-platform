#!/usr/bin/env python3
"""
Initialize team statistics database with current NBA season data
Run this once to bootstrap your predictions system
"""

import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / 'data'
TEAM_STATS_FILE = DATA_DIR / 'team_stats.json'
ESPN_SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'

TEAM_MAPPING = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

def fetch_historical_games(days_back=30):
    """Fetch completed games from the last N days"""
    logger.info(f"Fetching games from last {days_back} days...")
    
    all_games = []
    
    for days_ago in range(days_back):
        date = (datetime.utcnow() - timedelta(days=days_ago)).strftime('%Y%m%d')
        url = f"{ESPN_SCOREBOARD}?dates={date}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', [])
                
                for event in events:
                    try:
                        status = event.get('status', {}).get('type', {}).get('name', '')
                        if status != 'STATUS_FINAL':
                            continue
                        
                        comp = event.get('competitions', [{}])[0]
                        competitors = comp.get('competitors', [])
                        
                        if len(competitors) < 2:
                            continue
                        
                        home = next((c for c in competitors if c.get('homeAway') == 'home'), competitors[0])
                        away = next((c for c in competitors if c.get('homeAway') == 'away'), competitors[1])
                        
                        game = {
                            'date': event.get('date', ''),
                            'home_team': home.get('team', {}).get('displayName', ''),
                            'away_team': away.get('team', {}).get('displayName', ''),
                            'home_score': int(home.get('score', 0) or 0),
                            'away_score': int(away.get('score', 0) or 0)
                        }
                        
                        all_games.append(game)
                        
                    except Exception as e:
                        logger.debug(f"Error parsing game: {e}")
                        continue
        
        except Exception as e:
            logger.warning(f"Error fetching date {date}: {e}")
            continue
    
    logger.info(f"✓ Fetched {len(all_games)} completed games")
    return all_games

def initialize_team_stats(games):
    """Calculate initial team statistics from historical games"""
    logger.info("Calculating team statistics...")
    
    team_stats = {}
    
    # Initialize all teams
    for team_name, abbr in TEAM_MAPPING.items():
        team_stats[abbr] = {
            'full_name': team_name,
            'games_played': 0,
            'pts_avg': 110.0,
            'pts_allowed_avg': 110.0,
            'off_rank': 15,
            'def_rank': 15,
            'running_off_rank': 15,
            'running_def_rank': 15,
            'off_momentum': 0,
            'def_momentum': 0,
            'recent_scores': [],
            'recent_allowed': [],
            'wins': 0,
            'losses': 0
        }
    
    # Process games chronologically
    sorted_games = sorted(games, key=lambda x: x['date'])
    
    for game in sorted_games:
        home_team = game['home_team']
        away_team = game['away_team']
        home_score = game['home_score']
        away_score = game['away_score']
        
        home_abbr = TEAM_MAPPING.get(home_team)
        away_abbr = TEAM_MAPPING.get(away_team)
        
        if not home_abbr or not away_abbr:
            continue
        
        # Update home team
        home_stats = team_stats[home_abbr]
        home_stats['recent_scores'].append(home_score)
        home_stats['recent_allowed'].append(away_score)
        home_stats['recent_scores'] = home_stats['recent_scores'][-10:]
        home_stats['recent_allowed'] = home_stats['recent_allowed'][-10:]
        home_stats['games_played'] += 1
        
        if len(home_stats['recent_scores']) >= 3:
            home_stats['pts_avg'] = sum(home_stats['recent_scores']) / len(home_stats['recent_scores'])
            home_stats['pts_allowed_avg'] = sum(home_stats['recent_allowed']) / len(home_stats['recent_allowed'])
        
        if home_score > away_score:
            home_stats['wins'] += 1
        else:
            home_stats['losses'] += 1
        
        # Update away team
        away_stats = team_stats[away_abbr]
        away_stats['recent_scores'].append(away_score)
        away_stats['recent_allowed'].append(home_score)
        away_stats['recent_scores'] = away_stats['recent_scores'][-10:]
        away_stats['recent_allowed'] = away_stats['recent_allowed'][-10:]
        away_stats['games_played'] += 1
        
        if len(away_stats['recent_scores']) >= 3:
            away_stats['pts_avg'] = sum(away_stats['recent_scores']) / len(away_stats['recent_scores'])
            away_stats['pts_allowed_avg'] = sum(away_stats['recent_allowed']) / len(away_stats['recent_allowed'])
        
        if away_score > home_score:
            away_stats['wins'] += 1
        else:
            away_stats['losses'] += 1
    
    # Calculate rankings
    teams_with_data = [(abbr, stats) for abbr, stats in team_stats.items() if stats['games_played'] > 0]
    
    # Offensive rankings (higher PPG = better rank)
    sorted_off = sorted(teams_with_data, key=lambda x: x[1]['pts_avg'], reverse=True)
    for rank, (abbr, _) in enumerate(sorted_off, 1):
        team_stats[abbr]['off_rank'] = rank
        team_stats[abbr]['running_off_rank'] = rank
    
    # Defensive rankings (lower allowed = better rank)
    sorted_def = sorted(teams_with_data, key=lambda x: x[1]['pts_allowed_avg'])
    for rank, (abbr, _) in enumerate(sorted_def, 1):
        team_stats[abbr]['def_rank'] = rank
        team_stats[abbr]['running_def_rank'] = rank
    
    logger.info(f"✓ Calculated stats for {len(teams_with_data)} teams")
    
    return team_stats

def save_team_stats(team_stats):
    """Save team statistics to file"""
    DATA_DIR.mkdir(exist_ok=True)
    
    with open(TEAM_STATS_FILE, 'w') as f:
        json.dump(team_stats, f, indent=2)
    
    logger.info(f"✓ Saved team stats to {TEAM_STATS_FILE}")

def print_summary(team_stats):
    """Print summary of team statistics"""
    print("\n" + "="*70)
    print("TEAM STATISTICS SUMMARY")
    print("="*70)
    
    teams_with_data = [(abbr, stats) for abbr, stats in team_stats.items() if stats['games_played'] > 0]
    teams_with_data.sort(key=lambda x: x[1]['off_rank'])
    
    print(f"\n{'Rank':<6} {'Team':<25} {'Record':<10} {'PPG':<8} {'PAPG':<8} {'Off':<5} {'Def':<5}")
    print("-"*70)
    
    for i, (abbr, stats) in enumerate(teams_with_data[:10], 1):
        record = f"{stats['wins']}-{stats['losses']}"
        ppg = f"{stats['pts_avg']:.1f}"
        papg = f"{stats['pts_allowed_avg']:.1f}"
        
        print(f"{i:<6} {stats['full_name']:<25} {record:<10} {ppg:<8} {papg:<8} {stats['off_rank']:<5} {stats['def_rank']:<5}")
    
    print("-"*70)
    print(f"Total teams with data: {len(teams_with_data)}")
    print("="*70 + "\n")

def main():
    print("\n" + "="*70)
    print("NBA TEAM STATISTICS INITIALIZATION")
    print("="*70 + "\n")
    
    # Fetch historical games
    games = fetch_historical_games(days_back=30)
    
    if not games:
        logger.error("No games fetched. Check your internet connection or ESPN API availability.")
        return
    
    # Calculate team stats
    team_stats = initialize_team_stats(games)
    
    # Save to file
    save_team_stats(team_stats)
    
    # Print summary
    print_summary(team_stats)
    
    print("✅ Team statistics initialized successfully!")
    print(f"📁 File saved: {TEAM_STATS_FILE}")
    print("\nYou can now start the Flask API:")
    print("  python app.py")
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    main()