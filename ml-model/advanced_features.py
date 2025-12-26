import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Elite feature engineering with rankings + running momentum"""
    
    def __init__(self, games_df):
        """Initialize with raw game data"""
        self.games = games_df.sort_values('GAME_DATE_EST').reset_index(drop=True).copy()
        self.games['GAME_DATE_EST'] = pd.to_datetime(self.games['GAME_DATE_EST'])
        print(f"✓ Loaded {len(self.games)} games from {self.games['GAME_DATE_EST'].min()} to {self.games['GAME_DATE_EST'].max()}")
        print(f"✓ Available columns: {list(self.games.columns)}")
    
    def calculate_season_stats(self):
        """Calculate offensive/defensive stats per team per season"""
        print("\n📊 STEP 1: Calculating season-wide offensive/defensive stats...")
        
        season_stats = {}
        
        for season in self.games['SEASON'].unique():
            season_games = self.games[self.games['SEASON'] == season].copy()
            season_stats[season] = {}
            
            # Get all unique teams in this season
            all_teams = set(season_games['HOME_TEAM_ID'].unique()) | set(season_games['VISITOR_TEAM_ID'].unique())
            
            for team in all_teams:
                team_games_home = season_games[season_games['HOME_TEAM_ID'] == team]
                team_games_away = season_games[season_games['VISITOR_TEAM_ID'] == team]
                
                # Offensive stats (points scored)
                off_pts_home = team_games_home['PTS_home'].mean() if len(team_games_home) > 0 else 105
                off_pts_away = team_games_away['PTS_away'].mean() if len(team_games_away) > 0 else 105
                off_pts_avg = np.mean([off_pts_home, off_pts_away])
                
                # Defensive stats (points allowed)
                def_pts_home = team_games_home['PTS_away'].mean() if len(team_games_home) > 0 else 105
                def_pts_away = team_games_away['PTS_home'].mean() if len(team_games_away) > 0 else 105
                def_pts_avg = np.mean([def_pts_home, def_pts_away])
                
                season_stats[season][team] = {
                    'OFF_PTS': off_pts_avg,
                    'DEF_PTS': def_pts_avg,
                    'OFF_EFF': off_pts_avg / def_pts_avg if def_pts_avg > 0 else 1.0,
                    'GAMES': len(team_games_home) + len(team_games_away)
                }
        
        print(f"✓ Calculated stats for {len(all_teams)} teams across {len(season_stats)} seasons")
        return season_stats
    
    def calculate_season_rankings(self, season_stats):
        """Rank teams 1-30 (1=best, 30=worst) within each season"""
        print("\n🏆 STEP 2: Ranking teams within each season (1=best, 30=worst)...")
        
        rankings = {}
        
        for season in season_stats:
            teams_data = season_stats[season]
            
            # Sort by offensive points (descending: highest = rank 1)
            off_ranks = sorted(teams_data.items(), key=lambda x: x[1]['OFF_PTS'], reverse=True)
            off_rank_dict = {team: rank + 1 for rank, (team, _) in enumerate(off_ranks)}
            
            # Sort by defensive points (ascending: lowest = rank 1)
            def_ranks = sorted(teams_data.items(), key=lambda x: x[1]['DEF_PTS'])
            def_rank_dict = {team: rank + 1 for rank, (team, _) in enumerate(def_ranks)}
            
            rankings[season] = {
                'OFF': off_rank_dict,
                'DEF': def_rank_dict,
                'EFFICIENCY': {team: 1.0 / teams_data[team]['OFF_EFF'] if teams_data[team]['OFF_EFF'] > 0 else 1.0 for team in teams_data}
            }
        
        print(f"✓ Rankings created: {list(rankings.keys())}")
        return rankings
    
    def add_season_rankings(self, rankings):
        """Add season-based rankings to each game"""
        print("\n📍 STEP 3: Adding season rankings to games...")
        
        self.games['HOME_OFF_RANK'] = self.games.apply(
            lambda row: rankings[row['SEASON']]['OFF'].get(row['HOME_TEAM_ID'], 15), axis=1
        )
        self.games['HOME_DEF_RANK'] = self.games.apply(
            lambda row: rankings[row['SEASON']]['DEF'].get(row['HOME_TEAM_ID'], 15), axis=1
        )
        self.games['AWAY_OFF_RANK'] = self.games.apply(
            lambda row: rankings[row['SEASON']]['OFF'].get(row['VISITOR_TEAM_ID'], 15), axis=1
        )
        self.games['AWAY_DEF_RANK'] = self.games.apply(
            lambda row: rankings[row['SEASON']]['DEF'].get(row['VISITOR_TEAM_ID'], 15), axis=1
        )
        
        print(f"✓ Season rankings added to {len(self.games)} games")
    
    def calculate_running_ranks(self):
        """Running offensive/defensive ranks updated game-by-game"""
        print("\n⚡ STEP 4: Calculating running team ranks (momentum)...")
        
        self.games['HOME_RUNNING_OFF_RANK'] = 15
        self.games['HOME_RUNNING_DEF_RANK'] = 15
        self.games['AWAY_RUNNING_OFF_RANK'] = 15
        self.games['AWAY_RUNNING_DEF_RANK'] = 15
        
        # Track stats per team as we go through games
        team_running_stats = {}
        
        for idx, row in self.games.iterrows():
            home_team = row['HOME_TEAM_ID']
            away_team = row['VISITOR_TEAM_ID']
            
            # Initialize if first time seeing team
            if home_team not in team_running_stats:
                team_running_stats[home_team] = []
            if away_team not in team_running_stats:
                team_running_stats[away_team] = []
            
            # Get last 10 games for each team
            if len(team_running_stats[home_team]) >= 3:
                home_off_pts = np.mean([g['PTS_FOR'] for g in team_running_stats[home_team][-10:]])
                home_def_pts = np.mean([g['PTS_AGAINST'] for g in team_running_stats[home_team][-10:]])
            else:
                home_off_pts = row['PTS_home']
                home_def_pts = row['PTS_away']
            
            if len(team_running_stats[away_team]) >= 3:
                away_off_pts = np.mean([g['PTS_FOR'] for g in team_running_stats[away_team][-10:]])
                away_def_pts = np.mean([g['PTS_AGAINST'] for g in team_running_stats[away_team][-10:]])
            else:
                away_off_pts = row['PTS_away']
                away_def_pts = row['PTS_home']
            
            # Rank within active teams
            active_teams = list(team_running_stats.keys())
            if len(active_teams) >= 2:
                home_off_rank = 1 + sum([1 for t in active_teams if team_running_stats[t] and 
                                         np.mean([g['PTS_FOR'] for g in team_running_stats[t][-10:]]) > home_off_pts])
                home_def_rank = 1 + sum([1 for t in active_teams if team_running_stats[t] and 
                                         np.mean([g['PTS_AGAINST'] for g in team_running_stats[t][-10:]]) < home_def_pts])
            else:
                home_off_rank = 15
                home_def_rank = 15
            
            self.games.at[idx, 'HOME_RUNNING_OFF_RANK'] = home_off_rank
            self.games.at[idx, 'HOME_RUNNING_DEF_RANK'] = home_def_rank
            self.games.at[idx, 'AWAY_RUNNING_OFF_RANK'] = 31 - home_off_rank
            self.games.at[idx, 'AWAY_RUNNING_DEF_RANK'] = 31 - home_def_rank
            
            # Update running stats
            team_running_stats[home_team].append({
                'PTS_FOR': row['PTS_home'],
                'PTS_AGAINST': row['PTS_away']
            })
            team_running_stats[away_team].append({
                'PTS_FOR': row['PTS_away'],
                'PTS_AGAINST': row['PTS_home']
            })
        
        print(f"✓ Running ranks calculated for {len(self.games)} games")
    
    def add_rolling_stats(self):
        """5-game rolling averages (points, points allowed)"""
        print("\n📈 STEP 5: Adding rolling 5-game stats...")
        
        self.games['HOME_PTS_AVG_5'] = self.games.groupby('HOME_TEAM_ID')['PTS_home'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        self.games['HOME_PTS_ALLOWED_AVG_5'] = self.games.groupby('HOME_TEAM_ID')['PTS_away'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        
        self.games['AWAY_PTS_AVG_5'] = self.games.groupby('VISITOR_TEAM_ID')['PTS_away'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        self.games['AWAY_PTS_ALLOWED_AVG_5'] = self.games.groupby('VISITOR_TEAM_ID')['PTS_home'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        
        print(f"✓ Rolling stats added")
    
    def add_elite_features(self):
        """Master elite features"""
        print("\n💎 STEP 6: Creating elite derived features...")
        
        # 1. RANK DIFFERENTIALS (most important)
        self.games['OFF_RNK_DIFF'] = self.games['HOME_OFF_RANK'] - self.games['AWAY_OFF_RANK']
        self.games['DEF_RNK_DIFF'] = self.games['HOME_DEF_RANK'] - self.games['AWAY_DEF_RANK']
        
        # 2. POINTS DIFFERENTIAL
        self.games['PTS_AVG_DIFF'] = self.games['HOME_PTS_AVG_5'] - self.games['AWAY_PTS_AVG_5']
        self.games['DEF_AVG_DIFF'] = self.games['HOME_PTS_ALLOWED_AVG_5'] - self.games['AWAY_PTS_ALLOWED_AVG_5']
        
        # 3. MOMENTUM (running rank improvement)
        self.games['OFF_MOMENTUM'] = self.games['HOME_RUNNING_OFF_RANK'] - self.games['HOME_OFF_RANK']
        self.games['DEF_MOMENTUM'] = self.games['HOME_RUNNING_DEF_RANK'] - self.games['HOME_DEF_RANK']
        
        # 4. INTERACTION TERMS (capture complex relationships)
        self.games['RANK_INTERACTION'] = self.games['OFF_RNK_DIFF'] * self.games['DEF_RNK_DIFF']
        self.games['PTS_RANK_INTERACTION'] = self.games['PTS_AVG_DIFF'] * self.games['OFF_RNK_DIFF']
        
        # 5. HOME COURT
        self.games['HOME_COURT'] = 1
        
        # 6. GAME CONTEXT
        self.games['GAME_NUMBER'] = self.games.groupby('HOME_TEAM_ID').cumcount() + 1
        
        # 7. TARGET
        self.games['TARGET'] = (self.games['PTS_home'] > self.games['PTS_away']).astype(int)
        
        print(f"✓ Elite features created (12 total)")
    
    def engineer_all(self):
        """Master pipeline: Execute all steps"""
        print("\n" + "="*70)
        print("🏆 ELITE FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        # Step 1: Calculate season stats
        season_stats = self.calculate_season_stats()
        
        # Step 2: Calculate rankings
        rankings = self.calculate_season_rankings(season_stats)
        
        # Step 3: Add rankings to games
        self.add_season_rankings(rankings)
        
        # Step 4: Calculate running ranks
        self.calculate_running_ranks()
        
        # Step 5: Add rolling stats
        self.add_rolling_stats()
        
        # Step 6: Create elite features
        self.add_elite_features()
        
        # Final: Select features
        feature_columns = [
            'OFF_RNK_DIFF', 'DEF_RNK_DIFF',  # PRIMARY (45% + 35%)
            'PTS_AVG_DIFF', 'DEF_AVG_DIFF',  # ROLLING STATS
            'HOME_OFF_RANK', 'HOME_DEF_RANK', 'AWAY_OFF_RANK', 'AWAY_DEF_RANK',  # RANKINGS
            'HOME_RUNNING_OFF_RANK', 'HOME_RUNNING_DEF_RANK',  # MOMENTUM
            'OFF_MOMENTUM', 'DEF_MOMENTUM',  # MOMENTUM DELTA
            'RANK_INTERACTION', 'PTS_RANK_INTERACTION',  # INTERACTIONS
            'HOME_COURT', 'GAME_NUMBER',  # CONTEXT
            'TARGET'  # TARGET
        ]
        
        engineered = self.games[feature_columns].dropna()
        
        print("\n" + "="*70)
        print(f"✓ ELITE ENGINEERING COMPLETE")
        print(f"  • Total features: {len(feature_columns) - 1}")  # Exclude target
        print(f"  • Samples ready: {len(engineered)}")
        print(f"  • Target distribution: {engineered['TARGET'].value_counts().to_dict()}")
        print(f"  • Data quality: {100 * len(engineered) / len(self.games):.1f}%")
        print("="*70 + "\n")
        
        return engineered

# EXECUTION
if __name__ == '__main__':
    import sys
    
    # Load data
    games = pd.read_csv('../../nba-elite-prediction-platform/data/nba_games.csv')
    
    # Engineer
    engineer = AdvancedFeatureEngineer(games)
    elite_features = engineer.engineer_all()
    
    # Save
    output_path = 'data/nba_games_elite.csv'
    elite_features.to_csv(output_path, index=False)
    
    print(f"💾 Elite features saved to: {output_path}")
    print(f"\nFeature columns:")
    for i, col in enumerate(elite_features.columns, 1):
        print(f"  {i}. {col}")
