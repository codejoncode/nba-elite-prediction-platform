const fs = require('fs');
const path = require('path');

let gameCache = null;

/**
 * Load NBA games CSV with elite features
 * Expects: data/nba_games_elite.csv
 */
function loadGames() {
  if (gameCache) return gameCache;
  
  try {
    const csvPath = path.join(__dirname, '../../data/nba_games_elite.csv');
    const csv = fs.readFileSync(csvPath, 'utf8');
    const lines = csv.split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    gameCache = lines.slice(1).map(line => {
      if (!line.trim()) return null;
      
      const values = line.split(',').map(v => v.trim());
      const game = {};
      
      headers.forEach((h, i) => {
        const val = values[i];
        // Parse numbers
        if (val && !isNaN(val) && val !== '') {
          game[h] = parseFloat(val);
        } else {
          game[h] = val;
        }
      });
      
      return game;
    }).filter(g => g !== null && g.GAME_DATE);
    
    console.log(`✓ Loaded ${gameCache.length} games`);
    return gameCache;
  } catch (error) {
    console.error('Error loading games:', error);
    return [];
  }
}

module.exports = {
  /**
   * Get all games
   */
  getAllGames: () => {
    return loadGames();
  },
  
  /**
   * Get games by date range
   */
  getGames: (startDate, endDate) => {
    const games = loadGames();
    if (!startDate || !endDate) return games;
    
    const start = new Date(startDate);
    const end = new Date(endDate);
    
    return games.filter(g => {
      const gameDate = new Date(g.GAME_DATE);
      return gameDate >= start && gameDate <= end;
    });
  },
  
  /**
   * Get games by team
   */
  getTeamGames: (teamId) => {
    const games = loadGames();
    return games.filter(g => g.HOME_TEAM_ID == teamId || g.AWAY_TEAM_ID == teamId);
  },
  
  /**
   * Get recent games (last N)
   */
  getRecentGames: (n = 10) => {
    const games = loadGames();
    return games.slice(-n);
  }
};