const express = require('express');
const gameModel = require('../models/gameModel');

const router = express.Router();

// ==================== HELPER FUNCTIONS ====================

/**
 * Extract latest team rankings from games
 * Prioritizes most recent game for each team
 * @param {Array} games - Game data from database
 * @returns {Object} Map of team_id -> ranking data
 */
function extractLatestRankings(games) {
  const rankingMap = {};
  
  // Iterate through games (assuming chronological order)
  games.forEach(game => {
    // Extract home team ranking
    if (game.HOME_TEAM_ID && game.HOME_OFF_RANK !== undefined) {
      rankingMap[game.HOME_TEAM_ID] = {
        team_id: game.HOME_TEAM_ID,
        team_name: game.HOME_TEAM_NAME || null,
        team_abbr: game.HOME_TEAM_ABBR || null,
        
        // Seasonal rankings
        off_rank: game.HOME_OFF_RANK,
        def_rank: game.HOME_DEF_RANK,
        
        // Running/momentum ranks
        running_off_rank: game.HOME_RUNNING_OFF_RANK || game.HOME_OFF_RANK,
        running_def_rank: game.HOME_RUNNING_DEF_RANK || game.HOME_DEF_RANK,
        
        // Performance metrics
        pts_avg: game.HOME_PTS_AVG || null,
        pts_allowed_avg: game.HOME_PTS_ALLOWED_AVG || null,
        
        // Momentum indicators
        off_momentum: game.OFF_MOMENTUM || 0,
        def_momentum: game.DEF_MOMENTUM || 0,
        
        // Metadata
        last_game_date: game.GAME_DATE,
        is_home: true
      };
    }
    
    // Extract away team ranking
    if (game.AWAY_TEAM_ID && game.AWAY_OFF_RANK !== undefined) {
      rankingMap[game.AWAY_TEAM_ID] = {
        team_id: game.AWAY_TEAM_ID,
        team_name: game.AWAY_TEAM_NAME || null,
        team_abbr: game.AWAY_TEAM_ABBR || null,
        
        // Seasonal rankings
        off_rank: game.AWAY_OFF_RANK,
        def_rank: game.AWAY_DEF_RANK,
        
        // Running/momentum ranks
        running_off_rank: game.AWAY_RUNNING_OFF_RANK || game.AWAY_OFF_RANK,
        running_def_rank: game.AWAY_RUNNING_DEF_RANK || game.AWAY_DEF_RANK,
        
        // Performance metrics
        pts_avg: game.AWAY_PTS_AVG || null,
        pts_allowed_avg: game.AWAY_PTS_ALLOWED_AVG || null,
        
        // Momentum indicators
        off_momentum: game.OFF_MOMENTUM || 0,
        def_momentum: game.DEF_MOMENTUM || 0,
        
        // Metadata
        last_game_date: game.GAME_DATE,
        is_home: false
      };
    }
  });
  
  return rankingMap;
}

/**
 * Calculate ranking tiers (elite, strong, average, weak)
 * @param {number} rank - Team ranking (1-30)
 * @returns {string} Tier classification
 */
function getTier(rank) {
  if (rank <= 8) return 'Elite';
  if (rank <= 15) return 'Strong';
  if (rank <= 22) return 'Average';
  return 'Weak';
}

/**
 * Calculate momentum direction
 * @param {number} momentum - Momentum value (negative = improving, positive = declining)
 * @returns {string} Momentum direction
 */
function getMomentumDirection(momentum) {
  if (momentum < -2) return 'Improving';
  if (momentum > 2) return 'Declining';
  return 'Stable';
}

// ==================== ROUTES ====================

/**
 * GET /api/rankings
 * Get all team rankings sorted by offensive ranking
 * 
 * Query Parameters:
 *   - sort (default: 'off_rank') - 'off_rank', 'def_rank', 'team_name'
 *   - order (default: 'asc') - 'asc' or 'desc'
 *   - limit (default: 30) - Maximum teams to return
 *   - metric (default: 'seasonal') - 'seasonal' or 'running'
 * 
 * Returns array of teams with:
 *   - Team identifiers (id, name, abbreviation)
 *   - Offensive and defensive rankings
 *   - Performance metrics
 *   - Momentum indicators
 *   - Tier classification
 */
router.get('/', (req, res) => {
  try {
    const {
      sort = 'off_rank',
      order = 'asc',
      limit = 30,
      metric = 'seasonal'
    } = req.query;
    
    // Validate parameters
    const validSorts = ['off_rank', 'def_rank', 'team_name', 'running_off_rank', 'running_def_rank'];
    const validOrders = ['asc', 'desc'];
    const validMetrics = ['seasonal', 'running'];
    
    if (!validSorts.includes(sort)) {
      return res.status(400).json({
        success: false,
        error: `Invalid sort parameter. Must be one of: ${validSorts.join(', ')}`
      });
    }
    
    if (!validOrders.includes(order)) {
      return res.status(400).json({
        success: false,
        error: `Invalid order parameter. Must be 'asc' or 'desc'`
      });
    }
    
    if (!validMetrics.includes(metric)) {
      return res.status(400).json({
        success: false,
        error: `Invalid metric parameter. Must be 'seasonal' or 'running'`
      });
    }
    
    const games = gameModel.getAllGames();
    
    if (!games || games.length === 0) {
      return res.json({
        success: true,
        count: 0,
        message: 'No games found',
        data: []
      });
    }
    
    // Extract latest rankings
    const rankingMap = extractLatestRankings(games);
    let rankings = Object.values(rankingMap);
    
    // Add tier classification
    rankings = rankings.map(team => ({
      ...team,
      off_tier: getTier(team.off_rank),
      def_tier: getTier(team.def_rank),
      off_momentum_direction: getMomentumDirection(team.off_momentum),
      def_momentum_direction: getMomentumDirection(team.def_momentum)
    }));
    
    // Sort rankings
    rankings.sort((a, b) => {
      let aVal = a[sort];
      let bVal = b[sort];
      
      // Handle string sorting
      if (typeof aVal === 'string') {
        aVal = aVal.toLowerCase();
        bVal = bVal.toLowerCase();
      }
      
      const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      return order === 'asc' ? comparison : -comparison;
    });
    
    // Apply limit
    const limited = rankings.slice(0, parseInt(limit));
    
    res.json({
      success: true,
      count: limited.length,
      total_teams: rankings.length,
      sort_by: sort,
      sort_order: order,
      metric: metric,
      data: limited
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * GET /api/rankings/team/:team_id
 * Get detailed rankings for a specific team
 * 
 * Returns:
 *   - Team info (id, name, abbreviation)
 *   - Current seasonal and running rankings
 *   - Performance metrics
 *   - Momentum analysis
 *   - Tier classification
 *   - Recent games
 */
router.get('/team/:team_id', (req, res) => {
  try {
    const { team_id } = req.params;
    const games = gameModel.getAllGames();
    
    // Find all games for this team
    const teamGames = games.filter(
      g => g.HOME_TEAM_ID == team_id || g.AWAY_TEAM_ID == team_id
    );
    
    if (teamGames.length === 0) {
      return res.status(404).json({
        success: false,
        error: `No games found for team ${team_id}`
      });
    }
    
    // Get most recent game data (contains latest rankings)
    const latestGame = teamGames[teamGames.length - 1];
    const isHome = latestGame.HOME_TEAM_ID == team_id;
    
    const teamData = isHome ? {
      team_id: latestGame.HOME_TEAM_ID,
      team_name: latestGame.HOME_TEAM_NAME,
      team_abbr: latestGame.HOME_TEAM_ABBR,
      is_home: true,
      off_rank: latestGame.HOME_OFF_RANK,
      def_rank: latestGame.HOME_DEF_RANK,
      running_off_rank: latestGame.HOME_RUNNING_OFF_RANK,
      running_def_rank: latestGame.HOME_RUNNING_DEF_RANK,
      pts_avg: latestGame.HOME_PTS_AVG,
      pts_allowed_avg: latestGame.HOME_PTS_ALLOWED_AVG,
      off_momentum: latestGame.OFF_MOMENTUM,
      def_momentum: latestGame.DEF_MOMENTUM
    } : {
      team_id: latestGame.AWAY_TEAM_ID,
      team_name: latestGame.AWAY_TEAM_NAME,
      team_abbr: latestGame.AWAY_TEAM_ABBR,
      is_home: false,
      off_rank: latestGame.AWAY_OFF_RANK,
      def_rank: latestGame.AWAY_DEF_RANK,
      running_off_rank: latestGame.AWAY_RUNNING_OFF_RANK,
      running_def_rank: latestGame.AWAY_RUNNING_DEF_RANK,
      pts_avg: latestGame.AWAY_PTS_AVG,
      pts_allowed_avg: latestGame.AWAY_PTS_ALLOWED_AVG,
      off_momentum: latestGame.OFF_MOMENTUM,
      def_momentum: latestGame.DEF_MOMENTUM
    };
    
    // Calculate recent trend (last 5 games)
    const recentGames = teamGames.slice(-5);
    const avgOffRank = (
      recentGames.reduce((sum, g) => {
        const rank = isHome ? g.HOME_OFF_RANK : g.AWAY_OFF_RANK;
        return sum + (rank || 0);
      }, 0) / recentGames.length
    ).toFixed(1);
    
    const avgDefRank = (
      recentGames.reduce((sum, g) => {
        const rank = isHome ? g.HOME_DEF_RANK : g.AWAY_DEF_RANK;
        return sum + (rank || 0);
      }, 0) / recentGames.length
    ).toFixed(1);
    
    res.json({
      success: true,
      team: teamData,
      rankings: {
        offensive: {
          seasonal: teamData.off_rank,
          running: teamData.running_off_rank,
          momentum: teamData.off_momentum,
          momentum_direction: getMomentumDirection(teamData.off_momentum),
          tier: getTier(teamData.off_rank),
          recent_avg: parseFloat(avgOffRank)
        },
        defensive: {
          seasonal: teamData.def_rank,
          running: teamData.running_def_rank,
          momentum: teamData.def_momentum,
          momentum_direction: getMomentumDirection(teamData.def_momentum),
          tier: getTier(teamData.def_rank),
          recent_avg: parseFloat(avgDefRank)
        }
      },
      performance: {
        pts_avg: teamData.pts_avg,
        pts_allowed_avg: teamData.pts_allowed_avg
      },
      recent_games: recentGames.length,
      last_update: latestGame.GAME_DATE
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * GET /api/rankings/tier/:tier
 * Get all teams in a specific tier (Elite, Strong, Average, Weak)
 * 
 * Tier Definitions:
 *   - Elite: Rank 1-8
 *   - Strong: Rank 9-15
 *   - Average: Rank 16-22
 *   - Weak: Rank 23-30
 */
router.get('/tier/:tier', (req, res) => {
  try {
    const { tier } = req.params;
    const validTiers = ['Elite', 'Strong', 'Average', 'Weak'];
    
    if (!validTiers.includes(tier)) {
      return res.status(400).json({
        success: false,
        error: `Invalid tier. Must be one of: ${validTiers.join(', ')}`
      });
    }
    
    const games = gameModel.getAllGames();
    const rankingMap = extractLatestRankings(games);
    
    let rankings = Object.values(rankingMap).filter(team => {
      return getTier(team.off_rank) === tier;
    });
    
    rankings = rankings.map(team => ({
      ...team,
      off_tier: getTier(team.off_rank),
      def_tier: getTier(team.def_rank)
    }));
    
    rankings.sort((a, b) => a.off_rank - b.off_rank);
    
    res.json({
      success: true,
      tier,
      count: rankings.length,
      data: rankings
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * GET /api/rankings/momentum
 * Get teams sorted by momentum (improving vs declining)
 * 
 * Query Parameters:
 *   - direction (default: 'improving') - 'improving', 'declining', 'stable'
 *   - type (default: 'offensive') - 'offensive', 'defensive', 'both'
 */
router.get('/momentum', (req, res) => {
  try {
    const { direction = 'improving', type = 'offensive' } = req.query;
    
    const validDirections = ['improving', 'declining', 'stable'];
    const validTypes = ['offensive', 'defensive', 'both'];
    
    if (!validDirections.includes(direction)) {
      return res.status(400).json({
        success: false,
        error: `Invalid direction. Must be one of: ${validDirections.join(', ')}`
      });
    }
    
    if (!validTypes.includes(type)) {
      return res.status(400).json({
        success: false,
        error: `Invalid type. Must be one of: ${validTypes.join(', ')}`
      });
    }
    
    const games = gameModel.getAllGames();
    const rankingMap = extractLatestRankings(games);
    let rankings = Object.values(rankingMap);
    
    // Filter by momentum
    const directionMap = {
      'improving': (m) => m < -2,
      'declining': (m) => m > 2,
      'stable': (m) => m >= -2 && m <= 2
    };
    
    const isMatchingMomentum = directionMap[direction];
    
    rankings = rankings.filter(team => {
      if (type === 'offensive') return isMatchingMomentum(team.off_momentum);
      if (type === 'defensive') return isMatchingMomentum(team.def_momentum);
      return isMatchingMomentum(team.off_momentum) || isMatchingMomentum(team.def_momentum);
    });
    
    // Add momentum direction
    rankings = rankings.map(team => ({
      ...team,
      off_momentum_direction: getMomentumDirection(team.off_momentum),
      def_momentum_direction: getMomentumDirection(team.def_momentum)
    }));
    
    res.json({
      success: true,
      filter: {
        direction,
        type
      },
      count: rankings.length,
      data: rankings
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// ==================== ERROR HANDLERS ====================

/**
 * Handle 404 for this router
 */
router.use((req, res) => {
  res.status(404).json({
    success: false,
    error: 'Rankings endpoint not found',
    path: req.path,
    method: req.method
  });
});

module.exports = router;