const express = require('express');
const axios = require('axios');
const gameModel = require('../models/gameModel');

const router = express.Router();
const FLASK_URL = process.env.FLASK_BACKEND_URL || 'http://localhost:5001';

// ==================== REQUIRED FEATURES ====================
// All 16 features that the elite model requires
const REQUIRED_FEATURES = [
  'OFF_RNK_DIFF',           // Home off rank - Away off rank
  'DEF_RNK_DIFF',           // Home def rank - Away def rank
  'PTS_AVG_DIFF',           // Home 5-game PTS avg - Away 5-game PTS avg
  'DEF_AVG_DIFF',           // Home 5-game PTS allowed avg - Away 5-game PTS allowed avg
  'HOME_OFF_RANK',          // Home offensive rank (1-30)
  'HOME_DEF_RANK',          // Home defensive rank (1-30)
  'AWAY_OFF_RANK',          // Away offensive rank (1-30)
  'AWAY_DEF_RANK',          // Away defensive rank (1-30)
  'HOME_RUNNING_OFF_RANK',  // Home current running offensive rank
  'HOME_RUNNING_DEF_RANK',  // Home current running defensive rank
  'OFF_MOMENTUM',           // Change in offensive rank
  'DEF_MOMENTUM',           // Change in defensive rank
  'RANK_INTERACTION',       // OFF_RNK_DIFF * DEF_RNK_DIFF
  'PTS_RANK_INTERACTION',   // PTS_AVG_DIFF * OFF_RNK_DIFF
  'HOME_COURT',             // Always 1 (home advantage)
  'GAME_NUMBER'             // Cumulative game number for home team
];

// ==================== HELPER FUNCTIONS ====================

/**
 * Validate that ranking data has all required features
 * @param {Object} rankingData - Game ranking features
 * @returns {Object} { isValid: boolean, missingFeatures: array }
 */
function validateRankingData(rankingData) {
  const missingFeatures = REQUIRED_FEATURES.filter(
    feature => !(feature in rankingData)
  );
  
  return {
    isValid: missingFeatures.length === 0,
    missingFeatures
  };
}

/**
 * Build ranking data from game object with proper defaults
 * Ensures all 16 features are present
 * @param {Object} game - Game data from database
 * @returns {Object} Complete ranking data with all features
 */
function buildRankingData(game) {
  return {
    // Ranking differentials (offensive and defensive)
    OFF_RNK_DIFF: game.OFF_RNK_DIFF !== undefined ? game.OFF_RNK_DIFF : 0,
    DEF_RNK_DIFF: game.DEF_RNK_DIFF !== undefined ? game.DEF_RNK_DIFF : 0,
    
    // 5-game rolling averages
    PTS_AVG_DIFF: game.PTS_AVG_DIFF !== undefined ? game.PTS_AVG_DIFF : 0,
    DEF_AVG_DIFF: game.DEF_AVG_DIFF !== undefined ? game.DEF_AVG_DIFF : 0,
    
    // Home team seasonal rankings (1-30)
    HOME_OFF_RANK: game.HOME_OFF_RANK !== undefined ? game.HOME_OFF_RANK : 15,
    HOME_DEF_RANK: game.HOME_DEF_RANK !== undefined ? game.HOME_DEF_RANK : 15,
    
    // Away team seasonal rankings (1-30)
    AWAY_OFF_RANK: game.AWAY_OFF_RANK !== undefined ? game.AWAY_OFF_RANK : 15,
    AWAY_DEF_RANK: game.AWAY_DEF_RANK !== undefined ? game.AWAY_DEF_RANK : 15,
    
    // Running ranks (current momentum)
    HOME_RUNNING_OFF_RANK: game.HOME_RUNNING_OFF_RANK !== undefined ? game.HOME_RUNNING_OFF_RANK : 15,
    HOME_RUNNING_DEF_RANK: game.HOME_RUNNING_DEF_RANK !== undefined ? game.HOME_RUNNING_DEF_RANK : 15,
    
    // Momentum indicators (change in ranking)
    OFF_MOMENTUM: game.OFF_MOMENTUM !== undefined ? game.OFF_MOMENTUM : 0,
    DEF_MOMENTUM: game.DEF_MOMENTUM !== undefined ? game.DEF_MOMENTUM : 0,
    
    // Interaction terms (multiplicative relationships)
    RANK_INTERACTION: game.RANK_INTERACTION !== undefined ? game.RANK_INTERACTION : 0,
    PTS_RANK_INTERACTION: game.PTS_RANK_INTERACTION !== undefined ? game.PTS_RANK_INTERACTION : 0,
    
    // Home court advantage (always 1)
    HOME_COURT: 1,
    
    // Game counter
    GAME_NUMBER: game.GAME_NUMBER !== undefined ? game.GAME_NUMBER : 10
  };
}

/**
 * Transform Flask prediction response to API response format
 * @param {Object} game - Game data
 * @param {Object} flaskPrediction - Response from Flask /predict endpoint
 * @returns {Object} Formatted prediction response
 */
function formatPredictionResponse(game, flaskPrediction) {
  return {
    game_id: game.GAME_ID || null,
    game_date: game.GAME_DATE || null,
    home_team_id: game.HOME_TEAM_ID || null,
    away_team_id: game.AWAY_TEAM_ID || null,
    home_team_name: game.HOME_TEAM_NAME || `Team_${game.HOME_TEAM_ID}`,
    away_team_name: game.AWAY_TEAM_NAME || `Team_${game.AWAY_TEAM_ID}`,
    
    // Team rankings for context
    home_off_rank: game.HOME_OFF_RANK || null,
    home_def_rank: game.HOME_DEF_RANK || null,
    away_off_rank: game.AWAY_OFF_RANK || null,
    away_def_rank: game.AWAY_DEF_RANK || null,
    
    // Feature differentials
    off_rank_diff: game.OFF_RNK_DIFF || 0,
    def_rank_diff: game.DEF_RNK_DIFF || 0,
    pts_avg_diff: game.PTS_AVG_DIFF || 0,
    def_avg_diff: game.DEF_AVG_DIFF || 0,
    
    // Model predictions
    home_win_probability: flaskPrediction.home_win_probability,
    away_win_probability: flaskPrediction.away_win_probability,
    predicted_winner: flaskPrediction.predicted_winner,
    confidence: flaskPrediction.confidence,
    confidence_pct: flaskPrediction.confidence_pct,
    
    // Feature importance
    top_impacting_features: flaskPrediction.top_impacting_features,
    
    // Model metadata
    model_performance: flaskPrediction.model_performance
  };
}

// ==================== ROUTES ====================

/**
 * GET /api/predictions/health
 * Check Flask backend health
 */
router.get('/health', async (req, res) => {
  try {
    const flaskHealth = await axios.get(`${FLASK_URL}/health`);
    
    res.json({
      success: true,
      backend_status: flaskHealth.data.status,
      model_loaded: flaskHealth.data.model_loaded,
      api_url: FLASK_URL,
      required_features: REQUIRED_FEATURES.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Flask health check failed:', error.message);
    res.status(503).json({
      success: false,
      error: 'Flask backend unavailable',
      backend_url: FLASK_URL,
      details: error.message
    });
  }
});

/**
 * GET /api/predictions/features
 * Get list of required features and descriptions
 */
router.get('/features', async (req, res) => {
  try {
    const flaskFeatures = await axios.get(`${FLASK_URL}/features`);
    
    res.json({
      success: true,
      feature_count: REQUIRED_FEATURES.length,
      features: REQUIRED_FEATURES,
      descriptions: flaskFeatures.data.feature_descriptions,
      note: 'All 16 features must be present in ranking_data for predictions'
    });
  } catch (error) {
    console.error('Features endpoint error:', error.message);
    res.json({
      success: true,
      feature_count: REQUIRED_FEATURES.length,
      features: REQUIRED_FEATURES,
      note: 'All 16 features must be present in ranking_data for predictions'
    });
  }
});

/**
 * GET /api/predictions/metrics
 * Get model performance metrics
 */
router.get('/metrics', async (req, res) => {
  try {
    const flaskMetrics = await axios.get(`${FLASK_URL}/metrics`);
    
    res.json({
      success: true,
      metrics: flaskMetrics.data
    });
  } catch (error) {
    console.error('Metrics endpoint error:', error.message);
    res.status(503).json({
      success: false,
      error: 'Could not fetch metrics from Flask backend',
      details: error.message
    });
  }
});

/**
 * GET /api/predictions/upcoming
 * Get predictions for upcoming/recent games using elite rankings
 * 
 * Query Parameters:
 *   - limit (default: 10) - Number of recent games to predict
 * 
 * Returns array of predictions with:
 *   - Game info (date, teams, rankings)
 *   - Win probabilities
 *   - Top impacting features
 *   - Model confidence metrics
 */
router.get('/upcoming', async (req, res) => {
  try {
    const { limit = 10 } = req.query;
    const games = gameModel.getRecentGames(parseInt(limit));
    
    // Handle empty game list
    if (!games || games.length === 0) {
      return res.json({
        success: true,
        count: 0,
        message: 'No recent games found',
        data: []
      });
    }
    
    console.log(`Fetching predictions for ${games.length} games...`);
    
    // Get predictions from Flask service (concurrent)
    const predictions = await Promise.all(
      games.map(async (game) => {
        try {
          // Build complete ranking data with all 16 features
          const rankingData = buildRankingData(game);
          
          // Validate all features are present
          const validation = validateRankingData(rankingData);
          if (!validation.isValid) {
            console.warn(
              `Missing features for game ${game.GAME_ID}: ${validation.missingFeatures.join(', ')}`
            );
            return null;
          }
          
          // Call Flask prediction service
          const flaskRes = await axios.post(`${FLASK_URL}/predict`, {
            home_team: game.HOME_TEAM_NAME || `Team_${game.HOME_TEAM_ID}`,
            away_team: game.AWAY_TEAM_NAME || `Team_${game.AWAY_TEAM_ID}`,
            ranking_data: rankingData
          });
          
          // Handle successful prediction
          if (flaskRes.data.success && flaskRes.data.prediction.success) {
            return formatPredictionResponse(game, flaskRes.data.prediction);
          }
          
          console.warn(`Prediction failed for game ${game.GAME_ID}: ${flaskRes.data.prediction.error}`);
          return null;
        } catch (error) {
          console.error(`Prediction error for game ${game?.GAME_ID}: ${error.message}`);
          return null;
        }
      })
    );
    
    // Filter out failed predictions
    const validPredictions = predictions.filter(p => p !== null);
    
    res.json({
      success: true,
      count: validPredictions.length,
      total_games: games.length,
      failed_predictions: games.length - validPredictions.length,
      data: validPredictions
    });
  } catch (error) {
    console.error('Upcoming predictions error:', error.message);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * POST /api/predictions/single
 * Get prediction for a single game
 * 
 * Request Body:
 * {
 *   "game_id": "2025_LAL_BOS_1",
 *   "home_team_id": 1610612747,
 *   "away_team_id": 1610612738,
 *   "ranking_data": {
 *     "OFF_RNK_DIFF": 5,
 *     "DEF_RNK_DIFF": -3,
 *     ... (all 16 features)
 *   }
 * }
 * 
 * Returns: Single prediction with full model details
 */
router.post('/single', async (req, res) => {
  try {
    const { game_id, home_team_id, away_team_id, ranking_data } = req.body;
    
    // Validate request body
    if (!ranking_data) {
      return res.status(400).json({
        success: false,
        error: 'Missing ranking_data in request body'
      });
    }
    
    // Validate all features are present
    const validation = validateRankingData(ranking_data);
    if (!validation.isValid) {
      return res.status(400).json({
        success: false,
        error: 'Missing required features',
        missing_features: validation.missingFeatures,
        required_features: REQUIRED_FEATURES
      });
    }
    
    // Call Flask prediction service
    const flaskRes = await axios.post(`${FLASK_URL}/predict`, {
      home_team: req.body.home_team || `Team_${home_team_id}`,
      away_team: req.body.away_team || `Team_${away_team_id}`,
      ranking_data
    });
    
    if (!flaskRes.data.success) {
      return res.status(400).json({
        success: false,
        error: flaskRes.data.error,
        details: flaskRes.data
      });
    }
    
    res.json({
      success: true,
      game_id,
      home_team_id,
      away_team_id,
      prediction: flaskRes.data.prediction,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Single prediction error:', error.message);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * POST /api/predictions/batch
 * Get predictions for multiple games at once
 * 
 * Request Body:
 * {
 *   "games": [
 *     {
 *       "game_id": "2025_LAL_BOS_1",
 *       "home_team_id": 1610612747,
 *       "away_team_id": 1610612738,
 *       "ranking_data": { ... }
 *     },
 *     { ... }
 *   ]
 * }
 * 
 * Returns: Array of predictions with success/failure counts
 */
router.post('/batch', async (req, res) => {
  try {
    const { games } = req.body;
    
    if (!games || !Array.isArray(games)) {
      return res.status(400).json({
        success: false,
        error: 'games must be an array'
      });
    }
    
    if (games.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'games array cannot be empty'
      });
    }
    
    if (games.length > 100) {
      return res.status(400).json({
        success: false,
        error: 'Maximum 100 games per batch'
      });
    }
    
    console.log(`Processing batch prediction for ${games.length} games...`);
    
    // Get predictions concurrently
    const predictions = await Promise.all(
      games.map(async (gameReq) => {
        try {
          const { game_id, home_team_id, away_team_id, ranking_data } = gameReq;
          
          // Validate ranking data
          const validation = validateRankingData(ranking_data);
          if (!validation.isValid) {
            return {
              success: false,
              game_id,
              error: 'Missing required features',
              missing_features: validation.missingFeatures
            };
          }
          
          // Call Flask
          const flaskRes = await axios.post(`${FLASK_URL}/predict`, {
            home_team: gameReq.home_team || `Team_${home_team_id}`,
            away_team: gameReq.away_team || `Team_${away_team_id}`,
            ranking_data
          });
          
          if (flaskRes.data.success) {
            return {
              success: true,
              game_id,
              home_team_id,
              away_team_id,
              prediction: flaskRes.data.prediction
            };
          }
          
          return {
            success: false,
            game_id,
            error: flaskRes.data.error
          };
        } catch (error) {
          return {
            success: false,
            game_id: gameReq.game_id,
            error: error.message
          };
        }
      })
    );
    
    // Count results
    const successful = predictions.filter(p => p.success).length;
    const failed = predictions.length - successful;
    
    res.json({
      success: true,
      total_games: games.length,
      successful_predictions: successful,
      failed_predictions: failed,
      predictions
    });
  } catch (error) {
    console.error('Batch prediction error:', error.message);
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
    error: 'Prediction endpoint not found',
    path: req.path,
    method: req.method
  });
});

module.exports = router;