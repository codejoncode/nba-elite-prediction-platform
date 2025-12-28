const express = require('express');
const gameModel = require('../models/gameModel');

const router = express.Router();

/**
 * GET /api/games
 * Get games with optional filters
 * ?start_date=2023-01-01&end_date=2023-12-31
 */
router.get('/', (req, res) => {
  try {
    const { start_date, end_date, team_id, limit } = req.query;
    
    let games;
    
    if (team_id) {
      games = gameModel.getTeamGames(team_id);
    } else if (start_date && end_date) {
      games = gameModel.getGames(start_date, end_date);
    } else {
      games = gameModel.getAllGames();
    }
    
    // Apply limit
    if (limit) {
      games = games.slice(-parseInt(limit));
    }
    
    res.json({
      success: true,
      count: games.length,
      data: games
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

module.exports = router;