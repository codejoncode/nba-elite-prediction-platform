const express = require('express');
const path = require('path');
const fs = require('fs');
const csv = require('csv-parser');
const cors = require('cors');
require('dotenv').config();

const app = express();

// ============================================================================
// GAMES DATA LOADING FROM ML-MODEL CSV (SINGLE SOURCE OF TRUTH)
// ============================================================================

const GAMES_CSV_PATH = path.join(__dirname, '..', 'ml-model', 'data', 'nba_games_elite.csv');

async function loadGamesFromCSV() {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(GAMES_CSV_PATH)) {
      console.error(`❌ CSV not found at: ${GAMES_CSV_PATH}`);
      resolve([]);
      return;
    }

    const games = [];

    fs.createReadStream(GAMES_CSV_PATH)
      .pipe(csv())
      .on('data', (row) => {
        try {
          games.push({
            game_id: row.game_id,
            date: new Date(row.game_date),
            home_team: row.home_team,
            away_team: row.away_team,
            predicted_winner: row.predicted_winner,
            confidence: parseFloat(row.confidence) || 0.65,
            actual_winner: row.actual_winner || null,
            actual_score_home: row.actual_score_home ? parseInt(row.actual_score_home, 10) : null,
            actual_score_away: row.actual_score_away ? parseInt(row.actual_score_away, 10) : null,
            is_correct:
              row.is_correct === 'true' ||
              row.is_correct === 'True' ||
              row.is_correct === '1',
            created_at: new Date(),
          });
        } catch (e) {
          console.warn(`⚠️  Warning processing row: ${e.message}`);
        }
      })
      .on('end', () => {
        console.log(`✅ Loaded ${games.length} games from ml-model CSV`);
        resolve(games);
      })
      .on('error', (err) => {
        console.error(`❌ Error reading CSV: ${err.message}`);
        reject(err);
      });
  });
}

// In-memory store
let gamesData = [];

// Initialize on startup
async function initializeData() {
  try {
    console.log('📂 Loading games data from single source (ml-model CSV)...');
    gamesData = await loadGamesFromCSV();
    console.log(`✅ Backend data initialized with ${gamesData.length} games`);
    return true;
  } catch (error) {
    console.error('❌ Failed to load data:', error);
    return false;
  }
}

// ============================================================================
// MIDDLEWARE
// ============================================================================

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use(
  cors({
    origin: process.env.FRONTEND_URL || 'http://localhost:3000',
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
  })
);

// Logging middleware
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// ============================================================================
// ROUTES
// ============================================================================

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'nba-elite-backend',
    gamesLoaded: gamesData.length,
    timestamp: new Date().toISOString(),
  });
});

// Get all games
app.get('/api/games', (req, res) => {
  res.json({
    success: true,
    count: gamesData.length,
    games: gamesData,
  });
});

// Get upcoming games (future dates)
app.get('/api/games/upcoming', (req, res) => {
  try {
    const today = new Date();
    const upcoming = gamesData.filter((g) => new Date(g.date) > today);

    res.json({
      success: true,
      count: upcoming.length,
      games: upcoming.slice(0, 10),
    });
  } catch (error) {
    console.error('Error fetching upcoming games:', error);
    res.status(500).json({ error: 'Failed to fetch games' });
  }
});

// Get predictions history (past games)
app.get('/api/predictions/history', (req, res) => {
  try {
    const today = new Date();
    const history = gamesData.filter(
      (g) => new Date(g.date) <= today && g.is_correct !== null
    );

    const correct_count = history.filter((g) => g.is_correct === true).length;
    const accuracy = history.length > 0 ? (correct_count / history.length) * 100 : 0;

    res.json({
      success: true,
      predictions: history.slice(0, 20),
      count: history.length,
      accuracy: parseFloat(accuracy.toFixed(2)),
      record: `${correct_count}-${history.length - correct_count}`,
    });
  } catch (error) {
    console.error('Error fetching prediction history:', error);
    res.status(500).json({ error: 'Failed to fetch history' });
  }
});

// Get accuracy stats
app.get('/api/predictions/stats', (req, res) => {
  try {
    const total = gamesData.length;
    const correct = gamesData.filter((g) => g.is_correct === true).length;
    const incorrect = total - correct;
    const accuracy = total > 0 ? (correct / total) * 100 : 0;

    res.json({
      success: true,
      stats: {
        total_predictions: total,
        correct_predictions: correct,
        incorrect_predictions: incorrect,
        overall_accuracy: parseFloat(accuracy.toFixed(2)),
        calculated_at: new Date().toISOString(),
      },
    });
  } catch (error) {
    console.error('Error fetching stats:', error);
    res.status(500).json({ error: 'Failed to fetch stats' });
  }
});

// Get single game by ID
app.get('/api/games/:game_id', (req, res) => {
  try {
    const game = gamesData.find((g) => g.game_id === req.params.game_id);

    if (!game) {
      return res.status(404).json({ error: 'Game not found' });
    }

    res.json({
      success: true,
      game,
    });
  } catch (error) {
    console.error('Error fetching game:', error);
    res.status(500).json({ error: 'Failed to fetch game' });
  }
});

// ============================================================================
// ERROR HANDLERS
// ============================================================================

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
    path: req.path,
  });
});

// Error handler
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    success: false,
    error: err.message,
  });
});

// ============================================================================
// SERVER STARTUP
// ============================================================================

const PORT = process.env.PORT || 3001;
const HOST = process.env.HOST || '0.0.0.0';

async function startServer() {
  try {
    // Initialize data first
    await initializeData();

    // Then start server
    app.listen(PORT, HOST, () => {
      console.log('\n╔════════════════════════════════════════════════╗');
      console.log('║     🏀 NBA ELITE NODE.JS BACKEND SERVER        ║');
      console.log(`║     Host: ${HOST.padEnd(40)}║`);
      console.log(`║     Port: ${PORT.toString().padEnd(40)}║`);
      console.log(`║     Games: ${gamesData.length.toString().padEnd(39)}║`);
      console.log('║     Status: Running ✅                         ║');
      console.log('║     Data: ml-model/data/nba_games_elite.csv     ║');
      console.log('╚════════════════════════════════════════════════╝\n');
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Start the server
startServer();

module.exports = app;