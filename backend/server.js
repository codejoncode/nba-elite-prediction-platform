const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/games', require('./routes/games'));
app.use('/api/predictions', require('./routes/predictions'));
app.use('/api/rankings', require('./routes/rankings'));

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'nba-elite-backend',
    version: '2.0',
    timestamp: new Date().toISOString()
  });
});

// Error handler
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(500).json({
    success: false,
    error: err.message
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════════════╗
║     🏀 NBA ELITE BACKEND SERVER                ║
║     Port: ${PORT}                              ║
║     Status: Running                            ║
╚════════════════════════════════════════════════╝
  `);
});
