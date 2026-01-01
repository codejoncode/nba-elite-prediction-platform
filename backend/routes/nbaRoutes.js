const express = require('express');
const axios = require('axios');
const router = express.Router();

const ML_MODEL_API = process.env.ML_MODEL_API || 'http://localhost:5001';

router.get('/games/upcoming', async (req, res) => {
  try {
    console.log(`[NBA API] Fetching upcoming games from ${ML_MODEL_API}/api/games/upcoming`);
    
    const response = await axios.get(`${ML_MODEL_API}/api/games/upcoming`, {
      timeout: 5000
    });
    
    console.log(`[NBA API] ✅ Got ${response.data.count} upcoming games`);
    
    return res.json({
      success: true,
      data: response.data.games,
      count: response.data.count,
      source: 'ml-model'
    });
  } catch (error) {
    console.error(`[NBA API] ❌ Error fetching upcoming games:`, error.message);
    
    return res.status(error.response?.status || 500).json({
      success: false,
      error: error.message,
      details: error.response?.data || null
    });
  }
});

router.get('/predictions/history', async (req, res) => {
  try {
    console.log(`[NBA API] Fetching prediction history from ${ML_MODEL_API}/api/predictions/history`);
    
    const response = await axios.get(`${ML_MODEL_API}/api/predictions/history`, {
      timeout: 5000
    });
    
    console.log(`[NBA API] ✅ Got ${response.data.count} predictions`);
    
    return res.json({
      success: true,
      data: response.data.predictions,
      count: response.data.count,
      running_accuracy: response.data.running_accuracy,
      record: response.data.record,
      source: 'ml-model'
    });
  } catch (error) {
    console.error(`[NBA API] ❌ Error fetching prediction history:`, error.message);
    
    return res.status(error.response?.status || 500).json({
      success: false,
      error: error.message,
      details: error.response?.data || null
    });
  }
});

router.get('/predictions/stats', async (req, res) => {
  try {
    console.log(`[NBA API] Fetching prediction stats from ${ML_MODEL_API}/api/predictions/stats`);
    
    const response = await axios.get(`${ML_MODEL_API}/api/predictions/stats`, {
      timeout: 5000
    });
    
    console.log(`[NBA API] ✅ Got accuracy stats`);
    
    return res.json({
      success: true,
      data: response.data.stats,
      source: 'ml-model'
    });
  } catch (error) {
    console.error(`[NBA API] ❌ Error fetching prediction stats:`, error.message);
    
    return res.status(error.response?.status || 500).json({
      success: false,
      error: error.message,
      details: error.response?.data || null
    });
  }
});

router.post('/games/update-results', async (req, res) => {
  try {
    const cronToken = req.headers['x-cron-token'] || process.env.CRON_TOKEN || 'dev-token';
    
    console.log(`[NBA API] Updating game results with token: ${cronToken.substring(0, 10)}...`);
    
    const response = await axios.post(
      `${ML_MODEL_API}/api/games/update-results`,
      {},
      {
        headers: {
          'X-Cron-Token': cronToken
        },
        timeout: 10000
      }
    );
    
    console.log(`[NBA API] ✅ Game results updated`);
    
    return res.json({
      success: true,
      message: response.data.message,
      source: 'ml-model'
    });
  } catch (error) {
    console.error(`[NBA API] ❌ Error updating game results:`, error.message);
    
    return res.status(error.response?.status || 500).json({
      success: false,
      error: error.message,
      details: error.response?.data || null
    });
  }
});

router.get('/health', async (req, res) => {
  try {
    const mlResponse = await axios.get(`${ML_MODEL_API}/health`, {
      timeout: 3000
    });
    
    return res.json({
      success: true,
      backend: 'healthy',
      ml_model: mlResponse.data.status,
      ml_model_url: ML_MODEL_API
    });
  } catch (error) {
    return res.status(503).json({
      success: false,
      backend: 'healthy',
      ml_model: 'unreachable',
      ml_model_url: ML_MODEL_API,
      error: error.message
    });
  }
});

module.exports = router;