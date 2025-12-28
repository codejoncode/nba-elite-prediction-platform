import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';
import { TrendingUp, Trophy, Zap, Target, Plus, Send } from 'lucide-react';

// Flask backend URL
const FLASK_API = 'http://localhost:5001';

function App() {
  const [predictions, setPredictions] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showPredictForm, setShowPredictForm] = useState(false);
  const [formLoading, setFormLoading] = useState(false);
  const [formData, setFormData] = useState({
    homeTeam: 'Lakers',
    awayTeam: 'Celtics',
    OFF_RNK_DIFF: 5,
    DEF_RNK_DIFF: -3,
    PTS_AVG_DIFF: 2.5,
    DEF_AVG_DIFF: -1.2,
    HOME_OFF_RANK: 8,
    HOME_DEF_RANK: 12,
    AWAY_OFF_RANK: 3,
    AWAY_DEF_RANK: 15,
    HOME_RUNNING_OFF_RANK: 7,
    HOME_RUNNING_DEF_RANK: 11,
    OFF_MOMENTUM: -1,
    DEF_MOMENTUM: -1,
    RANK_INTERACTION: -15,
    PTS_RANK_INTERACTION: 12.5,
    HOME_COURT: 1,
    GAME_NUMBER: 10
  });

  useEffect(() => {
    fetchAll();
  }, []);

  const fetchAll = async () => {
    setLoading(true);
    try {
      // Get model metrics from Flask
      try {
        const metricsRes = await axios.get(`${FLASK_API}/metrics`);
        const metricsData = metricsRes.data.metrics || metricsRes.data;
        setMetrics(metricsData);
        console.log('✓ Metrics loaded:', metricsData);
      } catch (e) {
        console.warn('Could not fetch metrics:', e.message);
      }

      // For demo: Create sample prediction data
      const samplePredictions = [
        {
          home_team: 'Lakers',
          away_team: 'Celtics',
          prediction: {
            home_win_probability: 0.62,
            away_win_probability: 0.38,
            predicted_winner: 'HOME',
            confidence: 0.62
          }
        },
        {
          home_team: 'Warriors',
          away_team: 'Suns',
          prediction: {
            home_win_probability: 0.45,
            away_win_probability: 0.55,
            predicted_winner: 'AWAY',
            confidence: 0.55
          }
        },
        {
          home_team: 'Mavericks',
          away_team: 'Grizzlies',
          prediction: {
            home_win_probability: 0.58,
            away_win_probability: 0.42,
            predicted_winner: 'HOME',
            confidence: 0.58
          }
        },
        {
          home_team: 'Heat',
          away_team: 'Bucks',
          prediction: {
            home_win_probability: 0.48,
            away_win_probability: 0.52,
            predicted_winner: 'AWAY',
            confidence: 0.52
          }
        },
        {
          home_team: 'Nuggets',
          away_team: 'Kings',
          prediction: {
            home_win_probability: 0.71,
            away_win_probability: 0.29,
            predicted_winner: 'HOME',
            confidence: 0.71
          }
        }
      ];

      setPredictions(samplePredictions);
      console.log('✓ Predictions loaded:', samplePredictions.length);

    } catch (err) {
      console.error('Error loading data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Make a real prediction with ranking data
  const makePrediction = async (homeTeam, awayTeam, rankingData) => {
    setFormLoading(true);
    try {
      const response = await axios.post(
        `${FLASK_API}/predict`,
        {
          home_team: homeTeam,
          away_team: awayTeam,
          ranking_data: rankingData
        },
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );
      
      if (response.data.success) {
        // Add the new prediction to the list
        const newPred = {
          home_team: homeTeam,
          away_team: awayTeam,
          prediction: response.data.prediction
        };
        
        setPredictions([newPred, ...predictions]);
        alert(`✓ Prediction saved!\n${homeTeam} vs ${awayTeam}\nConfidence: ${(response.data.prediction.confidence * 100).toFixed(1)}%`);
        setShowPredictForm(false);
      } else {
        alert(`✗ Error: ${response.data.error}`);
      }
    } catch (err) {
      console.error('Prediction error:', err);
      alert(`✗ Prediction failed: ${err.response?.data?.error || err.message}`);
    } finally {
      setFormLoading(false);
    }
  };

  const handleFormChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: isNaN(value) ? value : parseFloat(value)
    }));
  };

  const handleSubmitPrediction = (e) => {
    e.preventDefault();
    
    // Build ranking_data object from form
    const rankingData = {
      OFF_RNK_DIFF: formData.OFF_RNK_DIFF,
      DEF_RNK_DIFF: formData.DEF_RNK_DIFF,
      PTS_AVG_DIFF: formData.PTS_AVG_DIFF,
      DEF_AVG_DIFF: formData.DEF_AVG_DIFF,
      HOME_OFF_RANK: formData.HOME_OFF_RANK,
      HOME_DEF_RANK: formData.HOME_DEF_RANK,
      AWAY_OFF_RANK: formData.AWAY_OFF_RANK,
      AWAY_DEF_RANK: formData.AWAY_DEF_RANK,
      HOME_RUNNING_OFF_RANK: formData.HOME_RUNNING_OFF_RANK,
      HOME_RUNNING_DEF_RANK: formData.HOME_RUNNING_DEF_RANK,
      OFF_MOMENTUM: formData.OFF_MOMENTUM,
      DEF_MOMENTUM: formData.DEF_MOMENTUM,
      RANK_INTERACTION: formData.RANK_INTERACTION,
      PTS_RANK_INTERACTION: formData.PTS_RANK_INTERACTION,
      HOME_COURT: formData.HOME_COURT,
      GAME_NUMBER: formData.GAME_NUMBER
    };
    
    makePrediction(formData.homeTeam, formData.awayTeam, rankingData);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900">
        <div className="text-center">
          <div className="text-4xl mb-4">🏀</div>
          <p className="text-white text-xl">Loading Elite Predictions...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900">
        <div className="text-red-400 text-lg">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <div className="flex items-center justify-between gap-4 mb-12">
          <div className="flex items-center gap-4">
            <div className="text-5xl">🏀</div>
            <div>
              <h1 className="text-5xl font-bold">NBA Elite Predictions</h1>
              <p className="text-gray-400 mt-2">XGBoost Model • 74.73% Accuracy • ROC-AUC: 0.8261</p>
            </div>
          </div>
          <button
            onClick={() => setShowPredictForm(!showPredictForm)}
            className="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg font-bold flex items-center gap-2"
          >
            <Plus className="w-5 h-5" />
            New Prediction
          </button>
        </div>

        {/* Prediction Form Modal */}
        {showPredictForm && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 max-w-2xl w-full mx-4 max-h-96 overflow-y-auto">
              <h2 className="text-2xl font-bold mb-6">Make a New Prediction</h2>
              
              <form onSubmit={handleSubmitPrediction}>
                {/* Team Names */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <label className="block text-sm font-semibold mb-2">Home Team</label>
                    <input
                      type="text"
                      name="homeTeam"
                      value={formData.homeTeam}
                      onChange={handleFormChange}
                      className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-semibold mb-2">Away Team</label>
                    <input
                      type="text"
                      name="awayTeam"
                      value={formData.awayTeam}
                      onChange={handleFormChange}
                      className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
                    />
                  </div>
                </div>

                {/* Ranking Features */}
                <div className="grid grid-cols-2 gap-3 mb-4 text-sm">
                  <div>
                    <label className="text-xs font-semibold text-gray-400">OFF_RNK_DIFF</label>
                    <input type="number" name="OFF_RNK_DIFF" value={formData.OFF_RNK_DIFF} onChange={handleFormChange} step="0.1" className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white" />
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-gray-400">DEF_RNK_DIFF</label>
                    <input type="number" name="DEF_RNK_DIFF" value={formData.DEF_RNK_DIFF} onChange={handleFormChange} step="0.1" className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white" />
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-gray-400">PTS_AVG_DIFF</label>
                    <input type="number" name="PTS_AVG_DIFF" value={formData.PTS_AVG_DIFF} onChange={handleFormChange} step="0.1" className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white" />
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-gray-400">DEF_AVG_DIFF</label>
                    <input type="number" name="DEF_AVG_DIFF" value={formData.DEF_AVG_DIFF} onChange={handleFormChange} step="0.1" className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white" />
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-gray-400">HOME_OFF_RANK</label>
                    <input type="number" name="HOME_OFF_RANK" value={formData.HOME_OFF_RANK} onChange={handleFormChange} step="1" className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white" />
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-gray-400">HOME_DEF_RANK</label>
                    <input type="number" name="HOME_DEF_RANK" value={formData.HOME_DEF_RANK} onChange={handleFormChange} step="1" className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white" />
                  </div>
                </div>

                {/* Buttons */}
                <div className="flex gap-3">
                  <button
                    type="submit"
                    disabled={formLoading}
                    className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-4 py-2 rounded font-bold flex items-center justify-center gap-2"
                  >
                    <Send className="w-4 h-4" />
                    {formLoading ? 'Predicting...' : 'Get Prediction'}
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowPredictForm(false)}
                    className="flex-1 bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded font-bold"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* Metrics Cards */}
        {metrics && (
          <div className="grid grid-cols-4 gap-4 mb-8">
            {/* Accuracy */}
            <div className="bg-gradient-to-br from-green-900 to-green-800 p-6 rounded-lg border border-green-700">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-5 h-5 text-green-400" />
                <span className="text-green-200 text-sm font-semibold">ACCURACY</span>
              </div>
              <div className="text-4xl font-bold text-green-400">
                {(metrics.accuracy * 100).toFixed(2)}%
              </div>
              <div className="text-xs text-green-300 mt-2">vs 50% baseline</div>
            </div>

            {/* ROC-AUC */}
            <div className="bg-gradient-to-br from-blue-900 to-blue-800 p-6 rounded-lg border border-blue-700">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-5 h-5 text-blue-400" />
                <span className="text-blue-200 text-sm font-semibold">ROC-AUC</span>
              </div>
              <div className="text-4xl font-bold text-blue-400">
                {metrics.roc_auc?.toFixed(4) || 'N/A'}
              </div>
              <div className="text-xs text-blue-300 mt-2">model discrimination</div>
            </div>

            {/* Sensitivity (Recall) */}
            <div className="bg-gradient-to-br from-purple-900 to-purple-800 p-6 rounded-lg border border-purple-700">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-5 h-5 text-purple-400" />
                <span className="text-purple-200 text-sm font-semibold">SENSITIVITY</span>
              </div>
              <div className="text-4xl font-bold text-purple-400">
                {(metrics.sensitivity * 100).toFixed(2)}%
              </div>
              <div className="text-xs text-purple-300 mt-2">true positive rate</div>
            </div>

            {/* Best Iteration */}
            <div className="bg-gradient-to-br from-orange-900 to-orange-800 p-6 rounded-lg border border-orange-700">
              <div className="flex items-center gap-2 mb-2">
                <Trophy className="w-5 h-5 text-orange-400" />
                <span className="text-orange-200 text-sm font-semibold">BEST ITERATION</span>
              </div>
              <div className="text-4xl font-bold text-orange-400">
                {metrics.best_iteration || 'N/A'}
              </div>
              <div className="text-xs text-orange-300 mt-2">early stopping</div>
            </div>
          </div>
        )}

        {/* Feature Importance Section */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-8">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
            <Trophy className="w-6 h-6 text-yellow-400" />
            Top 5 Features (Feature Importance)
          </h2>
          <div className="grid grid-cols-5 gap-4">
            <div className="bg-gray-700 p-4 rounded">
              <div className="text-yellow-400 font-bold mb-1">DEF_AVG_DIFF</div>
              <div className="text-2xl font-bold">21.81%</div>
              <div className="text-xs text-gray-400 mt-1">Defensive 5-game average</div>
            </div>
            <div className="bg-gray-700 p-4 rounded">
              <div className="text-yellow-400 font-bold mb-1">PTS_AVG_DIFF</div>
              <div className="text-2xl font-bold">18.84%</div>
              <div className="text-xs text-gray-400 mt-1">Scoring differential</div>
            </div>
            <div className="bg-gray-700 p-4 rounded">
              <div className="text-yellow-400 font-bold mb-1">PTS_RANK_INTERACTION</div>
              <div className="text-2xl font-bold">9.41%</div>
              <div className="text-xs text-gray-400 mt-1">Multiplicative interaction</div>
            </div>
            <div className="bg-gray-700 p-4 rounded">
              <div className="text-yellow-400 font-bold mb-1">OFF_RNK_DIFF</div>
              <div className="text-2xl font-bold">8.62%</div>
              <div className="text-xs text-gray-400 mt-1">Offensive rank difference</div>
            </div>
            <div className="bg-gray-700 p-4 rounded">
              <div className="text-yellow-400 font-bold mb-1">DEF_RNK_DIFF</div>
              <div className="text-2xl font-bold">7.43%</div>
              <div className="text-xs text-gray-400 mt-1">Defensive rank difference</div>
            </div>
          </div>
        </div>

        {/* Predictions Table */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden mb-8">
          <div className="px-6 py-4 border-b border-gray-700 bg-gray-750">
            <h2 className="text-xl font-bold">Recent Game Predictions ({predictions.length})</h2>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-700">
                <tr>
                  <th className="px-6 py-3 text-left">Home Team</th>
                  <th className="px-6 py-3 text-left">Away Team</th>
                  <th className="px-6 py-3 text-center">Home %</th>
                  <th className="px-6 py-3 text-center">Away %</th>
                  <th className="px-6 py-3 text-center">Prediction</th>
                  <th className="px-6 py-3 text-center">Confidence</th>
                  <th className="px-6 py-3 text-center">Status</th>
                </tr>
              </thead>
              <tbody>
                {predictions.slice(0, 10).map((pred, i) => (
                  <tr key={i} className="border-t border-gray-700 hover:bg-gray-700/50">
                    <td className="px-6 py-3 font-semibold text-blue-400">{pred.home_team}</td>
                    <td className="px-6 py-3 font-semibold text-red-400">{pred.away_team}</td>
                    <td className="px-6 py-3 text-center">
                      <span className="bg-blue-900 px-3 py-1 rounded font-semibold">
                        {(pred.prediction.home_win_probability * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="px-6 py-3 text-center">
                      <span className="bg-red-900 px-3 py-1 rounded font-semibold">
                        {(pred.prediction.away_win_probability * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="px-6 py-3 text-center">
                      <span className={`px-3 py-1 rounded font-bold text-white ${
                        pred.prediction.predicted_winner === 'HOME' 
                          ? 'bg-green-600' 
                          : 'bg-orange-600'
                      }`}>
                        {pred.prediction.predicted_winner === 'HOME' ? '🏠 Home' : '🛫 Away'}
                      </span>
                    </td>
                    <td className="px-6 py-3 text-center font-semibold">
                      {(pred.prediction.confidence * 100).toFixed(0)}%
                    </td>
                    <td className="px-6 py-3 text-center text-xs">
                      <span className="text-gray-400">Pending</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Charts Section */}
        {predictions.length > 0 && (
          <div className="grid grid-cols-2 gap-8">
            {/* Win Probability Chart */}
            <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
              <h3 className="text-lg font-bold mb-4">Win Probability Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={predictions.slice(0, 5).map(p => ({
                  match: `${p.home_team.substring(0, 3)} vs ${p.away_team.substring(0, 3)}`,
                  home: p.prediction.home_win_probability,
                  away: p.prediction.away_win_probability
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis dataKey="match" stroke="#999" />
                  <YAxis stroke="#999" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #666' }}
                    formatter={(value) => (value * 100).toFixed(1) + '%'}
                  />
                  <Legend />
                  <Bar dataKey="home" name="Home Win %" fill="#3b82f6" />
                  <Bar dataKey="away" name="Away Win %" fill="#ef4444" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Confidence Chart */}
            <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
              <h3 className="text-lg font-bold mb-4">Model Confidence</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={predictions.slice(0, 10).map((p, i) => ({
                  game: i + 1,
                  confidence: p.prediction.confidence
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis dataKey="game" stroke="#999" />
                  <YAxis stroke="#999" domain={[0, 1]} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #666' }}
                    formatter={(value) => (value * 100).toFixed(1) + '%'}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="confidence" name="Confidence" stroke="#fbbf24" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
