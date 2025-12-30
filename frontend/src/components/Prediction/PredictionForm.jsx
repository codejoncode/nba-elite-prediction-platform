import React, { useState } from 'react';
import { TeamSelector } from './TeamSelector';
import { Send } from 'lucide-react';
import { useAuth } from '../../hooks/useAuth';

export const PredictionForm = ({ onPredictionSuccess }) => {
  const { token } = useAuth();
  const [showForm, setShowForm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    homeTeam: '',
    awayTeam: '',
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

  const handleTeamChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleFeatureChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: isNaN(value) ? value : parseFloat(value)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!formData.homeTeam || !formData.awayTeam) {
      alert('Please select both teams');
      return;
    }

    if (formData.homeTeam === formData.awayTeam) {
      alert('Home and away teams must be different');
      return;
    }

    setLoading(true);

    try {
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

      const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          home_team: formData.homeTeam,
          away_team: formData.awayTeam,
          ranking_data: rankingData
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Prediction failed');
      }

      const result = await response.json();
      onPredictionSuccess(result.prediction);
      setShowForm(false);
      alert(`✓ Prediction saved!\n${formData.homeTeam} vs ${formData.awayTeam}\nConfidence: ${(result.prediction.confidence * 100).toFixed(1)}%`);
    } catch (err) {
      alert(`✗ Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <button
        onClick={() => setShowForm(!showForm)}
        className="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg font-bold flex items-center gap-2"
      >
        + New Prediction
      </button>

      {showForm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 max-w-2xl w-full max-h-96 overflow-y-auto">
            <h2 className="text-2xl font-bold mb-6">Make a New Prediction</h2>

            <form onSubmit={handleSubmit}>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <TeamSelector
                  label="Home Team"
                  value={formData.homeTeam}
                  onChange={(value) => handleTeamChange('homeTeam', value)}
                />
                <TeamSelector
                  label="Away Team"
                  value={formData.awayTeam}
                  onChange={(value) => handleTeamChange('awayTeam', value)}
                />
              </div>

              <div className="grid grid-cols-2 gap-3 mb-4 text-sm">
                {['OFF_RNK_DIFF', 'DEF_RNK_DIFF', 'PTS_AVG_DIFF', 'DEF_AVG_DIFF', 'HOME_OFF_RANK', 'HOME_DEF_RANK'].map(field => (
                  <div key={field}>
                    <label className="text-xs font-semibold text-gray-400">{field}</label>
                    <input
                      type="number"
                      name={field}
                      value={formData[field]}
                      onChange={handleFeatureChange}
                      step="0.1"
                      className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white"
                    />
                  </div>
                ))}
              </div>

              <div className="flex gap-3">
                <button
                  type="submit"
                  disabled={loading}
                  className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-4 py-2 rounded font-bold flex items-center justify-center gap-2"
                >
                  <Send className="w-4 h-4" />
                  {loading ? 'Predicting...' : 'Get Prediction'}
                </button>
                <button
                  type="button"
                  onClick={() => setShowForm(false)}
                  className="flex-1 bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded font-bold"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
};
