import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { TeamSelector } from "../components/Prediction/TeamSelector";

const Dashboard = () => {
  const user = (() => {
    const storedUser = localStorage.getItem("user");
    return storedUser ? JSON.parse(storedUser) : null;
  })();

  const [features, setFeatures] = useState([]);
  const [predictionData, setPredictionData] = useState({
    home_team: "",
    away_team: "",
    ranking_data: Array(16).fill(0),
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const initialized = useRef(false);

  // Make prediction
  const handlePrediction = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const token = localStorage.getItem("token");
      const response = await fetch("http://localhost:5001/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(predictionData),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Prediction failed:", error);
      setResult({ success: false, error: "Prediction failed" });
    }

    setLoading(false);
  };

  // Update feature input
  const handleInputChange = (index, value) => {
    const newData = [...predictionData.ranking_data];
    newData[index] = parseFloat(value) || 0;
    setPredictionData({ ...predictionData, ranking_data: newData });
  };

  // Logout
  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/login");
  };

  // Load features on mount
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;

    const token = localStorage.getItem("token");
    const storedUser = localStorage.getItem("user");

    if (!token || !storedUser) {
      navigate("/login");
      return;
    }

    const loadFeatures = async () => {
      try {
        const response = await fetch("http://localhost:5001/features", {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await response.json();
        if (data.success) {
          setFeatures(data.features);
        }
      } catch (error) {
        console.error("Failed to load features:", error);
      }
    };

    loadFeatures();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-900 to-purple-900">
        <div className="text-white text-2xl animate-pulse">
          Loading Dashboard...
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-12">
          <div>
            <h1 className="text-5xl font-bold text-white mb-2 drop-shadow-lg">
              🏀 NBA Elite Predictor
            </h1>
            <p className="text-xl text-blue-200 drop-shadow-md">
              74.73% Accuracy | XGBoost | {features.length} Features Loaded
            </p>
          </div>
          <div className="text-right">
            <p className="text-white text-lg mb-1 drop-shadow-md">
              Welcome, {user.name}
            </p>
            <p className="text-blue-300 text-sm mb-3 drop-shadow-sm">
              ({user.email})
            </p>
            <button
              onClick={logout}
              className="bg-red-600 hover:bg-red-700 text-white px-8 py-3 rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
            >
              🚪 Logout
            </button>
          </div>
        </div>

        {/* Prediction Form */}
        <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-12 shadow-2xl border border-white/20 mb-12">
          <h2 className="text-4xl font-bold text-white mb-10 drop-shadow-lg">
            Make Prediction
          </h2>

          <form
            onSubmit={handlePrediction}
            className="grid grid-cols-1 lg:grid-cols-2 gap-8"
          >
            {/* Teams */}
            <div className="space-y-6">
              <TeamSelector
                id="home-team"
                label="🏠 Home Team"
                value={predictionData.home_team}
                onChange={(value) =>
                  setPredictionData({
                    ...predictionData,
                    home_team: value,
                  })
                }
              />
              <TeamSelector
                id="away-team"
                label="✈️ Away Team"
                value={predictionData.away_team}
                onChange={(value) =>
                  setPredictionData({
                    ...predictionData,
                    away_team: value,
                  })
                }
              />
            </div>

            {/* Features */}
            <div className="space-y-6">
              <label className="block text-white font-bold text-xl mb-6 flex items-center">
                📊 16 Elite Ranking Features
                <span className="ml-4 px-4 py-1 bg-blue-500/30 text-blue-100 text-sm font-semibold rounded-full">
                  {features.length} Loaded
                </span>
              </label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-96 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-blue-500 scrollbar-track-blue-900">
                {features.length > 0 ? (
                  features.map((feature, index) => (
                    <div key={`${feature}-${index}`} className="space-y-2">
                      <label className="text-white/90 text-sm font-medium block truncate bg-black/20 px-3 py-1 rounded-lg">
                        {feature}
                      </label>
                      <input
                        type="number"
                        step="0.01"
                        value={predictionData.ranking_data[index] || ""}
                        onChange={(e) =>
                          handleInputChange(index, e.target.value)
                        }
                        className="w-full p-4 bg-white/20 border border-white/30 rounded-xl text-white font-mono focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-500/50 text-sm transition-all duration-200 hover:border-white/50"
                        placeholder="0.00"
                      />
                    </div>
                  ))
                ) : (
                  <div className="col-span-2 text-center py-12 text-white/50 text-lg">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
                    Loading features from backend...
                  </div>
                )}
              </div>
            </div>

            {/* Submit Button */}
            <div className="lg:col-span-2 pt-12">
              <button
                type="submit"
                disabled={
                  loading ||
                  features.length === 0 ||
                  !predictionData.home_team ||
                  !predictionData.away_team
                }
                className="w-full bg-gradient-to-r from-emerald-500 via-green-500 to-emerald-600 hover:from-emerald-600 hover:via-green-600 hover:to-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-black py-8 px-12 rounded-3xl text-2xl shadow-2xl hover:shadow-4xl transform hover:-translate-y-2 active:translate-y-0 transition-all duration-500 group"
              >
                <span className="block">
                  {loading ? (
                    <>
                      <span className="animate-spin inline-block mr-3 h-8 w-8 border-4 border-white border-t-transparent rounded-full"></span>
                      🔮 Predicting...
                    </>
                  ) : (
                    <>
                      🚀 Predict Game Outcome
                      <span className="ml-4 text-yellow-300 text-sm font-normal block opacity-0 group-hover:opacity-100 group-hover:translate-y-1 transition-all duration-300">
                        74.73% Accurate
                      </span>
                    </>
                  )}
                </span>
              </button>
            </div>
          </form>
        </div>

        {/* Results */}
        {result && (
          <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-12 shadow-2xl border border-white/20">
            <h3 className="text-4xl font-bold text-white mb-8 drop-shadow-lg flex items-center">
              📈 Prediction Results
              <span
                className={`ml-4 px-4 py-2 rounded-full text-sm font-semibold ${
                  result.success
                    ? "bg-emerald-500/20 text-emerald-200 border-emerald-500/50 border"
                    : "bg-red-500/20 text-red-200 border-red-500/50 border"
                }`}
              >
                {result.success ? "✅ SUCCESS" : "❌ ERROR"}
              </span>
            </h3>

            {result.success ? (
              <div className="space-y-8">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                  <div className="bg-gradient-to-br from-green-500/20 to-emerald-500/20 border-2 border-green-500/50 p-8 rounded-3xl backdrop-blur-sm shadow-2xl">
                    <div className="text-center">
                      <h4 className="text-2xl font-bold text-green-100 mb-4 flex items-center justify-center">
                        🏠 {predictionData.home_team || "Home Team"}
                      </h4>
                      <div className="text-5xl font-black text-green-200 mb-2 drop-shadow-lg">
                        {result.prediction?.home_win_probability
                          ? (
                              result.prediction.home_win_probability * 100
                            ).toFixed(1)
                          : "N/A"}
                        %
                      </div>
                      <div className="text-green-100 text-lg">
                        Win Probability
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-red-500/20 to-pink-500/20 border-2 border-red-500/50 p-8 rounded-3xl backdrop-blur-sm shadow-2xl">
                    <div className="text-center">
                      <h4 className="text-2xl font-bold text-red-100 mb-4 flex items-center justify-center">
                        ✈️ {predictionData.away_team || "Away Team"}
                      </h4>
                      <div className="text-5xl font-black text-red-200 mb-2 drop-shadow-lg">
                        {result.prediction?.away_win_probability
                          ? (
                              result.prediction.away_win_probability * 100
                            ).toFixed(1)
                          : "N/A"}
                        %
                      </div>
                      <div className="text-red-100 text-lg">
                        Win Probability
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-purple-500/20 to-indigo-500/20 border-2 border-purple-500/50 p-8 rounded-3xl backdrop-blur-sm shadow-2xl col-span-1 md:col-span-2 lg:col-span-1">
                    <div className="text-center">
                      <div className="text-6xl font-black text-white mb-4 drop-shadow-2xl">
                        {result.prediction?.predicted_winner || "N/A"}
                      </div>
                      <div className="text-2xl font-bold text-purple-100 mb-2">
                        Predicted Winner
                      </div>
                      <div className="text-xl text-blue-200">
                        Confidence:{" "}
                        {result.prediction?.confidence
                          ? (result.prediction.confidence * 100).toFixed(1)
                          : "N/A"}
                        %
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center p-16 bg-gradient-to-r from-red-500/10 to-rose-500/10 border-2 border-red-500/40 rounded-3xl backdrop-blur-sm shadow-xl">
                <div className="text-6xl mb-6">⚠️</div>
                <h4 className="text-3xl font-bold text-red-200 mb-4 drop-shadow-lg">
                  Prediction Error
                </h4>
                <p className="text-xl text-red-300 max-w-2xl mx-auto leading-relaxed">
                  {result.error}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
