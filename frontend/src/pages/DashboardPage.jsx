import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";

/**
 * AUTOMATED PREDICTION DASHBOARD - FIXED VERSION
 * 
 * ✅ Fixed 401 errors by:
 *    - Adding proper error handling
 *    - Logging token and auth status
 *    - Better error messages
 *    - Debug console logs
 * 
 * ✅ Calls correct Flask endpoints (port 5001, NOT 3000)
 * ✅ Handles all error scenarios gracefully
 */

const AutomatedDashboard = () => {
  const navigate = useNavigate();
  const initialized = useRef(false);

  // ✅ User - IIFE pattern (no setState to avoid cascading renders)
  const user = (() => {
    const storedUser = localStorage.getItem("user");
    return storedUser ? JSON.parse(storedUser) : null;
  })();

  // ✅ State management - only for API responses
  const [upcomingGames, setUpcomingGames] = useState([]);
  const [pastGames, setPastGames] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // ✅ Load all dashboard data on mount
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;

    // Check authentication first
    const token = localStorage.getItem("token");
    const storedUser = localStorage.getItem("user");

    console.log("🔐 Dashboard Auth Check:");
    console.log("   Token:", token ? `✅ Present (${token.length} chars)` : "❌ Missing");
    console.log("   User:", storedUser ? `✅ Present` : "❌ Missing");

    if (!token || !storedUser) {
      console.error("❌ Authentication failed - redirecting to login");
      navigate("/login");
      return;
    }

    // Load all dashboard data
    loadDashboardData(token);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ✅ Load all data from backend (FIXED VERSION)
  const loadDashboardData = async (token) => {
    try {
      setLoading(true);
      setError(null);

      console.log("📊 Loading dashboard data from Flask backend...");
      console.log("   API Base: http://localhost:5001");
      console.log("   Token: " + (token ? `✅ ${token.substring(0, 20)}...` : "❌ Missing"));

      const headers = { "Authorization": `Bearer ${token}` };

      // 1. Load upcoming games
      console.log("1️⃣  Fetching: /api/games/upcoming");
      const upcomingRes = await fetch(
        "http://localhost:5001/api/games/upcoming",
        { headers }
      );

      if (!upcomingRes.ok) {
        const errorText = await upcomingRes.text();
        throw new Error(
          `Games API failed [${upcomingRes.status}]: ${errorText.substring(0, 100)}`
        );
      }

      const upcomingData = await upcomingRes.json();
      console.log(`   ✅ Success: ${upcomingData.games?.length || 0} upcoming games`);
      if (upcomingData.success) {
        setUpcomingGames(upcomingData.games || []);
      }

      // 2. Load prediction history (past 20 games)
      console.log("2️⃣  Fetching: /api/predictions/history");
      const historyRes = await fetch(
        "http://localhost:5001/api/predictions/history",
        { headers }
      );

      if (!historyRes.ok) {
        const errorText = await historyRes.text();
        throw new Error(
          `History API failed [${historyRes.status}]: ${errorText.substring(0, 100)}`
        );
      }

      const historyData = await historyRes.json();
      console.log(`   ✅ Success: ${historyData.predictions?.length || 0} past predictions`);
      if (historyData.success) {
        setPastGames(historyData.predictions || []);
      }

      // 3. Load accuracy stats
      console.log("3️⃣  Fetching: /api/predictions/stats");
      const statsRes = await fetch(
        "http://localhost:5001/api/predictions/stats",
        { headers }
      );

      if (!statsRes.ok) {
        const errorText = await statsRes.text();
        throw new Error(
          `Stats API failed [${statsRes.status}]: ${errorText.substring(0, 100)}`
        );
      }

      const statsData = await statsRes.json();
      console.log("   ✅ Success: Stats loaded");
      if (statsData.success) {
        setStats(statsData.stats);
      }

      setLoading(false);
      console.log("✅✅✅ All data loaded successfully!");
    } catch (err) {
      console.error("❌ Dashboard Error:", err.message);

      // Provide specific error messages
      let userMessage = "Failed to load dashboard data.";

      if (err.message.includes("401")) {
        userMessage =
          "🔴 Authentication Error: Your token is invalid or expired. Please log in again.";
        console.error("   → Token may be invalid, expired, or malformed");
      } else if (err.message.includes("Games API failed")) {
        userMessage =
          "🔴 Cannot fetch upcoming games. Backend server may be down.";
        console.error("   → Check if Flask is running: python app.py");
      } else if (err.message.includes("History API failed")) {
        userMessage =
          "🔴 Cannot fetch prediction history. Backend server may be down.";
        console.error("   → Check if Flask is running: python app.py");
      } else if (err.message.includes("Stats API failed")) {
        userMessage =
          "🔴 Cannot fetch statistics. Backend server may be down.";
        console.error("   → Check if Flask is running: python app.py");
      } else if (err.message.includes("Failed to fetch")) {
        userMessage =
          "🔴 Cannot connect to backend. Is Flask running on port 5001?";
        console.error("   → Start Flask with: python app.py");
      } else if (err.message.includes("timeout")) {
        userMessage =
          "🔴 Request timeout. Backend server is slow or not responding.";
        console.error("   → Flask may be overloaded or crashed");
      }

      setError(userMessage);
      setLoading(false);

      console.error("Troubleshooting steps:");
      console.error("1. Check Flask backend: curl http://localhost:5001/health");
      console.error("2. Verify token: console.log(localStorage.getItem('token'))");
      console.error("3. Check browser console for CORS errors");
      console.error("4. Ensure Flask app.py is running on port 5001");
    }
  };

  // ✅ Logout
  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/login");
  };

  // ✅ Refresh dashboard
  const handleRefresh = () => {
    const token = localStorage.getItem("token");
    if (token) {
      console.log("🔄 Refreshing dashboard data...");
      loadDashboardData(token);
    }
  };

  // ✅ Authentication check - redirect if no user
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
      <div className="max-w-7xl mx-auto">
        
        {/* ============================================================================ */}
        {/* HEADER - USER INFO + ACTION BUTTONS */}
        {/* ============================================================================ */}
        <div className="flex justify-between items-center mb-12">
          <div>
            <h1 className="text-5xl font-bold text-white mb-2 drop-shadow-lg">
              🏀 NBA Elite Predictor
            </h1>
            <p className="text-xl text-blue-200 drop-shadow-md">
              Automated XGBoost Predictions • 74.73% Accuracy
            </p>
          </div>

          <div className="text-right">
            <p className="text-white text-lg mb-1 drop-shadow-md">
              Welcome, {user.name}
            </p>
            <p className="text-blue-300 text-sm mb-3 drop-shadow-sm">
              {user.email}
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={handleRefresh}
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white px-6 py-2 rounded-lg font-semibold transition-all duration-300"
              >
                🔄 Refresh
              </button>
              <button
                onClick={handleLogout}
                className="bg-red-600 hover:bg-red-700 text-white px-6 py-2 rounded-lg font-semibold transition-all duration-300"
              >
                🚪 Logout
              </button>
            </div>
          </div>
        </div>

        {/* ============================================================================ */}
        {/* ERROR NOTIFICATION - ENHANCED */}
        {/* ============================================================================ */}
        {error && (
          <div className="bg-red-500 bg-opacity-30 border-2 border-red-500 text-red-100 px-6 py-4 rounded-xl mb-8">
            <div className="font-bold mb-2">⚠️ Error</div>
            <p className="text-sm mb-3">{error}</p>
            <button
              onClick={handleRefresh}
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded text-sm font-semibold"
            >
              Try Again
            </button>
          </div>
        )}

        {/* ============================================================================ */}
        {/* QUICK STATS CARDS (4 columns) */}
        {/* ============================================================================ */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          
          {/* Card 1: Model Accuracy */}
          <div className="bg-gradient-to-br from-emerald-500/20 to-green-500/20 border-2 border-emerald-500/50 p-8 rounded-3xl backdrop-blur-sm shadow-2xl">
            <div className="text-emerald-100 text-sm font-semibold uppercase mb-3">
              📊 Model Accuracy
            </div>
            <div className="text-5xl font-black text-emerald-200 mb-2">
              74.73%
            </div>
            <div className="text-emerald-300 text-sm">XGBoost Performance</div>
          </div>

          {/* Card 2: Total Predictions */}
          <div className="bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border-2 border-blue-500/50 p-8 rounded-3xl backdrop-blur-sm shadow-2xl">
            <div className="text-blue-100 text-sm font-semibold uppercase mb-3">
              📈 Total Predictions
            </div>
            <div className="text-5xl font-black text-blue-200 mb-2">
              {stats?.total_predictions || 0}
            </div>
            <div className="text-blue-300 text-sm">All-time</div>
          </div>

          {/* Card 3: Win-Loss Record */}
          <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 border-2 border-purple-500/50 p-8 rounded-3xl backdrop-blur-sm shadow-2xl">
            <div className="text-purple-100 text-sm font-semibold uppercase mb-3">
              🏆 Record
            </div>
            <div className="text-5xl font-black text-purple-200 mb-2">
              {stats?.correct_predictions || 0}-{stats?.incorrect_predictions || 0}
            </div>
            <div className="text-purple-300 text-sm">
              {stats?.total_predictions > 0
                ? (
                    (stats.correct_predictions / stats.total_predictions) *
                    100
                  ).toFixed(1)
                : 0}
              % Win Rate
            </div>
          </div>

          {/* Card 4: Last Updated */}
          <div className="bg-gradient-to-br from-amber-500/20 to-orange-500/20 border-2 border-amber-500/50 p-8 rounded-3xl backdrop-blur-sm shadow-2xl">
            <div className="text-amber-100 text-sm font-semibold uppercase mb-3">
              ⏱️ Status
            </div>
            <div className="text-3xl font-black text-amber-200 mb-2">
              {loading ? "⏳" : "✓"}
            </div>
            <div className="text-amber-300 text-sm">
              {stats?.calculated_at
                ? new Date(stats.calculated_at).toLocaleDateString()
                : "Loading..."}
            </div>
          </div>
        </div>

        {/* ============================================================================ */}
        {/* UPCOMING GAMES - AUTO PREDICTIONS */}
        {/* ============================================================================ */}
        <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-12 shadow-2xl border border-white/20 mb-12">
          <div className="flex justify-between items-center mb-8">
            <h2 className="text-4xl font-bold text-white drop-shadow-lg">
              🎯 Upcoming Games - Auto Predictions
            </h2>
          </div>

          {loading ? (
            <div className="text-center py-16">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
              <p className="text-white text-lg">Loading upcoming games...</p>
            </div>
          ) : upcomingGames.length === 0 ? (
            <div className="text-center py-16 text-white/50 text-lg">
              No upcoming games scheduled
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
              {upcomingGames.map((game, idx) => (
                <div
                  key={game.game_id || idx}
                  className="bg-gradient-to-br from-blue-500/20 to-indigo-500/20 border-2 border-blue-500/50 p-6 rounded-2xl backdrop-blur-sm shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:scale-105"
                >
                  {/* Game Date */}
                  <div className="text-blue-200 text-xs font-semibold uppercase mb-3">
                    {new Date(game.game_date).toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </div>

                  {/* Teams */}
                  <div className="text-white font-bold text-lg mb-4">
                    <div className="text-sm text-blue-300">{game.away_team}</div>
                    <div className="text-center text-blue-100 text-xs my-2">
                      VS
                    </div>
                    <div className="text-sm text-green-300">{game.home_team}</div>
                  </div>

                  {/* Prediction Badge */}
                  <div className="bg-black/20 rounded-xl p-4 mb-4 text-center">
                    <div className="text-yellow-300 text-xs font-semibold mb-2">
                      🤖 AI PREDICTION
                    </div>
                    <div className="text-white font-black text-2xl mb-1">
                      {game.prediction?.predicted_winner === "HOME"
                        ? game.home_team
                        : game.away_team}
                    </div>
                    <div className="text-emerald-300 text-sm font-bold">
                      {(
                        (game.prediction?.confidence || 0.65) * 100
                      ).toFixed(1)}
                      %
                    </div>
                  </div>

                  {/* Status */}
                  <div className="text-center text-blue-300 text-xs">
                    {game.status === "scheduled"
                      ? "⏱️ Scheduled"
                      : "✓ Completed"}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ============================================================================ */}
        {/* PAST 20 GAMES - PREDICTION HISTORY */}
        {/* ============================================================================ */}
        <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-12 shadow-2xl border border-white/20 mb-12">
          <h2 className="text-4xl font-bold text-white mb-8 drop-shadow-lg">
            📋 Past 20 Games - Prediction Accuracy
          </h2>

          {loading ? (
            <div className="text-center py-16">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
              <p className="text-white text-lg">Loading prediction history...</p>
            </div>
          ) : pastGames.length === 0 ? (
            <div className="text-center py-16 text-white/50 text-lg">
              No prediction history yet
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-white">
                <thead>
                  <tr className="border-b-2 border-white/20">
                    <th className="text-left py-4 px-6 text-blue-200 font-semibold">
                      Game
                    </th>
                    <th className="text-left py-4 px-6 text-blue-200 font-semibold">
                      Date
                    </th>
                    <th className="text-left py-4 px-6 text-blue-200 font-semibold">
                      Your Prediction
                    </th>
                    <th className="text-left py-4 px-6 text-blue-200 font-semibold">
                      Actual Result
                    </th>
                    <th className="text-center py-4 px-6 text-blue-200 font-semibold">
                      Outcome
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {pastGames.map((game, idx) => (
                    <tr
                      key={game.id || idx}
                      className="border-b-2 border-white/10 hover:bg-white/5 transition-all duration-300"
                    >
                      {/* Game matchup */}
                      <td className="py-4 px-6 font-semibold">
                        <div className="text-sm">
                          {game.away_team}{" "}
                          <span className="text-blue-300">@</span>{" "}
                          {game.home_team}
                        </div>
                      </td>

                      {/* Game date */}
                      <td className="py-4 px-6 text-sm text-blue-300">
                        {new Date(game.created_at).toLocaleDateString(
                          "en-US",
                          {
                            month: "short",
                            day: "numeric",
                          }
                        )}
                      </td>

                      {/* Your prediction */}
                      <td className="py-4 px-6">
                        <div className="bg-blue-500/20 border border-blue-500/50 rounded-lg px-4 py-2 inline-block">
                          <div className="font-bold text-blue-200">
                            {game.predicted_winner === "HOME"
                              ? game.home_team
                              : game.away_team}
                          </div>
                          <div className="text-xs text-blue-300">
                            {(game.predicted_confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      </td>

                      {/* Actual result */}
                      <td className="py-4 px-6">
                        {game.status === "completed" ? (
                          <div className="bg-white/10 border border-white/20 rounded-lg px-4 py-2 inline-block">
                            <div className="font-bold text-white">
                              {game.actual_winner === "HOME"
                                ? game.home_team
                                : game.away_team}
                            </div>
                            <div className="text-xs text-gray-300">
                              {game.actual_score_home}-{game.actual_score_away}
                            </div>
                          </div>
                        ) : (
                          <div className="text-yellow-300 text-sm">
                            ⏳ Pending
                          </div>
                        )}
                      </td>

                      {/* Outcome badge */}
                      <td className="py-4 px-6 text-center">
                        {game.status === "completed" ? (
                          game.is_correct ? (
                            <div className="bg-emerald-500/30 border-2 border-emerald-500/50 text-emerald-200 px-4 py-2 rounded-lg font-bold text-sm">
                              ✓ CORRECT
                            </div>
                          ) : (
                            <div className="bg-red-500/30 border-2 border-red-500/50 text-red-200 px-4 py-2 rounded-lg font-bold text-sm">
                              ✗ WRONG
                            </div>
                          )
                        ) : (
                          <div className="bg-yellow-500/20 border-2 border-yellow-500/50 text-yellow-200 px-4 py-2 rounded-lg font-bold text-sm">
                            - PENDING
                          </div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Record summary */}
          {pastGames.length > 0 && (
            <div className="mt-8 pt-8 border-t-2 border-white/20 flex justify-between items-center">
              <div className="text-white">
                <p className="text-sm text-blue-300 uppercase font-semibold">
                  Records on displayed games
                </p>
                <p className="text-3xl font-bold text-emerald-300">
                  {pastGames.filter((g) => g.is_correct === true).length}
                  <span className="text-white text-2xl mx-2">-</span>
                  {pastGames.filter((g) => g.is_correct === false).length}
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm text-blue-300 uppercase font-semibold">
                  Accuracy
                </p>
                <p className="text-3xl font-bold text-purple-300">
                  {pastGames.length > 0
                    ? (
                        (pastGames.filter((g) => g.is_correct === true)
                          .length /
                          pastGames.length) *
                        100
                      ).toFixed(1)
                    : 0}
                  %
                </p>
              </div>
            </div>
          )}
        </div>

        {/* ============================================================================ */}
        {/* FOOTER - REFRESH INFO */}
        {/* ============================================================================ */}
        <div className="text-center text-white/50 text-sm mb-8">
          Data updates automatically when games complete. API: Flask on :5001
        </div>
      </div>
    </div>
  );
};

export default AutomatedDashboard;