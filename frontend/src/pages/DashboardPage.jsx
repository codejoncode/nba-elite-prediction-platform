import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import StatsCards from "../components/StatsCards";
import UpcomingGamesGrid from "../components/UpcomingGamesGrid";
import PastGamesTable from "../components/PastGamesTable";
import DashboardControls from "../components/DashboardControls";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorNotification from "../components/ErrorNotification";

const DashboardPage = () => {
  const navigate = useNavigate();
  const initialized = useRef(false);

  // Authentication
  const user = (() => {
    const storedUser = localStorage.getItem("user");
    return storedUser ? JSON.parse(storedUser) : null;
  })();

  // State management
  const [gameResults, setGameResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastRefresh, setLastRefresh] = useState(null);

  // Load dashboard data from Flask
  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch("http://localhost:5001/api/game_results");

      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }

      const data = await response.json();
      
      // Validate data structure
      if (!data.metadata || !data.recent_results || !data.upcoming_games) {
        throw new Error("Invalid data structure from API");
      }
      
      setGameResults(data);
      setLastRefresh(new Date());
      
      console.log('✓ Dashboard data loaded:', {
        accuracy: data.metadata.accuracy_all_time_percent,
        recent: data.recent_results.length,
        upcoming: data.upcoming_games.length
      });
      
    } catch (err) {
      const errorMsg = err.message || "Failed to load dashboard data";
      setError(errorMsg);
      console.error("Dashboard data error:", err);
    } finally {
      setLoading(false);
    }
  };

  // Initial load on mount
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;

    if (!user) {
      navigate("/login");
      return;
    }

    loadDashboardData();
  }, [navigate, user]);

  // Auto-refresh every 5 minutes
  useEffect(() => {
    const interval = setInterval(loadDashboardData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  if (!user) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-8">
      
      {/* Header */}
      <div className="mb-12">
        <h1 className="text-5xl font-black mb-2">🏀 NBA Elite Predictions</h1>
        <p className="text-slate-400">
          Real-time ML predictions powered by XGBoost • {gameResults?.metadata.total_games_tracked || 0} games tracked
        </p>
      </div>

      {/* Controls - Manual Refresh + Status */}
      <DashboardControls 
        onRefresh={loadDashboardData}
        loading={loading}
        lastRefresh={lastRefresh}
      />

      {/* Error Notification */}
      {error && <ErrorNotification error={error} onDismiss={() => setError(null)} />}

      {/* Loading Spinner */}
      {loading && !gameResults && <LoadingSpinner />}

      {/* Dashboard Content */}
      {!loading && gameResults && (
        <>
          {/* Stats Cards */}
          <StatsCards 
            metadata={gameResults.metadata}
          />

          {/* Quick Stats Banner */}
          <div className="mb-8 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gradient-to-r from-green-900/40 to-green-800/40 border border-green-700/50 rounded-lg p-4">
              <div className="text-green-400 text-sm font-medium mb-1">All-Time Record</div>
              <div className="text-3xl font-bold text-white">
                {gameResults.metadata.record_wins}-{gameResults.metadata.record_losses}
              </div>
              <div className="text-green-300 text-sm mt-1">
                {gameResults.metadata.accuracy_all_time_percent}% accuracy
              </div>
            </div>

            <div className="bg-gradient-to-r from-blue-900/40 to-blue-800/40 border border-blue-700/50 rounded-lg p-4">
              <div className="text-blue-400 text-sm font-medium mb-1">Last 20 Games</div>
              <div className="text-3xl font-bold text-white">
                {gameResults.metadata.accuracy_last_20_percent}%
              </div>
              <div className="text-blue-300 text-sm mt-1">
                {Math.round(gameResults.recent_results.filter(g => g.is_correct).length)} correct predictions
              </div>
            </div>

            <div className="bg-gradient-to-r from-purple-900/40 to-purple-800/40 border border-purple-700/50 rounded-lg p-4">
              <div className="text-purple-400 text-sm font-medium mb-1">Upcoming Predictions</div>
              <div className="text-3xl font-bold text-white">
                {gameResults.metadata.upcoming_games_count}
              </div>
              <div className="text-purple-300 text-sm mt-1">
                Next games forecasted
              </div>
            </div>
          </div>

          {/* Upcoming Games */}
          <div className="mb-8">
            <h2 className="text-3xl font-bold mb-4">🔮 Upcoming Predictions</h2>
            <UpcomingGamesGrid 
              games={gameResults.upcoming_games}
              loading={loading}
            />
          </div>

          {/* Past 20 Games */}
          <div>
            <h2 className="text-3xl font-bold mb-4">📊 Recent Results</h2>
            <PastGamesTable 
              games={gameResults.recent_results}
              loading={loading}
            />
          </div>

          {/* System Info Footer */}
          <div className="mt-8 text-center text-slate-500 text-sm">
            <p>
              Last updated: {new Date(gameResults.metadata.last_updated).toLocaleString()} • 
              Status: {gameResults.metadata.sync_status} • 
              Model: XGBoost Elite v2.1
            </p>
          </div>
        </>
      )}

      {/* Empty State */}
      {!loading && !gameResults && !error && (
        <div className="text-center py-20">
          <div className="text-6xl mb-4">🏀</div>
          <h3 className="text-2xl font-bold text-slate-300 mb-2">No Data Available</h3>
          <p className="text-slate-500 mb-6">Click the refresh button to load predictions</p>
          <button
            onClick={loadDashboardData}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition"
          >
            Load Dashboard
          </button>
        </div>
      )}
    </div>
  );
};

export default DashboardPage;