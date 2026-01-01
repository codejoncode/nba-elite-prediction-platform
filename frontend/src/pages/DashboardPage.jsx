import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";

// Import modular components
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

  // State management - clean and minimal
  const [gameResults, setGameResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastRefresh, setLastRefresh] = useState(null);

  // Load dashboard data
  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Call Flask game_results endpoint (no auth needed)
      const response = await fetch("http://localhost:5001/api/game_results");

      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }

      const data = await response.json();
      setGameResults(data);
      setLastRefresh(new Date());

    } catch (err) {
      setError(err.message || "Failed to load dashboard data");
      console.error("Dashboard data error:", err);
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;

    // Require authentication
    if (!user) {
      navigate("/login");
      return;
    }

    loadDashboardData();

  }, [navigate, user]);

  // Manual refresh handler
  const handleManualRefresh = async () => {
    await loadDashboardData();
  };

  // Auto-refresh every 5 minutes (optional)
  useEffect(() => {
    const interval = setInterval(loadDashboardData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  // Not authenticated
  if (!user) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-8">
      
      {/* Header */}
      <div className="mb-12">
        <h1 className="text-5xl font-black mb-2">🏀 NBA Elite Predictions</h1>
        <p className="text-slate-400">Real-time model performance & upcoming game forecasts</p>
      </div>

      {/* Controls - Manual Refresh + Status */}
      <DashboardControls 
        onRefresh={handleManualRefresh}
        loading={loading}
        lastRefresh={lastRefresh}
      />

      {/* Error Notification */}
      {error && <ErrorNotification error={error} onDismiss={() => setError(null)} />}

      {/* Loading Spinner */}
      {loading && <LoadingSpinner />}

      {/* Dashboard Content */}
      {!loading && gameResults && (
        <>
          {/* Stats Cards - FIXED: removed unused recentResults prop */}
          <StatsCards 
            metadata={gameResults.metadata}
          />

          {/* Upcoming Games */}
          <UpcomingGamesGrid 
            games={gameResults.upcoming_games}
            loading={loading}
          />

          {/* Past 20 Games */}
          <PastGamesTable 
            games={gameResults.recent_results}
            loading={loading}
          />
        </>
      )}
    </div>
  );
};

export default DashboardPage;