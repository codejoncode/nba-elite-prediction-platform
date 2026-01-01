import React from "react";

const UpcomingGamesGrid = ({ games, loading }) => {
  if (loading) return null;
  if (!games || games.length === 0) {
    return (
      <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-12 shadow-2xl border border-white/20 mb-12">
        <h2 className="text-4xl font-bold text-white mb-8">🎯 Upcoming Games</h2>
        <div className="text-center py-16 text-white/50 text-lg">
          No upcoming games scheduled
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-12 shadow-2xl border border-white/20 mb-12">
      <h2 className="text-4xl font-bold text-white mb-8">🎯 Upcoming Games - AI Predictions</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        {games.map((game, idx) => (
          <div
            key={game.id || idx}
            className="bg-gradient-to-br from-blue-500/20 to-indigo-500/20 border-2 border-blue-500/50 p-6 rounded-2xl backdrop-blur-sm shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:scale-105"
          >
            {/* Date */}
            <div className="text-blue-200 text-xs font-semibold uppercase mb-3">
              {new Date(game.date).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              })}
            </div>

            {/* Teams */}
            <div className="text-white font-bold text-lg mb-4">
              <div className="text-sm text-blue-300">{game.away_team}</div>
              <div className="text-center text-blue-100 text-xs my-2">VS</div>
              <div className="text-sm text-green-300">{game.home_team}</div>
            </div>

            {/* Prediction */}
            <div className="bg-black/20 rounded-xl p-4 mb-4 text-center">
              <div className="text-yellow-300 text-xs font-semibold mb-2">
                🤖 PREDICTION
              </div>
              <div className="text-white font-black text-2xl mb-1">
                {game.predicted === "HOME" ? game.home_team : game.away_team}
              </div>
              <div className="text-emerald-300 text-sm font-bold">
                {(game.confidence || 65).toFixed(1)}%
              </div>
            </div>

            {/* Status */}
            <div className="text-center text-blue-300 text-xs">
              ⏱️ Scheduled
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default UpcomingGamesGrid;
