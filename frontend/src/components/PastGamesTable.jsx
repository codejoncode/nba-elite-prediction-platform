import React from "react";

const PastGamesTable = ({ games, loading }) => {
  if (loading) return null;
  if (!games || games.length === 0) {
    return (
      <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-12 shadow-2xl border border-white/20">
        <h2 className="text-4xl font-bold text-white mb-8">📋 Past Games - Accuracy</h2>
        <div className="text-center py-16 text-white/50 text-lg">
          No completed games yet
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-12 shadow-2xl border border-white/20">
      <h2 className="text-4xl font-bold text-white mb-8">📋 Past 20 Games - Accuracy</h2>

      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-white/20">
              <th className="px-6 py-3 text-sm font-semibold text-slate-300">Date</th>
              <th className="px-6 py-3 text-sm font-semibold text-slate-300">Away</th>
              <th className="px-6 py-3 text-sm font-semibold text-slate-300">Home</th>
              <th className="px-6 py-3 text-sm font-semibold text-slate-300">Score</th>
              <th className="px-6 py-3 text-sm font-semibold text-slate-300">Winner</th>
              <th className="px-6 py-3 text-sm font-semibold text-slate-300">Predicted</th>
              <th className="px-6 py-3 text-sm font-semibold text-slate-300">Result</th>
            </tr>
          </thead>
          <tbody>
            {games.map((game, idx) => (
              <tr key={idx} className="border-b border-white/10 hover:bg-white/5 transition">
                <td className="px-6 py-4 text-sm text-slate-300">
                  {new Date(game.date).toLocaleDateString()}
                </td>
                <td className="px-6 py-4 text-sm text-white font-semibold">{game.away_team}</td>
                <td className="px-6 py-4 text-sm text-white font-semibold">{game.home_team}</td>
                <td className="px-6 py-4 text-sm text-slate-400">{game.score}</td>
                <td className="px-6 py-4 text-sm text-emerald-400 font-semibold">
                  {game.actual_winner}
                </td>
                <td className="px-6 py-4 text-sm text-amber-300">{game.predicted}</td>
                <td className="px-6 py-4 text-sm font-bold">
                  {game.result === "win" ? (
                    <span className="text-emerald-400">✓ Win</span>
                  ) : (
                    <span className="text-red-400">✗ Loss</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PastGamesTable;