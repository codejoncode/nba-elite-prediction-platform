import React from "react";

/**
 * StatsCards Component
 * Displays 4 key metrics: accuracy, record, last 20, and last updated
 * 
 * Props:
 *   metadata (object) - Contains accuracy stats and metadata
 * 
 * Returns: Grid of 4 stat cards
 */
const StatsCards = ({ metadata }) => {
  if (!metadata) return null;

  const cardData = [
    {
      title: "📊 Model Accuracy",
      value: metadata.accuracy_all_time_percent?.toFixed(2) || "0",
      unit: "%",
      subtitle: "All-time XGBoost Performance",
      color: "from-emerald-500/20 to-green-500/20",
      border: "border-emerald-500/50"
    },
    {
      title: "🏆 Record",
      value: metadata.correct_predictions,
      unit: `/${metadata.correct_predictions + metadata.incorrect_predictions}`,
      subtitle: `${metadata.incorrect_predictions} Losses`,
      color: "from-blue-500/20 to-cyan-500/20",
      border: "border-blue-500/50"
    },
    {
      title: "📈 Last 20 Games",
      value: metadata.accuracy_last_20_percent?.toFixed(1) || "0",
      unit: "%",
      subtitle: "Recent Performance",
      color: "from-purple-500/20 to-pink-500/20",
      border: "border-purple-500/50"
    },
    {
      title: "⏱️ Last Updated",
      value: metadata.last_updated ? new Date(metadata.last_updated).toLocaleDateString() : "N/A",
      unit: "",
      subtitle: "Database sync",
      color: "from-amber-500/20 to-orange-500/20",
      border: "border-amber-500/50"
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
      {cardData.map((card, idx) => (
        <div
          key={idx}
          className={`bg-gradient-to-br ${card.color} border-2 ${card.border} p-8 rounded-3xl backdrop-blur-sm shadow-2xl`}
        >
          <div className="text-slate-100 text-sm font-semibold uppercase mb-3">
            {card.title}
          </div>
          <div className="text-5xl font-black text-white mb-2">
            {card.value}<span className="text-2xl">{card.unit}</span>
          </div>
          <div className="text-slate-300 text-sm">{card.subtitle}</div>
        </div>
      ))}
    </div>
  );
};

export default StatsCards;