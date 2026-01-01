import React from "react";

const DashboardControls = ({ onRefresh, loading, lastRefresh }) => {
  return (
    <div className="flex items-center justify-between mb-8 pb-8 border-b border-white/20">
      <div className="text-slate-400 text-sm">
        {lastRefresh ? (
          <>
            Last updated: {lastRefresh.toLocaleTimeString()}
          </>
        ) : (
          "Loading data..."
        )}
      </div>

      <button
        onClick={onRefresh}
        disabled={loading}
        className={`px-6 py-3 rounded-xl font-semibold flex items-center gap-2 transition-all duration-300 ${
          loading
            ? "bg-slate-600 text-slate-300 cursor-not-allowed opacity-50"
            : "bg-emerald-600 hover:bg-emerald-700 text-white shadow-lg hover:shadow-emerald-500/50"
        }`}
      >
        {loading ? (
          <>
            <span className="animate-spin">⏳</span>
            Refreshing...
          </>
        ) : (
          <>
            <span>🔄</span>
            Check for Updates
          </>
        )}
      </button>
    </div>
  );
};

export default DashboardControls;