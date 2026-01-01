import React from "react";

const LoadingSpinner = () => {
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-slate-800/90 p-12 rounded-3xl border border-white/20 shadow-2xl text-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-emerald-500 mx-auto mb-6"></div>
        <p className="text-white text-lg font-semibold">Updating predictions...</p>
        <p className="text-slate-400 text-sm mt-2">This may take a moment</p>
      </div>
    </div>
  );
};

export default LoadingSpinner;
