import React from "react";

const ErrorNotification = ({ error, onDismiss }) => {
  return (
    <div className="bg-red-500/20 border-2 border-red-500/50 text-red-100 px-6 py-4 rounded-2xl mb-8 backdrop-blur-sm">
      <div className="flex items-start justify-between">
        <div>
          <div className="font-bold mb-2">⚠️ Error</div>
          <p className="text-sm">{error}</p>
        </div>
        <button
          onClick={onDismiss}
          className="text-red-300 hover:text-red-100 ml-4"
        >
          ✕
        </button>
      </div>
    </div>
  );
};

export default ErrorNotification;