import React from 'react';

export const MetricsCard = ({ icon: Icon, label, value, description }) => {
  const bgClass = {
    'ACCURACY': 'from-green-900 to-green-800',
    'ROC-AUC': 'from-blue-900 to-blue-800',
    'SENSITIVITY': 'from-purple-900 to-purple-800',
    'BEST ITERATION': 'from-orange-900 to-orange-800'
  }[label] || 'from-gray-900 to-gray-800';

  const colorClass = {
    'ACCURACY': 'text-green-400',
    'ROC-AUC': 'text-blue-400',
    'SENSITIVITY': 'text-purple-400',
    'BEST ITERATION': 'text-orange-400'
  }[label] || 'text-gray-400';

  return (
    <div className={`bg-gradient-to-br ${bgClass} p-6 rounded-lg border border-opacity-50`}>
      <div className="flex items-center gap-2 mb-2">
        {Icon && <Icon className="w-5 h-5" />} {/* ✅ Icon IS used here */}
        <span className="text-sm font-semibold uppercase tracking-wide">{label}</span>
      </div>
      <div className={`text-4xl font-bold ${colorClass}`}>{value}</div>
      <div className="text-xs opacity-75 mt-2">{description}</div>
    </div>
  );
};