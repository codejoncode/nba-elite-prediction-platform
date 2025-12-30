import React from 'react';

const NBA_TEAMS = [
  'Lakers', 'Celtics', 'Warriors', 'Suns', 'Mavericks', 'Grizzlies',
  'Heat', 'Bucks', 'Nuggets', 'Kings', 'Clippers', 'Timberwolves',
  '76ers', 'Cavaliers', 'Pacers', 'Nets', 'Knicks', 'Hawks',
  'Raptors', 'Pistons', 'Hornets', 'Spurs', 'Rockets', 'Pelicans',
  'Blazers', 'Pistons', 'Bulls', 'Jayz', 'Magic', 'Rockets'
].sort();

export const TeamSelector = ({ label, value, onChange, disabled = false }) => {
  return (
    <div>
      <label className="block text-sm font-semibold text-gray-200 mb-2">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="w-full bg-gray-700 border border-gray-600 rounded px-4 py-2 text-white focus:outline-none focus:border-blue-500"
      >
        <option value="">Select a team</option>
        {NBA_TEAMS.map(team => (
          <option key={team} value={team}>{team}</option>
        ))}
      </select>
    </div>
  );
};
