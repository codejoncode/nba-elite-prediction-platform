import React from 'react';
import { v4 as uuid4 } from 'uuid';

const NBA_TEAMS = [
  'Lakers', 'Celtics', 'Warriors', 'Suns', 'Mavericks', 'Grizzlies',
  'Heat', 'Bucks', 'Nuggets', 'Kings', 'Clippers', 'Timberwolves',
  '76ers', 'Cavaliers', 'Pacers', 'Nets', 'Knicks', 'Hawks',
  'Raptors', 'Pistons', 'Hornets', 'Spurs', 'Rockets', 'Pelicans',
  'Blazers', 'Pistons', 'Bulls', 'Jayz', 'Magic', 'Rockets'
].sort();

export const TeamSelector = ({ id, label, value, onChange, disabled = false }) => {
  if (!id) {
    throw new Error("TeamSelector component requires a unique 'id' prop.");
  }

  const getOptionKey = (teamName, selectorId) => {
    return `${teamName}-${selectorId}-${uuid4().slice(0, 8)}`;
  }

  return (
    <div>
      <label className="block text-sm font-semibold text-gray-200 mb-2">{label}</label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="w-full bg-gray-700 border border-gray-600 rounded px-4 py-2 text-white focus:outline-none focus:border-blue-500"
      >
        <option value="">Select a team</option>
        {NBA_TEAMS.map(team => (
          <option key={getOptionKey(team, id)} value={team}>{team}</option>
        ))}
      </select>
    </div>
  );
};
