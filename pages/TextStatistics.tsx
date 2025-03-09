import React, { useState, useEffect } from 'react';
import { getTextStats } from '../services/api';

const TextStatistics: React.FC = () => {
  const [stats, setStats] = useState<any>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const token = localStorage.getItem('token') || '';
        const data = await getTextStats(token);
        setStats(data);
      } catch (err) {
        setError('Failed to fetch text statistics.');
      }
    };
    fetchStats();
  }, []);

  return (
    <div className="bg-white rounded-lg shadow-lg p-8 max-w-3xl mx-auto">
      <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">Text Statistics</h2>
      <p className="text-gray-600 mb-6 text-center">
        View statistics about your analyzed texts.
      </p>
      {stats ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-50 p-4 rounded-lg shadow-inner">
              <h3 className="text-lg font-semibold text-gray-800">Total Analyses</h3>
              <p className="text-2xl text-indigo-600">{stats.total_analyses}</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg shadow-inner">
              <h3 className="text-lg font-semibold text-gray-800">Average Word Count</h3>
              <p className="text-2xl text-indigo-600">{stats.avg_word_count.toFixed(2)}</p>
            </div>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg shadow-inner">
            <h3 className="text-lg font-semibold text-gray-800 mb-2">Sentiment Distribution</h3>
            <ul className="space-y-2">
              {Object.entries(stats.sentiment_distribution).map(([sentiment, count]) => (
                <li key={sentiment} className="flex justify-between">
                  <span className="text-gray-700">{sentiment}</span>
                  <span className="text-indigo-600">{count as number}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      ) : (
        <p className="text-gray-600 text-center">Loading statistics...</p>
      )}
      {error && <p className="text-red-500 mt-4 text-center">{error}</p>}
    </div>
  );
};

export default TextStatistics;
