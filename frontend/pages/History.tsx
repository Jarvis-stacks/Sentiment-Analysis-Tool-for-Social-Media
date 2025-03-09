import React, { useState, useEffect } from 'react';
import { getAnalyses } from '../services/api';

const History: React.FC = () => {
  const [analyses, setAnalyses] = useState<any[]>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchAnalyses = async () => {
      try {
        const token = localStorage.getItem('token') || '';
        const data = await getAnalyses(token);
        setAnalyses(data);
      } catch (err) {
        setError('Failed to fetch analysis history.');
      }
    };
    fetchAnalyses();
  }, []);

  return (
    <div className="bg-white rounded-lg shadow-lg p-8 max-w-4xl mx-auto">
      <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">Analysis History</h2>
      {analyses.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="min-w-full bg-gray-50 rounded-lg shadow-inner">
            <thead>
              <tr className="bg-gray-200">
                <th className="p-3 text-left text-sm font-medium text-gray-700">Text</th>
                <th className="p-3 text-left text-sm font-medium text-gray-700">Sentiment</th>
                <th className="p-3 text-left text-sm font-medium text-gray-700">Confidence</th>
                <th className="p-3 text-left text-sm font-medium text-gray-700">Model</th>
                <th className="p-3 text-left text-sm font-medium text-gray-700">Date</th>
              </tr>
            </thead>
            <tbody>
              {analyses.map((analysis) => (
                <tr key={analysis.id} className="border-t">
                  <td className="p-3 text-gray-600">{analysis.text}</td>
                  <td className="p-3 text-gray-600">{analysis.sentiment}</td>
                  <td className="p-3 text-gray-600">{(analysis.confidence * 100).toFixed(2)}%</td>
                  <td className="p-3 text-gray-600">{analysis.model_used}</td>
                  <td className="p-3 text-gray-600">{new Date(analysis.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-gray-600 text-center">No analysis history available.</p>
      )}
      {error && <p className="text-red-500 mt-4 text-center">{error}</p>}
    </div>
  );
};

export default History;
