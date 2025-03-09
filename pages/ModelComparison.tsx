import React, { useState } from 'react';
import { compareModels } from '../services/api';

const ModelComparison: React.FC = () => {
  const [text, setText] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setResults([]);
    try {
      const token = localStorage.getItem('token') || '';
      const data = await compareModels(text, token);
      setResults(data);
    } catch (err) {
      setError('Failed to compare models. Please try again.');
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-8 max-w-3xl mx-auto">
      <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">Model Comparison</h2>
      <p className="text-gray-600 mb-6 text-center">
        Compare how different models analyze the sentiment of your text.
      </p>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="text" className="block text-sm font-medium text-gray-700">Text to Compare</label>
          <textarea
            id="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
            placeholder="Enter text to compare across models..."
            rows={5}
            required
          />
        </div>
        <button
          type="submit"
          className="w-full bg-indigo-600 text-white py-3 rounded-lg hover:bg-indigo-700 transition duration-300"
        >
          Compare Models
        </button>
      </form>
      {results.length > 0 && (
        <div className="mt-6">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Comparison Results</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-gray-50 rounded-lg shadow-inner">
              <thead>
                <tr className="bg-gray-200">
                  <th className="p-3 text-left text-sm font-medium text-gray-700">Model</th>
                  <th className="p-3 text-left text-sm font-medium text-gray-700">Sentiment</th>
                  <th className="p-3 text-left text-sm font-medium text-gray-700">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result, index) => (
                  <tr key={index} className="border-t">
                    <td className="p-3 text-gray-600">{result.model}</td>
                    <td className="p-3 text-gray-600">{result.sentiment}</td>
                    <td className="p-3 text-gray-600">{(result.confidence * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {error && <p className="text-red-500 mt-4 text-center">{error}</p>}
    </div>
  );
};

export default ModelComparison;
