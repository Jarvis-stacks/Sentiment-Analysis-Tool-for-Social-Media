import React, { useState } from 'react';
import { analyzeText } from '../services/api';

const Analyze: React.FC = () => {
  const [text, setText] = useState('');
  const [model, setModel] = useState('default');
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setResult(null);
    try {
      const token = localStorage.getItem('token') || '';
      const data = await analyzeText(text, model, token);
      setResult(data);
    } catch (err) {
      setError('Failed to analyze text. Please try again.');
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-8 max-w-2xl mx-auto">
      <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">Analyze Sentiment</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="text" className="block text-sm font-medium text-gray-700">Text to Analyze</label>
          <textarea
            id="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
            placeholder="Enter your text here..."
            rows={5}
            required
          />
        </div>
        <div>
          <label htmlFor="model" className="block text-sm font-medium text-gray-700">Model</label>
          <select
            id="model"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
          >
            <option value="default">Default</option>
            <option value="distilbert">DistilBERT</option>
            <option value="roberta">RoBERTa</option>
          </select>
        </div>
        <button
          type="submit"
          className="w-full bg-purple-600 text-white py-3 rounded-lg hover:bg-purple-700 transition duration-300"
        >
          Analyze
        </button>
      </form>
      {result && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg shadow-inner">
          <h3 className="text-xl font-semibold text-gray-800">Result</h3>
          <p className="mt-2">Sentiment: <span className="font-medium">{result.sentiment}</span></p>
          <p>Confidence: <span className="font-medium">{(result.confidence * 100).toFixed(2)}%</span></p>
          <p>Model Used: <span className="font-medium">{result.model_used}</span></p>
        </div>
      )}
      {error && <p className="text-red-500 mt-4 text-center">{error}</p>}
    </div>
  );
};

export default Analyze;
