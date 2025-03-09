import React, { useState } from 'react';
import { batchAnalyze } from '../services/api';

const BatchAnalyze: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState('default');
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError('Please upload a CSV file.');
      return;
    }
    setError('');
    setResults([]);
    try {
      const token = localStorage.getItem('token') || '';
      const formData = new FormData();
      formData.append('file', file);
      formData.append('model', model);
      const data = await batchAnalyze(formData, token);
      setResults(data);
    } catch (err) {
      setError('Failed to process batch analysis.');
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-8 max-w-4xl mx-auto">
      <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">Batch Analyze</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="file" className="block text-sm font-medium text-gray-700">Upload CSV File</label>
          <input
            type="file"
            id="file"
            accept=".csv"
            onChange={handleFileChange}
            className="w-full p-3 border border-gray-300 rounded-lg"
          />
          <p className="text-sm text-gray-500 mt-1">CSV should contain a column with text to analyze.</p>
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
          Analyze Batch
        </button>
      </form>
      {results.length > 0 && (
        <div className="mt-6">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Results</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-gray-50 rounded-lg shadow-inner">
              <thead>
                <tr className="bg-gray-200">
                  <th className="p-3 text-left text-sm font-medium text-gray-700">Text</th>
                  <th className="p-3 text-left text-sm font-medium text-gray-700">Sentiment</th>
                  <th className="p-3 text-left text-sm font-medium text-gray-700">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result, index) => (
                  <tr key={index} className="border-t">
                    <td className="p-3 text-gray-600">{result.text}</td>
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

export default BatchAnalyze;
