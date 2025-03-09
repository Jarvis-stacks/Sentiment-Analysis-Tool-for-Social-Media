import React, { useState } from 'react';
import { exportAnalyses } from '../services/api';

const ExportAnalysis: React.FC = () => {
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handleExport = async () => {
    setMessage('');
    setError('');
    try {
      const token = localStorage.getItem('token') || '';
      const blob = await exportAnalyses(token);
      const url = window.URL.createObjectURL(new Blob([blob]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'analysis_history.csv');
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
      setMessage('Export successful! Check your downloads.');
    } catch (err) {
      setError('Failed to export analyses.');
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-8 max-w-md mx-auto">
      <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">Export Analysis</h2>
      <p className="text-gray-600 mb-6 text-center">
        Export your analysis history as a CSV file.
      </p>
      <button
        onClick={handleExport}
        className="w-full bg-green-600 text-white py-3 rounded-lg hover:bg-green-700 transition duration-300"
      >
        Export to CSV
      </button>
      {message && <p className="text-green-500 mt-4 text-center">{message}</p>}
      {error && <p className="text-red-500 mt-4 text-center">{error}</p>}
    </div>
  );
};

export default ExportAnalysis;
