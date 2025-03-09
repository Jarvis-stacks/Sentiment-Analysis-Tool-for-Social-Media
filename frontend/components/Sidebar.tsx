import React from 'react';
import { Link } from 'react-router-dom';

const Sidebar: React.FC = () => {
  const token = localStorage.getItem('token');

  return (
    <aside className="w-64 bg-gray-900 text-white p-6 hidden md:block shadow-lg">
      <h2 className="text-xl font-semibold mb-6">Navigation</h2>
      <ul className="space-y-4">
        <li><Link to="/" className="block hover:bg-gray-700 p-2 rounded transition duration-300">Home</Link></li>
        {token && (
          <>
            <li><Link to="/analyze" className="block hover:bg-gray-700 p-2 rounded transition duration-300">Analyze</Link></li>
            <li><Link to="/batch" className="block hover:bg-gray-700 p-2 rounded transition duration-300">Batch Analyze</Link></li>
            <li><Link to="/history" className="block hover:bg-gray-700 p-2 rounded transition duration-300">History</Link></li>
            <li><Link to="/profile" className="block hover:bg-gray-700 p-2 rounded transition duration-300">Profile</Link></li>
            <li><Link to="/model-comparison" className="block hover:bg-gray-700 p-2 rounded transition duration-300">Model Comparison</Link></li>
<li><Link to="/text-statistics" className="block hover:bg-gray-700 p-2 rounded transition duration-300">Text Statistics</Link></li>
<li><Link to="/export-analysis" className="block hover:bg-gray-700 p-2 rounded transition duration-300">Export Analysis</Link></li>
          </>
        )}
        {!token && (
          <>
            <li><Link to="/login" className="block hover:bg-gray-700 p-2 rounded transition duration-300">Login</Link></li>
            <li><Link to="/register" className="block hover:bg-gray-700 p-2 rounded transition duration-300">Register</Link></li>
          </>
        )}
      </ul>
    </aside>
  );
};

export default Sidebar;
