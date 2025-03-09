import React from 'react';
import { Link, useNavigate } from 'react-router-dom';

const Header: React.FC = () => {
  const navigate = useNavigate();
  const token = localStorage.getItem('token');

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/login');
  };

  return (
    <header className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-4 shadow-lg">
      <div className="container mx-auto flex justify-between items-center">
        <Link to="/" className="text-2xl font-bold tracking-tight">Sentiment Analyzer</Link>
        <nav className="space-x-6">
          {token ? (
            <>
              <Link to="/analyze" className="hover:text-indigo-200 transition duration-300">Analyze</Link>
              <Link to="/batch" className="hover:text-indigo-200 transition duration-300">Batch Analyze</Link>
              <Link to="/history" className="hover:text-indigo-200 transition duration-300">History</Link>
              <Link to="/profile" className="hover:text-indigo-200 transition duration-300">Profile</Link>
              <button onClick={handleLogout} className="bg-red-500 hover:bg-red-600 px-4 py-2 rounded-lg transition duration-300">
                Logout
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="hover:text-indigo-200 transition duration-300">Login</Link>
              <Link to="/register" className="hover:text-indigo-200 transition duration-300">Register</Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
};

export default Header;
