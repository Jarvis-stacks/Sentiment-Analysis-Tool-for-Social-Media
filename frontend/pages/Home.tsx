import React from 'react';
import { Link } from 'react-router-dom';

const Home: React.FC = () => {
  return (
    <div className="bg-white rounded-lg shadow-lg p-8 max-w-3xl mx-auto">
      <h1 className="text-4xl font-bold text-gray-800 mb-6 text-center">Welcome to Sentiment Analyzer</h1>
      <p className="text-lg text-gray-600 mb-8 text-center">
        Analyze the sentiment of your text with cutting-edge AI models. Whether it’s a single sentence or a batch of texts, we’ve got you covered!
      </p>
      <div className="flex justify-center space-x-6">
        <Link to="/login" className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition duration-300">
          Login
        </Link>
        <Link to="/register" className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition duration-300">
          Register
        </Link>
        <Link to="/analyze" className="bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700 transition duration-300">
          Start Analyzing
        </Link>
      </div>
    </div>
  );
};

export default Home;
