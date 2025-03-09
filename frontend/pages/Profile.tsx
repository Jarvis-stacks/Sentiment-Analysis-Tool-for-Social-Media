import React, { useState, useEffect } from 'react';
import { getUserProfile, updateUserProfile } from '../services/api';

const Profile: React.FC = () => {
  const [profile, setProfile] = useState<any>(null);
  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const token = localStorage.getItem('token') || '';
        const data = await getUserProfile(token);
        setProfile(data);
        setEmail(data.email || '');
        setFullName(data.full_name || '');
      } catch (err) {
        setError('Failed to fetch profile.');
      }
    };
    fetchProfile();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    try {
      const token = localStorage.getItem('token') || '';
      const updatedProfile = await updateUserProfile({ email, full_name: fullName }, token);
      setProfile(updatedProfile);
      setSuccess('Profile updated successfully!');
    } catch (err) {
      setError('Failed to update profile.');
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-8 max-w-md mx-auto">
      <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">Profile</h2>
      {profile ? (
        <div className="space-y-6">
          <div>
            <p className="text-gray-700"><strong>Username:</strong> {profile.username}</p>
            <p className="text-gray-700"><strong>Last Login:</strong> {new Date(profile.last_login).toLocaleString()}</p>
          </div>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700">Email</label>
              <input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                placeholder="Update your email"
              />
            </div>
            <div>
              <label htmlFor="fullName" className="block text-sm font-medium text-gray-700">Full Name</label>
              <input
                type="text"
                id="fullName"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                placeholder="Update your full name"
              />
            </div>
            {success && <p className="text-green-500 text-center">{success}</p>}
            {error && <p className="text-red-500 text-center">{error}</p>}
            <button
              type="submit"
              className="w-full bg-indigo-600 text-white py-3 rounded-lg hover:bg-indigo-700 transition duration-300"
            >
              Update Profile
            </button>
          </form>
        </div>
      ) : (
        <p className="text-gray-600 text-center">Loading profile...</p>
      )}
    </div>
  );
};

export default Profile;
