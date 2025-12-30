import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { GoogleLogin } from '@react-oauth/google';

export const LoginPage = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  // ✅ Check if already logged in - use useEffect
  useEffect(() => {
    const token = localStorage.getItem('token');
    const user = localStorage.getItem('user');
    
    if (token && user) {
      navigate('/dashboard');
    }
  }, [navigate]);

  const handleGoogleLoginSuccess = async (credentialResponse) => {
    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:5001/auth/google-login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          id_token: credentialResponse.credential,
        }),
      });

      const data = await response.json();

      if (data.success) {
        localStorage.setItem('token', data.token);
        localStorage.setItem('user', JSON.stringify(data.user));
        navigate('/dashboard');
      } else {
        setError(data.error || 'Login failed');
      }
    } catch (err) {
      console.error('Login error:', err);
      setError('Network error. Please try again.');
    }

    setLoading(false);
  };

  const handleGoogleLoginError = () => {
    setError('Google login failed. Please try again.');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-12 shadow-2xl border border-white/20">
          <div className="text-center mb-12">
            <h1 className="text-5xl font-black text-white mb-4 drop-shadow-lg">
              🏀 NBA Elite
            </h1>
            <h2 className="text-2xl font-bold text-blue-200 mb-2">
              Predictor
            </h2>
            <p className="text-blue-300 text-lg">
              74.73% Accuracy | Powered by XGBoost
            </p>
          </div>

          {error && (
            <div className="mb-8 p-4 bg-red-500/20 border border-red-500/50 rounded-2xl">
              <p className="text-red-200 text-center font-semibold">{error}</p>
            </div>
          )}

          <div className="space-y-6">
            <div>
              <h3 className="text-white font-bold text-lg mb-6 text-center">
                🔐 Sign In with Google
              </h3>
              <div className="flex justify-center">
                <GoogleLogin
                  onSuccess={handleGoogleLoginSuccess}
                  onError={handleGoogleLoginError}
                  text="signin_with"
                  size="large"
                  theme="dark"
                />
              </div>
            </div>

            {loading && (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-4 border-white border-t-blue-500 mx-auto mb-4"></div>
                <p className="text-white font-semibold">Signing in...</p>
              </div>
            )}
          </div>

          <div className="mt-12 pt-8 border-t border-white/20">
            <p className="text-blue-200 text-sm text-center leading-relaxed">
              🔒 Your login is secure. We use OAuth 2.0 with Google authentication.
              No passwords are stored on our servers.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

