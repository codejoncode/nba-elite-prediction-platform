import React from 'react'
import { useNavigate } from 'react-router-dom'
import { GoogleLogin } from '@react-oauth/google'
import { useAuth } from '../hooks/useAuth'

export const LoginPage = () => {
  const { loginWithGoogle, loading, error, isAuthenticated } = useAuth()
  const navigate = useNavigate()

  const handleGoogleSuccess = async (credentialResponse) => {
    console.log('Google success callback') // Debug log
    
    try {
      const success = await loginWithGoogle(credentialResponse)
      console.log('loginWithGoogle result:', success) // Debug log
      
      if (success) {
        console.log('Navigating to dashboard') // Debug log
        setTimeout(() => {
          navigate('/dashboard', { replace: true })
        }, 100) // Small delay for state update
      }
    } catch (err) {
      console.error('Navigation error:', err)
    }
  }

  const handleGoogleError = (error) => {
    console.error('Google login error:', error)
  }

  // Auto-redirect if already authenticated
  if (isAuthenticated && !loading) {
    console.log('Already authenticated, redirecting...')
    navigate('/dashboard', { replace: true })
    return null
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 flex items-center justify-center px-4 py-12">
      <div className="bg-gray-800/80 backdrop-blur-xl p-12 rounded-2xl w-full max-w-md border border-gray-700 shadow-2xl">
        <div className="text-center mb-10">
          <div className="text-6xl mb-6 mx-auto w-24 h-24 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center shadow-lg">
            🏀
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-white to-gray-200 bg-clip-text text-transparent mb-3">
            NBA Elite
          </h1>
          <p className="text-xl text-gray-300">AI Game Predictions</p>
        </div>

        {error && (
          <div className="bg-red-500/90 backdrop-blur-sm text-white p-4 rounded-xl mb-8 border border-red-400 text-sm animate-pulse">
            {error}
          </div>
        )}

        <div className="space-y-6">
          <div className="relative">
            <GoogleLogin
              onSuccess={handleGoogleSuccess}
              onError={handleGoogleError}
              theme="filled_blue"
              size="large"
              text="signin_with"
              shape="rectangular"
              width="100%"
              disabled={loading}
            />
          </div>
        </div>

        <div className="mt-10 pt-8 border-t border-gray-700">
          <div className="grid grid-cols-2 gap-4 text-sm text-gray-400">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span>No passwords stored</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span>Google secure</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
