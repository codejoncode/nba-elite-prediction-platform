import React, { useState, useCallback, useEffect } from 'react'
import { jwtDecode } from 'jwt-decode'
import { AuthContext } from './AuthContext'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001'

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(localStorage.getItem('token') || null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Decode token to get user info
  useEffect(() => {
    if (token) {
      try {
        const decoded = jwtDecode(token)
        setUser({
          username: decoded.username || decoded.sub,
          email: decoded.email,
          name: decoded.name || decoded.username,
          picture: decoded.picture
        })
      } catch (err) {
        console.error('Token decode error:', err)
        localStorage.removeItem('token')
        setToken(null)
        setUser(null)
      }
    }
  }, [token])

  const loginWithGoogle = useCallback(async (credentialResponse) => {
    console.log('loginWithGoogle called') // Debug
    try {
      setLoading(true)
      setError(null)

      const googleToken = credentialResponse.credential
      
      const response = await fetch(`${API_URL}/auth/google-login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          id_token: googleToken
        })
      })

      const data = await response.json()
      console.log('Backend response:', data) // Debug

      if (!response.ok) {
        setError(data.error || 'Google login failed')
        return false
      }

      // Success - save token and user
      setUser(data.user)
      setToken(data.token)
      localStorage.setItem('token', data.token)
      
      console.log('Login successful, token saved') // Debug
      return true
    } catch (err) {
      const errorMsg = err.message || 'Google login error'
      setError(errorMsg)
      console.error('Google login error:', err)
      return false
    } finally {
      setLoading(false)
    }
  }, [])

  const logout = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)

      if (token) {
        try {
          await fetch(`${API_URL}/auth/logout`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${token}`
            }
          })
        } catch (err) {
          console.warn('Logout endpoint error:', err)
        }
      }

      setUser(null)
      setToken(null)
      localStorage.removeItem('token')
      return true
    } catch (err) {
      console.error('Logout error:', err)
      return false
    } finally {
      setLoading(false)
    }
  }, [token])

  const value = {
    // State
    user,
    token,
    loading,
    error,
    
    // Methods - THIS WAS MISSING!
    loginWithGoogle,
    logout,
    
    // Computed
    isAuthenticated: !!token && !!user
  }

  console.log('AuthProvider value:', value) // Debug

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}
