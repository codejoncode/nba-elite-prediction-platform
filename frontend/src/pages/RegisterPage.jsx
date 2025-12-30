import React from 'react'
import { Navigate } from 'react-router-dom'

export const RegisterPage = () => {
  // Google OAuth handles registration automatically
  return <Navigate to="/login" replace />
}
