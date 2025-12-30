import { createContext } from "react";

export const AuthContext = createContext({
  user: null,
  token: null,
  loading: false,
  error: null,
  
  // Methods
  register: async () => {},
  login: async () => {},
  logout: async () => {},
  changePassword: async () => {},
  deleteAccount: async () => {},
  
  // Computed
  isAuthenticated: false
});
