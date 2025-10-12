import React, { createContext, useEffect, useState, useContext, ReactNode, Dispatch, SetStateAction  } from 'react';

// Define the User type
// export type User = { email: string; id?: number } | null;

// usercontext.tsx
export interface User {
  email: string;
  id?: number
}

export interface UserContextType {
  user: User | null; // User can be null initially
  setUser: React.Dispatch<SetStateAction<User | null>>; // The setter function
}

// 3. Create the Context with a default value of null, and specify the type
export const UserContext = createContext<UserContextType | null>(null);

// 4. Create a custom hook to consume the context
export const useUser = () => {
  const context = useContext(UserContext);
  if (context === null) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};

// 5. Create a Provider component to manage the user state and provide it to the context
interface UserProviderProps {
  children: React.ReactNode;
}

export const UserProvider: React.FC<UserProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null); // Initial user state is null
  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  useEffect(() => {
    if (user) {
      localStorage.setItem("user", JSON.stringify(user));
    } else {
      localStorage.removeItem("user");
    }
  }, [user]);

  return (
    <UserContext.Provider value={{ user, setUser }}>
      {children}
    </UserContext.Provider>
  );
};


/*
// src/context/UserContext.tsx
import React, {
  createContext,
  useEffect,
  useState,
  useContext,
  ReactNode,
  Dispatch,
  SetStateAction
} from "react";

export interface User {
  email: string;
  id?: number;
}

export interface UserContextType {
  user: User | null;
  setUser: Dispatch<SetStateAction<User | null>>;
}

export const UserContext = createContext<UserContextType | null>(null);

export const useUser = () => {
  const context = useContext(UserContext);
  if (context === null) {
    throw new Error("useUser must be used within a UserProvider");
  }
  return context;
};

interface UserProviderProps {
  children: ReactNode;
}

export const UserProvider: React.FC<UserProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  useEffect(() => {
    if (user) {
      localStorage.setItem("user", JSON.stringify(user));
    } else {
      localStorage.removeItem("user");
    }
  }, [user]);

  return (
    <UserContext.Provider value={{ user, setUser }}>
      {children}
    </UserContext.Provider>
  );
};
*/