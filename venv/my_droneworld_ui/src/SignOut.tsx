// src/pages/SignOut.tsx
import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useUser } from "./UserContext";

const SignOut: React.FC = () => {
  const { setUser } = useUser();
  const navigate = useNavigate();

  useEffect(() => {
    // Clear context
    setUser(null);

    // Clear localStorage (optional, since UserProvider handles it, but we do extra safety here)
    localStorage.removeItem("user");

    // Redirect to home or signin
    navigate("/signin", { replace: true });
  }, [setUser, navigate]);

  return <p>Signing you out...</p>;
};

export default SignOut;
