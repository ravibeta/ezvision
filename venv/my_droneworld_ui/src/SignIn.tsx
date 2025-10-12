import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useUser } from "./UserContext";
import { API_ENDPOINT_URL } from './constants';
import { Link } from "react-router-dom";
const SIGNIN_ENDPOINT_URL = `${API_ENDPOINT_URL.replace(/\/$/, '')}/signin/`;


const SignIn: React.FC = () => {
  // const [email, setEmail] = useState("");
  const [inputEmail, setInputEmail] = useState("");
  const [inputPassword, setInputPassword] = useState("");
  const [emailError, setEmailError] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const { setUser } = useUser();
  const navigate = useNavigate();

  // Email regex (basic, not fully RFC compliant for brevity)
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  // Form validation function
  const validate = (): boolean => {
    let valid = true;
    setEmailError(null);
    setPasswordError(null);

    if (!emailRegex.test(inputEmail)) {
      setEmailError("Please enter a valid email address.");
      valid = false;
    }
    if (inputPassword.length < 6) {
      setPasswordError("Password must be at least 6 characters.");
      valid = false;
    }
    return valid;
  };
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccessMessage(null);

    if (!validate()) return;

    setLoading(true);
	
    const response = await fetch(SIGNIN_ENDPOINT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json",
          "Accept": "application/json" },
      body: JSON.stringify({ email: inputEmail, password: inputPassword }),
    });
    if (response.ok) {
      const data = await response.json();
      if (data.exists) {
          setSuccessMessage(data.message || "SignIn successful! You can now proceed to videos.");
          setInputEmail(inputEmail);
		  // setEmail(inputEmail);
          setInputPassword("");
          // setUser(User(inputEmail));
		  setUser({ email: inputEmail, id: data.id});
		  navigate("/videos", { replace: true });
		  // history.push("/videos");
      } else {
        setError("Invalid email or password");
      }
    } else {
      setError("Not Found. Login request failed");
    }
  };

  return (
    <div style={{ maxWidth: 400, margin: "40px auto", padding: 24, border: "1px solid #ddd", borderRadius: 8 }}>
      <h2>Sign In</h2>
      <form onSubmit={handleSubmit} autoComplete="off">
        {/* Email */}
        <div style={{ marginBottom: 16 }}>
          <label htmlFor="email" style={{ display: "block", fontWeight: 500 }}>
            Email
          </label>
          <input
            id="email"
            type="email"
            autoComplete="username"
            value={inputEmail}
            onChange={(e) => setInputEmail(e.target.value)}
            required
            style={{ width: "100%", padding: 8, marginBottom: 4 }}
          />
          {emailError && <div style={{ color: "red", fontSize: 13 }}>{emailError}</div>}
        </div>

        {/* Password */}
        <div style={{ marginBottom: 16 }}>
          <label htmlFor="password" style={{ display: "block", fontWeight: 500 }}>
            Password
          </label>
          <input
            id="password"
            type="password"
            autoComplete="new-password"
            value={inputPassword}
            onChange={(e) => setInputPassword(e.target.value)}
            required
            style={{ width: "100%", padding: 8, marginBottom: 4 }}
          />
          {passwordError && <div style={{ color: "red", fontSize: 13 }}>{passwordError}</div>}
        </div>

        <button
          type="submit"
          style={{
            padding: "10px 20px",
            fontWeight: 600,
            background: "#007bff",
            color: "#fff",
            border: "none",
            borderRadius: 4,
            cursor: loading ? "not-allowed" : "pointer"
          }}
          disabled={loading}
        >
          {loading ? "Signing in..." : "Sign In"}
        </button>
        {error && <div style={{ color: "red", marginTop: 12 }}>{error}</div>}
        {successMessage && <div style={{ color: "green", marginTop: 12 }}>{successMessage}</div>}
		<div>Don't have an account? <Link to="/register" style={{ color: "gray" }}>Register</Link></div>
      </form>
    </div>
  );
  
};

export default SignIn;
