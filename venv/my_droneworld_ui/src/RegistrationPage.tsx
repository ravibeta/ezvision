import React, { useState } from "react";
import { Link } from "react-router-dom";
import { API_ENDPOINT_URL } from './constants';
const REGISTER_V1_ENDPOINT_URL = `${API_ENDPOINT_URL.replace(/\/$/, '')}/register/`;

interface RegistrationResponse {
  success?: boolean;
  message?: string;
  [key: string]: any; // For extensibility
}

const RegistrationPage: React.FC = () => {
  // Form state
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [passwordAgain, setPasswordAgain] = useState("");

  // Validation/Error state
  const [emailError, setEmailError] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [passwordAgainError, setPasswordAgainError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Email regex (basic, not fully RFC compliant for brevity)
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  // Form validation function
  const validate = (): boolean => {
    let valid = true;
    setEmailError(null);
    setPasswordError(null);
    setPasswordAgainError(null);

    if (!emailRegex.test(email)) {
      setEmailError("Please enter a valid email address.");
      valid = false;
    }
    if (password.length < 6) {
      setPasswordError("Password must be at least 6 characters.");
      valid = false;
    }
    if (password !== passwordAgain) {
      setPasswordAgainError("Passwords do not match.");
      valid = false;
    }
    return valid;
  };

  // Handler
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitError(null);
    setSuccessMessage(null);

    if (!validate()) return;

    setLoading(true);

    try {
      const response = await fetch(REGISTER_V1_ENDPOINT_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({ email, password })
      });

      if (response.ok) {
        const data: RegistrationResponse = await response.json();
        if (data.success) {
          setSuccessMessage(data.message || "Registration successful! You can now log in.");
          setEmail("");
          setPassword("");
          setPasswordAgain("");
        } else {
          setSubmitError(data.message || "Registration failed.");
        }
      } else {
        let details = await response.json();
        setSubmitError(details.message || "Registration failed.");
      }
    } catch (err) {
      setSubmitError("Network or server error.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 400, margin: "40px auto", padding: 24, border: "1px solid #ddd", borderRadius: 8 }}>
      <h2>Register</h2>
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
            value={email}
            onChange={(e) => setEmail(e.target.value)}
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
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={{ width: "100%", padding: 8, marginBottom: 4 }}
          />
          {passwordError && <div style={{ color: "red", fontSize: 13 }}>{passwordError}</div>}
        </div>

        {/* Password (Again) */}
        <div style={{ marginBottom: 16 }}>
          <label htmlFor="passwordAgain" style={{ display: "block", fontWeight: 500 }}>
            Confirm Password
          </label>
          <input
            id="passwordAgain"
            type="password"
            autoComplete="new-password"
            value={passwordAgain}
            onChange={(e) => setPasswordAgain(e.target.value)}
            required
            style={{ width: "100%", padding: 8, marginBottom: 4 }}
          />
          {passwordAgainError && <div style={{ color: "red", fontSize: 13 }}>{passwordAgainError}</div>}
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
          {loading ? "Registering..." : "Register"}
        </button>
        {submitError && <div style={{ color: "red", marginTop: 12 }}>{submitError}</div>}
        {successMessage && <div style={{ color: "green", marginTop: 12 }}>{successMessage}</div>}
		<div>Already Registered? <Link to="/signin" style={{ color: "gray" }}>Sign In</Link></div>
      </form>
    </div>
  );
};

export default RegistrationPage;
