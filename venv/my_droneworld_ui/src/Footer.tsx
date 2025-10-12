import React from "react";
import { Link } from "react-router-dom";

const Footer: React.FC = () => (
  <footer
    style={{
      textAlign: "center",
      padding: 18,
      background: "#eee",
      color: "#555",
      fontSize: 13,
      position: "fixed",
      bottom: 0,
      left: 0,
      right: 0,
    }}
  >
    &copy; {new Date().getFullYear()} EZCloudIaC.com. Your data is private and secure. Visit our <Link to="/privacy" style={{ color: "blue", marginRight: 10 }}>privacy policy</Link>to know more.  
  </footer>
);

export default Footer;
