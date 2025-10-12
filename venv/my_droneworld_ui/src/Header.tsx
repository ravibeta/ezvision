import React from "react";
import { Link } from "react-router-dom";

const Header: React.FC = () => (
  // <header
    // style={{
      // padding: "16px 24px",
      // background: "#004080",
      // color: "white",
      // display: "flex",
      // alignItems: "center",
      // justifyContent: "space-between",
      // fontWeight: 600,
    // }}
  // >
	<header className="App-header">
        <img src="/images/DroneWorld.jpg" className="App-logo" alt="logo" />
    <div>
      <Link to="/" style={{ color: "white", textDecoration: "none", fontSize: 22 }}>
        DroneWorld Explorer App
      </Link>
    </div>
    <nav>
      <Link to="/videos" style={{ color: "white", marginRight: 18 }}>My Videos</Link>
      <Link to="/upload" style={{ color: "white", marginRight: 18 }}>Upload MP4</Link>
	  <Link to="/analytics" style={{ color: "white", marginRight: 18  }}>Analyze</Link>
      <Link to="/register" style={{ color: "white", marginRight: 18  }}>Register</Link>
	  <Link to="/signout" style={{ color: "white", marginRight: 18  }}>Sign Out</Link>
    </nav>
  </header>
);

export default Header;
