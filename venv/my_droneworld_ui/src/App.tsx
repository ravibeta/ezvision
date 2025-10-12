import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import React from 'react';
import './App.css';
import Header from "./Header";
import Footer from "./Footer";
import { UserProvider, useUser } from "./UserContext";
import UploadVideoPage from './UploadVideoPage';
import RegistrationPage from "./RegistrationPage";
import SignIn from "./SignIn";
import SignOut from "./SignOut";
import Privacy from "./Privacy";
import Dronebot from "./pages/Dronebot";
import VideoManager from "./VideoManager";
// logo from https://img.freepik.com/premium-vector/drone-world-logo-design-vector-illustration_685330-2011.jpg


const AppRoutes = () => {
  const { user } = useUser();
  return (
    <Routes>
	  <Route path="/upload" element={<UploadVideoPage />} />
	  <Route path="/signin" element={<SignIn />} />
      <Route path="/register" element={<RegistrationPage />} />
      <Route path="/videos" element={
        user && user.email ? <VideoManager /> : <Navigate to="/SignIn" />
      } />
      <Route path="*" element={<Navigate to={user && user.email ? "/videos" : "/SignIn"} />} />
	  <Route path="/signout" element={<SignOut />} />
	  <Route path="/privacy" element={<Privacy />} />
	  <Route path="/analytics" element={<Dronebot />} />
    </Routes>
  );
};

const App = () => (
  <UserProvider>
    <BrowserRouter>
		<Header />
      <AppRoutes />
	  	<Footer/>
    </BrowserRouter>
  </UserProvider>
);

export default App;
