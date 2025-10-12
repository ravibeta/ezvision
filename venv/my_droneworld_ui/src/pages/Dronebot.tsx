import React, { useEffect, useState } from "react";
import { useUser } from "../UserContext";
import { Message } from "../types/Message";
import DroneMessageComponent from "../components/DroneMessage";
import DroneInput from "../components/DroneInput";
import DroneMessage from "../components/DroneMessage";
import { v4 as uuidv4 } from "uuid";
import { API_ENDPOINT_URL } from '../constants';
import { useNavigate } from "react-router-dom";

const VIDEO_ENDPOINT_URL = `${API_ENDPOINT_URL.replace(/\/$/, '')}/api/chat/`;

const Dronebot: React.FC = () => {
	
  const { user } = useUser();
  const [video, setVideo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  function escapeAndQuote(input: string): string {
		  const escaped = input
			.replace(/\\/g, '\\\\')   // escape backslashes
			.replace(/"/g, '\\"')     // escape double quotes
			.replace(/\n/g, '\\n')    // escape newlines
			.replace(/\r/g, '\\r')    // escape carriage returns
			.replace(/\t/g, '\\t');   // escape tabs

		  return `"${escaped}"`; // wrap in double quotes
		};
  const sendMessage = async (text: string) => {
    setError(null);
    if (!user || !user.email) return;
    const userMessage: Message = {
      id: uuidv4(),
      type: "user",
      text,
    };
    setMessages((prev) => [...prev, userMessage]);

    try {

	  const user_id = user != null ? `${user.id}` : "";
	  const video_id = video != null ? `${video}`: "";
      const response = await fetch(VIDEO_ENDPOINT_URL + `?account_id=${encodeURIComponent(user_id)}&video_id=${encodeURIComponent(video_id)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: escapeAndQuote(text), account_id: user_id, video_id: video_id}),
      });
      // console.log("awaiting...")
      const data = await response.json();
	  // console.log(data);
      const botMessage: Message = {
        id: uuidv4(),
        type: "bot",
        text: data.text,
        imageUrl: data.imageUrl,
        downloadUrl: data.downloadUrl,
      };
      // console.log(botMessage);
      setMessages((prev) => [...prev, botMessage]);
	  
    } catch (err: any) {
      console.error("Error fetching response:", err.message);
	  setError(err.message);
    }
  };
  const navigate = useNavigate();
  useEffect(() => {
    if (!user || !user.email) {
      navigate("/signin", { replace: true });
    } else {
	  const queryParams = new URLSearchParams(window.location.search);
	  const id = queryParams.get('video_id');
	  const vid = id != null ? `${id}`: "";
	  // console.log("Video:" + `video_id=${encodeURIComponent(vid)}`);
	  setVideo(id);
	}
    // eslint-disable-next-line
  }, [user]);
  return (
    <div style={{ maxWidth: "600px", margin: "2rem auto", padding: "1rem", backgroundColor: "#fff", borderRadius: "8px", boxShadow: "0 0 10px rgba(0,0,0,0.1)" }}>
      <h2>Drone Video Dronebot</h2>
      <div style={{ maxHeight: "400px", overflowY: "auto", marginBottom: "1rem" }}>
        {messages.map((msg) => (
          <DroneMessageComponent key={msg.id} message={msg} />
        ))}
      </div>
      <DroneInput onSend={sendMessage} />
    </div>
  );
};

export default Dronebot;
