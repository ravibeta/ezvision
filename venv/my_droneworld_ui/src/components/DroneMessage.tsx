import React from "react";
import { Message } from "../types/Message";

interface Props {
  message: Message;
}

const DroneMessage: React.FC<Props> = ({ message }) => {
  return (
    <div style={{ marginBottom: "1rem", textAlign: message.type === "user" ? "right" : "left" }}>
      <div style={{ padding: "0.5rem 1rem", backgroundColor: message.type === "user" ? "#d1e7dd" : "#f8f9fa", borderRadius: "8px", display: "inline-block" }}>
        <p>{message.text}</p>
        {message.imageUrl && (
          <a href={message.downloadUrl || message.imageUrl} target="_blank" rel="noopener noreferrer">
            <img src={message.imageUrl} alt="response" style={{ maxWidth: "150px", marginTop: "0.5rem", borderRadius: "4px" }} />
          </a>
        )}
      </div>
    </div>
  );
};

export default DroneMessage;
