import React, { useState } from "react";

interface Props {
  onSend: (text: string) => void;
}

const DroneInput: React.FC<Props> = ({ onSend }) => {
  const [input, setInput] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      onSend(input.trim());
      setInput("");
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: "flex", marginTop: "1rem" }}>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Type your message..."
        style={{ flex: 1, padding: "0.5rem", borderRadius: "4px", border: "1px solid #ccc" }}
      />
      <button type="submit" style={{ marginLeft: "0.5rem", padding: "0.5rem 1rem" }}>Send</button>
    </form>
  );
};

export default DroneInput;
