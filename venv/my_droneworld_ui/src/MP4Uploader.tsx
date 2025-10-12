import React, { useRef, useEffect, useState } from "react";
import { useUser } from "./UserContext";
import { API_ENDPOINT_URL } from './constants';
import { useNavigate } from "react-router-dom";

const VIDEO_ENDPOINT_URL = `${API_ENDPOINT_URL.replace(/\/$/, '')}/api/upload-video/`;


interface MP4UploaderProps {
  // accountId: string;
  onUploadSuccess?: (sasUrl: string) => void;
}
// const MP4Uploader: React.FC<MP4UploaderProps> = ({ accountId, onUploadSuccess }) => {
const MP4Uploader: React.FC<MP4UploaderProps> = ({ onUploadSuccess }) => {
  const { user } = useUser();
  const fileInput = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sasUrl, setSasUrl] = useState<string | null>(null);

  const handleSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      if (files[0].type !== "video/mp4") {
        setError("Only MP4 files are allowed.");
        setFile(null);
      } else {
        setFile(files[0]);
        setError(null);
      }
    } else {
      setFile(null);
    }
    setSasUrl(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select an MP4 file.");
      return;
    }
    setUploading(true);
    setError(null);
    if (!user || !user.email) return;
    setSasUrl(null);
    const user_id = user != null ? `${user.id}` : "";
    const form = new FormData();
    form.append("file", file);
    form.append("account_id", user_id);

    try {
      const res = await fetch(VIDEO_ENDPOINT_URL, {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      if (res.ok && data.sas_url) {
        setSasUrl(data.sas_url);
        onUploadSuccess?.(data.sas_url);
      } else if (data.error) {
        setError(data.error as string);
      } else {
        setError("Upload failed.");
      }
    } catch (e: any) {
      setError("Network error during upload.");
    } finally {
      setUploading(false);
    }
  };

  const navigate = useNavigate();
  useEffect(() => {
    if (!user || !user.email) {
      navigate("/signin", { replace: true });
    }
    // eslint-disable-next-line
  }, [user]);
  
  return (
    <div style={{ maxWidth: 420, margin: "38px auto", padding: 28, border: "1px solid #ddd", borderRadius: 10 }}>
	  <h3>MP4 File Upload for analysis:</h3>
      <input
        type="file"
        accept="video/mp4"
        onChange={handleSelect}
        ref={fileInput}
        style={{ marginBottom: 14, display: "block" }}
        disabled={uploading}
      />
      {file && <div style={{ marginBottom: 10, color: "#222" }}>Selected: <b>{file.name}</b></div>}
      <button
        onClick={handleUpload}
        disabled={!file || uploading}
        style={{ padding: "8px 20px", fontWeight: 600, background: "#004080", color: "white", border: "none", borderRadius: 4 }}
      >
        {uploading ? "Uploading..." : "Upload"}
      </button>
      {error && <div style={{ color: "red", marginTop: 16 }}>{error}</div>}
      {sasUrl &&
        <div style={{ color: "green", marginTop: 16 }}>
          <div>Upload successful! SAS URL:</div>
          <a href={sasUrl} target="_blank" rel="noopener noreferrer">{sasUrl}</a>
        </div>
      }
    </div>
  );
};

export default MP4Uploader;
