import React, { useEffect, useState } from "react";
import { useUser } from "./UserContext";
import { useNavigate } from "react-router-dom";
import { API_ENDPOINT_URL } from './constants';
const VIDEO_ENDPOINT_URL = `${API_ENDPOINT_URL.replace(/\/$/, '')}/api/video-entities/`;

type Video = {
  id: number;
  status: string;
  sas_url: string;
  account_id: string;
  created: string;
};

type VideoFormProps = {
  video?: Video;
  onSave: (data: Omit<Video, 'id'>, id?: number) => void;
  onCancel: () => void;
};

const VideoForm: React.FC<VideoFormProps> = ({ video, onSave, onCancel }) => {
  const [status, setStatus] = useState(video?.status ?? "");
  const [url, setUrl] = useState(video?.sas_url ?? "");
  const [created, setCreated] = useState(video?.created ?? "");
  const [sas_url, setSas_url] = useState(video?.sas_url ?? "");
  const [account_id, setAccount_id] = useState(video?.account_id ?? "");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave({ status, sas_url, account_id, created }, video?.id);
  };
  // console.log(video?.id ?? "None");
  useEffect(() => {
    setStatus(video?.status ?? "Initialized");
    setUrl(video?.sas_url ?? "");
	setSas_url(video?.sas_url ?? "");
    setCreated(video?.created ?? "");
	setAccount_id(video?.account_id ?? "");
  }, [video]);

  return (
    <form onSubmit={handleSubmit} style={{border: '1px solid #eee', padding: '16px', borderRadius: 8, marginBottom: 16}}>
      <h3>{video ? "Edit Video" : "Add Video"}</h3>
      <input required type="text" placeholder="Status" value={status}
        onChange={e => setStatus(e.target.value)} style={{ width: "100%", marginBottom: 6, padding: 8 }}/>
      <input required type="sas_url" placeholder="Video URL" value={url}
        onChange={e => setSas_url(e.target.value)} style={{ width: "100%", marginBottom: 6, padding: 8 }}/>
      <textarea placeholder="Created" value={created}
        onChange={e => setCreated(e.target.value)} style={{ width: "100%", marginBottom: 8, padding: 8 }}/>
      <div>
        <button type="submit" style={{ marginRight: 8 }}>{video ? "Update" : "Create"}</button>
        <button onClick={onCancel} type="button">Cancel</button>
      </div>
    </form>
  );
};

const VideoManager: React.FC = () => {
  const { user } = useUser();
  const [videos, setVideos] = useState<Video[]>([]);
  const [loading, setLoading] = useState(false);
  const [editing, setEditing] = useState<Video | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // useEffect(() => {
    // // if (!user || !user.email) {
    // //   navigate("/register");
    // // } else {
      // fetchVideos();
    // // }
    // // eslint-disable-next-line
  // }, [user]);


  const navigate = useNavigate();
  useEffect(() => {
    if (!user || !user.email) {
      navigate("/signin", { replace: true });
    } else {
		fetchVideos();
    }
    // eslint-disable-next-line
  }, [user]);
  
  const fetchVideos = async () => {
    if (!user || !user.email) return;
    setLoading(true);
    try {
	  const user_id = user != null ? `${user.id}` : "";
      const resp = await fetch(VIDEO_ENDPOINT_URL + `?account_id=${encodeURIComponent(user_id)}`);
      if (resp.ok) {
        setVideos(await resp.json());
      }
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = () => { setEditing(null); setShowForm(true); };
  const handleEdit = (video: Video) => { setEditing(video); setShowForm(true); };
  const handleAnalyze = (video: Video) => { 
	  const video_id = video != null ? `${video.id}` : "";
	  navigate("/analytics" + `?video_id=${encodeURIComponent(video_id)}`, { replace: true });
  };

  // Always include user_email in body for POST/PUT
  const handleSave = async (data: Omit<Video, 'id'>, id?: number) => {
    setError(null);
    if (!user || !user.email) return;
    try {
      let resp;
      const body = JSON.stringify({ ...data, user_email: user.email });
	  const user_id = user != null ? `${user.id}` : "";
	  
      if (id) {
        resp = await fetch(VIDEO_ENDPOINT_URL + `${id}/?account_id=${encodeURIComponent(user_id)}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body
        });
      } else {
        resp = await fetch(VIDEO_ENDPOINT_URL + `?account_id=${encodeURIComponent(user_id)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body
        });
      }
      if (!resp.ok) throw new Error("Failed to save video");
      setShowForm(false);
      setEditing(null);
      fetchVideos();
    } catch (err: any) {
      setError(err.message);
    }
  };

  // Also pass user_email query param for delete
  const handleDelete = async (id: number) => {
    if (!window.confirm("Delete this video?")) return;
    if (!user || !user.email) return;
    try {
	  const user_id = user != null ? `${user.id}` : "";	
      const resp = await fetch(VIDEO_ENDPOINT_URL + `${id}/?account_id=${encodeURIComponent(user_id)}`, { method: "DELETE" });
      if (!resp.ok) throw new Error("Failed to delete video");
      setVideos(videos.filter(v => v.id !== id));
    } catch (err: any) {
      setError(err.message);
    }
  };

  if (!user || !user.email) return null; // Or a loading spinner if desired

  return (
    <div style={{ maxWidth: 700, margin: "40px auto", padding: 24 }}>
      <h2>Video Manager</h2>
      <div style={{marginBottom:16, color:"#333"}}>Logged in as: <b>{user.email}</b></div>
      {loading ? <p>Loading...</p> :
        <div>
          <button onClick={handleCreate} disabled={showForm} style={{marginBottom: 16}}>Add Video</button>
          {showForm && (
            <VideoForm
              video={editing ?? undefined}
              onSave={handleSave}
              onCancel={() => { setShowForm(false); setEditing(null); }}
            />
          )}
          {error && <div style={{ color: "red" }}>{error}</div>}
          <table width="100%" cellPadding={8} style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr style={{background:"#eee"}}>
                <th>Status</th>
                <th>URL</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {videos.map(v => (
                <tr key={v.id}>
                  <td>{v.status}</td>
                  <td><a target="_blank" href={v.sas_url} rel="noopener noreferrer">{v.sas_url}</a></td>
                  <td>{v.created}</td>
                  <td>
					<button onClick={() => handleAnalyze(v)} style={{ color: "green" }}>Analyze</button>
                    <button onClick={() => handleEdit(v)} style={{ marginRight: 5 }}>Edit</button>
                    <button onClick={() => handleDelete(v.id)} style={{ color: "red" }}>Delete</button>
                  </td>
                </tr>
              ))}
              {videos.length === 0 && <tr><td colSpan={4}>No videos yet.</td></tr>}
            </tbody>
          </table>
        </div>
      }
    </div>
  );
};

export default VideoManager;
