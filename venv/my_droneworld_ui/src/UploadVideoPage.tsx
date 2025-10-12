// In some UploadVideoPage.tsx
import React from "react";
import MP4Uploader from "./MP4Uploader";
import { useUser } from "./UserContext";
// AccountId could come from context, user profile, or a prop
const accountId = "demo_account_id";

const UploadVideoPage: React.FC = () => (
  <div>
    <MP4Uploader />
  </div>
);

export default UploadVideoPage;
