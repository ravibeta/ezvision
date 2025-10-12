import React, { useState } from 'react';
import { BlockBlobClient } from "@azure/storage-blob";
// import * as dotenv from 'dotenv';
// dotenv.config();

// const uploadSasUrl = process.env.AZURE_UPLOADER_SAS_URL.replace(/"/g, '');
const AZURE_UPLOADER_SAS_URL="https://sadronevideo.blob.core.windows.net/input?sp=racwdl&st=2025-07-14T20:35:02Z&se=2025-09-01T04:50:02Z&spr=https&sv=2024-11-04&sr=c&sig=AKY%2B3mgxNd3Ayx8oQ3w2j98Hk2BUoYIDP3nN5ENpNCg%3D"
const uploadSasUrl = AZURE_UPLOADER_SAS_URL.replace(/"/g, '');
console.log(uploadSasUrl);
if (!uploadSasUrl) {
   console.log('The value of AZURE_UPLOADER_SAS_URL does not seem to be set');
   process.exit(1);
}
const accountId = "0000";

const Uploader = () => {
  const [file, setFile] = useState<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    const sasToken = uploadSasUrl.split('?')[1];
    const storageAccount = uploadSasUrl.split('?')[0].replace("https://", "").replace(".blob.core.windows.net","").split('/')[0];
    const containerName = uploadSasUrl.split('?')[0].replace("https://", "").replace(".blob.core.windows.net","").replace(storageAccount, "").split('/')[1];
    const uploadingName = file.name.replace(/\s/g, '').replace(".mp4","").substring(0, 128);
    const blobName = `${accountId}-${uploadingName}.mp4`;

    const blobServiceUrl = `https://${storageAccount}.blob.core.windows.net/${containerName}/${blobName}?${sasToken}`;
    console.log(blobServiceUrl)
    const blockBlobClient = new BlockBlobClient(blobServiceUrl);

    try {
      await blockBlobClient.uploadData(file);
      alert("File uploaded successfully!");
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("Error uploading file");
    }
  };

  return (
    <div>
		<h1>MP4 File Upload for analysis:</h1>
		<input type="file" accept=".mp4" onChange={handleFileChange} />
		<button onClick={handleUpload}>Upload</button>
    </div>
  );
};

export default Uploader;
