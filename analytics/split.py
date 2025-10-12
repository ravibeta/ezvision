import requests
import os
import cv2
import io
import uuid
from urllib.parse import urlparse
from azure.storage.blob import BlobClient
from dotenv import load_dotenv
load_dotenv(override=True)
import time

video_indexer_endpoint = os.getenv("AZURE_VIDEO_INDEXER_URL", "https://api.videoindexer.ai")
video_indexer_region = os.getenv("AZURE_VIDEO_INDEXER_REGION", "eastus")
video_indexer_account_id = os.getenv("AZURE_VIDEO_INDEXER_ACCOUNT")
video_Indexer_api_key = os.getenv("AZURE_VIDEO_INDEXER_API_KEY")
video_indexer_project = os.getenv("AZURE_VIDEO_INDEXER_PROJECT", "wfmat1ysct")
video_file_path = os.getenv("AZURE_VIDEO_INPUT", "mainindexedvideo.mp4")
access_token = os.getenv("AZURE_VIDEO_INDEXER_ACCESS_TOKEN")
video_id = os.getenv("AZURE_VIDEO_ID", "lwxjba8wy3")
duration = os.getenv("AZURE_VIDEO_DURATION_IN_SECONDS", "307")
video_sas_url=os.getenv("AZURE_VIDEO_SAS_URL")
local_only = True

def get_image_blob_url(video_url, frame_number, prefix='frame', include_name=False):
    # Parse the original video URL to get account, container, and path
    parsed = urlparse(video_url)
    path_parts = parsed.path.split('/')
    blob_name = path_parts[-1].split('.')[0]
    container = path_parts[1]
    blob_path = '/'.join(path_parts[2:])
    # Remove the file name from the blob path
    blob_dir = '/'.join(blob_path.split('/')[:-1])
    if blob_dir == "" or blob_dir == None:
        blob_dir = "output"
    if include_name:
        prefix += f"{blob_name}-{frame}"
    # Create image path
    image_path = f"{blob_dir}/images/{prefix}{frame_number}.jpg"
    # Rebuild the base URL (without SAS token)
    base_url = f"{parsed.scheme}://{parsed.netloc}/{container}/{image_path}"
    # Add the SAS token if present
    sas_token = parsed.query
    if sas_token:
        image_url = f"{base_url}?{sas_token}"
    else:
        image_url = base_url
    return image_url

def download_blob_to_stream(blob_client):
    download_stream = blob_client.download_blob()
    return io.BytesIO(download_stream.readall())

def extract_and_upload_frames(video_sas_url):
    # Set up blob client for video
    video_blob_client = BlobClient.from_blob_url(video_sas_url)
    # Download video to memory stream
    video_stream = download_blob_to_stream(video_blob_client)
    # Use OpenCV to read from memory
    video_bytes = video_stream.getvalue()
    # Use cv2 to read from bytes
    video_stream.seek(0)
    video_temp = os.path.join(os.getcwd(), f"temp_{uuid.uuid4()}.mp4")
    print(video_temp)
    with open(video_temp, 'wb') as f:
        f.write(video_bytes)
    vidcap = cv2.VideoCapture(video_temp)
    # Extract frames
    frame_number = 0
    while True:
        success, frame = vidcap.read()
        if not success:
            break
        # Convert frame to bytes
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        if local_only:
            image_path = f"frame{frame_number}.jpg"
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
        else:
            # Generate image blob URL
            image_url = get_image_blob_url(video_sas_url, frame_number)
            image_blob_client = BlobClient.from_blob_url(image_url)
            # Upload frame as image
            image_blob_client.upload_blob(image_bytes, overwrite=True)
            print(f"Uploaded frame {frame_number} to {image_url}")
        frame_number += 1
    # Clean up temp file
    vidcap.release()
    os.remove(video_temp)

video_sas_url=video_sas_url.strip('"')
print(video_sas_url)
extract_and_upload_frames(video_sas_url)
