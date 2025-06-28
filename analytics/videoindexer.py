import requests
import time
import os
from dotenv import load_dotenv
load_dotenv(override=True)


# Replace these with your actual values
AZURE_VIDEO_INDEXER_API_URL = "https://api.videoindexer.ai" # https://www.videoindexer.ai/account/26ff36de-cac7-4bea-ad7a-abdf0d63c19c/settings
# Ravi Rajamani Current ravibetahotmail.onmicrosoft.com 1f4c33e1-e960-43bf-a135-6db8b82b6885
AZURE_VIDEO_INDEXER_ACCESS_TOKEN = os.getenv("AZURE_VIDEO_INDEXER_ACCESS_TOKEN")
video_indexer_region = "eastus"
video_indexer_account_id = os.getenv("AZURE_VIDEO_INDEXER_ACCOUNT")
AZURE_API_KEY = os.getenv("AZURE_VIDEO_INDEXER_API_KEY")
VIDEO_FILE_PATH = os.getenv("AZURE_VIDEO_INPUT", "mainindexedvideo.mp4")
video_indexer_endpoint = os.getenv("AZURE_VIDEO_INDEXER_URL", "https://api.videoindexer.ai")
video_indexer_region = os.getenv("AZURE_VIDEO_INDEXER_REGION", "eastus")
video_indexer_account_id = os.getenv("AZURE_VIDEO_INDEXER_ACCOUNT")
video_Indexer_api_key = os.getenv("AZURE_VIDEO_INDEXER_API_KEY")
video_file_path = os.getenv("AZURE_VIDEO_INPUT", "mainindexedvideo.mp4")
access_token = os.getenv("AZURE_VIDEO_INDEXER_ACCESS_TOKEN")
# Step 1: Get an access token
def get_access_token():
    url = f"{video_indexer_endpoint}/auth/{video_indexer_region}/Accounts/{video_indexer_account_id}/AccessToken"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_API_KEY
    }
    response = requests.get(url, headers=headers)
    return response.text.strip('"')

# Step 2: Upload video and start indexing
def upload_and_index_video(video_file_path, access_token):
    video_name = os.path.basename(video_file_path)
    url = f"{video_indexer_endpoint}/{video_indexer_region}/Accounts/{video_indexer_account_id}/Videos?name={video_name}&accessToken={access_token}&privacy=Private"
    with open(video_file_path, 'rb') as video_file:
        files = {'file': video_file}
        response = requests.post(url, files=files)
    return response.json()

# Step 3: Wait for indexing to complete and get insights
def get_video_insights(access_token, video_id):
    url = f"{video_indexer_endpoint}/{video_indexer_region}/Accounts/{video_indexer_account_id}/Videos/{video_id}/Index?accessToken={access_token}"
    while True:
        response = requests.get(url)
        data = response.json()
        if data['state'] == 'Processed':
            return data
        time.sleep(10)  # Wait 10 seconds before checking again

# Step 4: Main workflow
# access_token = get_access_token()
access_token=os.getenv("AZURE_VIDEO_INDEXER_ACCESS_TOKEN")
video_data = upload_and_index_video(video_file_path, access_token)
print(video_data)
video_id = video_data['id']
insights = get_video_insights(access_token, video_id)

print("Video highlights and key insights:")
print("=" * 50)
# Extract highlights: keyframes, topics, and summarization
if 'summarizedInsights' in insights:
    for theme in insights['summarizedInsights']['themes']:
        print(f"Theme: {theme['name']}")
        for highlight in theme['keyframes']:
            print(f"  Keyframe at {highlight['adjustedStart']} to {highlight['adjustedEnd']}")
            print(f"  Thumbnail: {highlight['thumbnailId']}")
            print(f"  Description: {highlight.get('description', 'No description')}")
else:
    print("No summarization available. See full insights:", insights)
"""
{'accountId': '26ff36de-cac7-4bea-ad7a-abdf0d63c19c', 'id': 'lwxjba8wy3', 'partition': None, 'externalId': None, 'metadata': None, 'name': 'mainindexedvideo.mp4', 'description': None, 'created': '2025-06-25T03:54:44.3133333+00:00', 'lastModified': '2025-06-25T03:54:44.3133333+00:00', 'lastIndexed': '2025-06-25T03:54:44.3133333+00:00', 'privacyMode': 'Private', 'userName': 'Ravi Rajamani', 'isOwned': True, 'isBase': True, 'hasSourceVideoFile': True, 'state': 'Uploaded', 'moderationState': 'OK', 'reviewState': 'None', 'isSearchable': True, 'processingProgress': '1%', 'durationInSeconds': 0, 'thumbnailVideoId': 'lwxjba8wy3', 'thumbnailId': '00000000-0000-0000-0000-000000000000', 'searchMatches': [], 'indexingPreset': 'Default', 'streamingPreset': 'Default', 'sourceLanguage': 'en-US', 'sourceLanguages': ['en-US'], 'personModelId': '00000000-0000-0000-0000-000000000000'}
"""