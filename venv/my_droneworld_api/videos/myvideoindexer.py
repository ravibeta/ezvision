import requests
import time
import os
import cv2
import io
import uuid
from django.conf import settings
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv(override=True)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures, ImageAnalysisResult
from azure.storage.blob import BlobClient
from azure.search.documents import SearchClient
from tenacity import retry, stop_after_attempt, wait_fixed
from pprint import pprint, pformat
from dotenv import load_dotenv  
import json 
import http.client
import numpy as np

vision_api_key = settings.AZURE_AI_VISION_API_KEY
vision_api_version = settings.VISION_API_VERSION
vision_region = settings.AZURE_AI_VISION_REGION
vision_endpoint =  settings.AZURE_AI_VISION_ENDPOINT
api_version = settings.SEARCH_API_VERSION
model_version = settings.MODEL_VERSION
search_endpoint = settings.AZURE_SEARCH_SERVICE_ENDPOINT
search_api_key  = settings.AZURE_SEARCH_ADMIN_KEY
index_name = settings.AZURE_SEARCH_INDEX_NAME
video_indexer_endpoint = settings.AZURE_VIDEO_INDEXER_URL
video_indexer_region = settings.AZURE_VIDEO_INDEXER_REGION
video_indexer_account_id = settings.AZURE_VIDEO_INDEXER_ACCOUNT
perplexity_geo_api_key = settings.PERPLEXITY_GEO_API_KEY
perplexity_geo_api_url = settings.PERPLEXITY_GEO_API_URL

        
# Step 1: Get an access token
def get_access_token():
    url = f"{settings.AZURE_VIDEO_INDEXER_URL}/auth/{settings.AZURE_VIDEO_INDEXER_REGION}/Accounts/{settings.AZURE_VIDEO_INDEXER_ACCOUNT}/AccessToken"
    headers = {
        "Ocp-Apim-Subscription-Key": settings.AZURE_VIDEO_INDEXER_API_KEY
    }
    response = requests.get(url, headers=headers)
    return response.text.strip('"')
    


def trim_filename(filename: str, max_length: int = 255) -> str:
    # Separate base name and extension
    import os
    base, ext = os.path.splitext(filename)
    
    # Truncate base if total exceeds max_length
    allowed_base_length = max_length - len(ext)
    trimmed_base = base[:allowed_base_length]

    return trimmed_base + ext
    
# Step 2: Upload video and start indexing
def upload_and_index_video(access_token, accountId, video_file_path, video_url = None):
    video_name = None
    if video_url:
        parsed_url = urlparse(video_url)
        video_path = parsed_url.path
        path_parts = video_path.split('/')
        video_name= path_parts[-1]
        print(f"{video_path}/{video_name}")
    if video_file_path:
        video_name = trim_filename(os.path.basename(video_file_path))
        print(f"video_file_path={video_file_path}/video_name={video_name}")
    # https://api-portal.videoindexer.ai/api-details#api=Operations&operation=Upload-Video
    url = f"{settings.AZURE_VIDEO_INDEXER_URL}/{settings.AZURE_VIDEO_INDEXER_REGION}/Accounts/{settings.AZURE_VIDEO_INDEXER_ACCOUNT}/Videos?name={video_name}&accessToken={access_token}" # &privacy=Private"
    print(f"url={url}")
    if video_url:
        import urllib
        encoded_url = urllib.parse.quote(video_url, safe='')
        url += f"&videoUrl={encoded_url}"
        headers = {}
        headers["Ocp-apim-subscription-key"]=""+settings.AZURE_VIDEO_INDEXER_API_KEY
        headers["Cache-Control"]="no-cache"
        headers["Authorization"]="Bearer "+ access_token
        print(f"headers={headers}")
        response = requests.post(url,headers=headers)
        return response.json()
    else:
        with open(video_file_path, 'rb') as video_file:
            files = {'file': video_file}
            response = requests.post(url, files=files)
            return response.json()

# Step 3: Wait for indexing to complete and get insights
def get_video_insights(access_token, video_id):
    url = f"{settings.AZURE_VIDEO_INDEXER_URL}/{settings.AZURE_VIDEO_INDEXER_REGION}/Accounts/{settings.AZURE_VIDEO_INDEXER_ACCOUNT}/Videos/{video_id}/Index?accessToken={access_token}"
    while True:
        response = requests.get(url)
        data = response.json()
        if data['state'] == 'Processed':
            return data
        time.sleep(10)  # Wait 10 seconds before checking again

# Step 4: Main workflow
def get_uploaded_video_id(access_token, accountId, video_file_path, video_url = None):
    video_data = upload_and_index_video(access_token, accountId, video_file_path, video_url)
    print(video_data)
    if 'ErrorType' in video_data and 'Message' in video_data:
        print(f"Error type: {video_data['ErrorType']} and message: {video_data['Message']}")
    if 'id' in video_data:
        video_id = video_data['id']
        return video_id
    return None

def get_insights_formatted(access_token, video_id):
    insights = get_video_insights(access_token, video_id)
    value = "Video highlights and key insights:\n"
    value += ("=" * 50) + "\n"
    # Extract highlights: keyframes, topics, and summarization
    if 'summarizedInsights' in insights:
        for theme in insights['summarizedInsights']['themes']:
            value += f"Theme: {theme['name']}"
            for highlight in theme['keyframes']:
                value += f"  Keyframe at {highlight['adjustedStart']} to {highlight['adjustedEnd']}\n"
                value += f"  Thumbnail: {highlight['thumbnailId']}\n"
                value += f"  Description: {highlight.get('description', 'No description')}\n"
    else:
        value += f"No summarization available. See full insights: {insights}"
    return value
"""
{'accountId': '26ff36de-cac7-4bea-ad7a-abdf0d63c19c', 'id': 'lwxjba8wy3', 'partition': None, 'externalId': None, 'metadata': None, 'name': 'mainindexedvideo.mp4', 'description': None, 'created': '2025-06-25T03:54:44.3133333+00:00', 'lastModified': '2025-06-25T03:54:44.3133333+00:00', 'lastIndexed': '2025-06-25T03:54:44.3133333+00:00', 'privacyMode': 'Private', 'userName': 'Ravi Rajamani', 'isOwned': True, 'isBase': True, 'hasSourceVideoFile': True, 'state': 'Uploaded', 'moderationState': 'OK', 'reviewState': 'None', 'isSearchable': True, 'processingProgress': '1%', 'durationInSeconds': 0, 'thumbnailVideoId': 'lwxjba8wy3', 'thumbnailId': '00000000-0000-0000-0000-000000000000', 'searchMatches': [], 'indexingPreset': 'Default', 'streamingPreset': 'Default', 'sourceLanguage': 'en-US', 'sourceLanguages': ['en-US'], 'personModelId': '00000000-0000-0000-0000-000000000000'}
"""


def repeat_video_index(access_token, video_id):
    """Retrieve the index/insights for a video by its ID."""
    url = f"{video_indexer_endpoint}/{video_indexer_region}/Accounts/{video_indexer_account_id}/Videos/{video_id}/ReIndex?accessToken={access_token}"
    response = requests.put(url)
    if response.status_code == 200:
        return response
    return get_video_insights(access_token, video_id)
    
def get_video_insights(access_token, video_id):
    url = f"{video_indexer_endpoint}/{video_indexer_region}/Accounts/{video_indexer_account_id}/Videos/{video_id}/Index?accessToken={access_token}"
    count = 0
    while True:
        response = requests.get(url)
        data = response.json()
        if 'ErrorType' in data and data['ErrorType'] == "INVALID_VIDEO_ID":
            return None
        if "state" in data and data['state'] == 'Processed':
            return data
        count+=1
        if count%10 == 0:
            print(data)
        print("Sleeping for ten seconds...")
        time.sleep(10)  # Wait 10 seconds before checking again

def get_selected_segments(insights, threshold):
        indexed_duration = insights["summarizedInsights"]["duration"]["seconds"]
        reduced_duration = (threshold * indexed_duration) / 100
        selected_segments = []
        # total_duration = 0
        for video in insights["videos"]:
            for shot in video["insights"]["shots"]:
                shot_id = shot["id"]
                for key_frame in shot["keyFrames"]:
                    key_frame_id = key_frame["id"]
                    start = key_frame["instances"][0]["start"]
                    end = key_frame["instances"][0]["end"]
                    # total_duration += float(end) - float(start)
                    print(f"Clipping shot: {shot_id}, key_frame: {key_frame_id}, start: {start}, end: {end}")
                    selected_segments +=[(start,end)]
        # print(f"Total duration: {total_duration}")
        return selected_segments

def create_project(access_token, video_id, selected_segments):
        import random
        import string
        video_ranges = []
        for start,end in selected_segments:
            intervals = {}
            intervals["videoId"] = video_id
            intervalRange = {}
            intervalRange["start"] = start
            intervalRange["end"] = end
            intervals["range"] = intervalRange
            video_ranges += [intervals]
        project_name = ''.join(random.choices(string.hexdigits, k=8))
        data = {
            "name": project_name,
            "videosRanges": video_ranges,
            "isSearchable": "false"
        }
        headers = {
            "Content-Type": "application/json"
        }
        url = f"{video_indexer_endpoint}/{video_indexer_region}/Accounts/{video_indexer_account_id}/Projects?accessToken={access_token}"
        response = requests.post(url, json=data, headers=headers)
        print(response.content)
        if response.status_code == 200:
            data = response.json()
            project_id = data["id"]
            return project_id
        else:
            return None
        
def render_video(access_token, project_id):
        url = f"{video_indexer_endpoint}/{video_indexer_region}/Accounts/{video_indexer_account_id}/Projects/{project_id}/render?sendCompletionEmail=false&accessToken={access_token}"
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers)
        print(response.content)
        if response.status_code == 202:
            return response
        else:
            return None
            
def get_render_operation(access_token, project_id):
    url = f"{video_indexer_endpoint}/{video_indexer_region}/Accounts/{video_indexer_account_id}/Projects/{project_id}/renderoperation?accessToken={access_token}"
    while True:
        response = requests.get(url)
        data = response.json()
        if "state" in data and data['state'] == 'Succeeded':
            return data
        print("Sleeping for ten seconds before checking on rendering...")
        time.sleep(10)  # Wait 10 seconds before checking again        

def download_rendered_file(access_token, project_id):
    url = f"{video_indexer_endpoint}/{video_indexer_region}/Accounts/{video_indexer_account_id}/Projects/{project_id}/renderedfile/downloadurl?accessToken={access_token}"
    response = requests.get(url)
    if response.status_code == 200:
        print(response.content)
        data = response.json()
        if "downloadUrl" in data:
            return data["downloadUrl"]
    return None

def index_and_download_video(account_id = None, project_id = None, video_id = None, video_file_path = None, video_url = None, repeat = True):
    if not account_id:
        account_id = settings.AZURE_VIDEO_INDEXER_ACCOUNT
    # Main workflow
    access_token = settings.AZURE_VIDEO_INDEXER_ACCESS_TOKEN.strip('"')
    # '''
    if not access_token:
        access_token = get_access_token()
    # print(access_token)
    if not access_token:
        access_token = get_access_token()
    # print(access_token)
    if not video_id and not video_file_path and not video_url:
        return None
    if not video_id:
        if video_file_path:
            video_id = get_uploaded_video_id(access_token, account_id, video_file_path)
        if video_url:
            video_id = get_uploaded_video_id(access_token, account_id, video_file_path, video_url=video_url)
    print(f"Video_id={video_id}")
    if not video_id:
        print("No video.")
        return None
    insights = get_video_insights(access_token, video_id)
    if repeat:
        insights = repeat_video_index(access_token, video_id)
    selected_segments = get_selected_segments(insights, 10)
    if not project_id:
        project_id = create_project(access_token, video_id, selected_segments)
    print(project_id)
    render_response = render_video(access_token, project_id)
    print(render_response)
    if render_response:
        status = get_render_operation(access_token, project_id)
        print(status)
        download_url = download_rendered_file(access_token, project_id)
        print(download_url)
        return download_url
    '''
    download_url=video_url.strip('"')
    return download_url
    # return None
    '''
    return None
    
local_only = False

def get_image_blob_url(video_url, frame_number, folder='images', prefix='frame', include_name=False, video_id=None):
    # Parse the original video URL to get account, container, and path
    parsed = urlparse(video_url)
    path_parts = parsed.path.split('/')
    blob_name = path_parts[-1].split('.')[0]
    container = path_parts[1]
    blob_path = '/'.join(path_parts[2:])
    # Remove the file name from the blob path
    blob_dir = '/'.join(blob_path.split('/')[:-1])
    # print(f"parsed={parsed},path_parts={path_parts},blob_name={blob_name},container={container},blob_path={blob_path},blob_dir={blob_dir}")
    if blob_dir == "" or blob_dir == None:
        blob_dir = "output"
    new_path = f"{blob_dir}/{folder}"
    if video_id:
        new_path += "/" + str(video_id)
    # Create image path
    if include_name:
        prefix += blob_name
    numeral=str(frame_number)
    image_path = f"{new_path}/{prefix}{numeral}.jpg"
    # print(f"image_path={image_path}")
    
    # Rebuild the base URL (without SAS token)
    base_url = f"{parsed.scheme}://{parsed.netloc}/{container}/{image_path}"
    # Add the SAS token if present
    sas_token = parsed.query
    if sas_token:
        image_url = f"{base_url}?{sas_token}"
    else:
        image_url = base_url
    # print(f"output={image_url}")
    return image_url

def download_blob_to_stream(blob_client):
    download_stream = blob_client.download_blob()
    return io.BytesIO(download_stream.readall())

def extract_and_upload_frames(video_sas_url, video_id = None):
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
            image_url = get_image_blob_url(video_sas_url, frame_number, video_id=video_id).strip('"')
            # print(image_url)
            image_blob_client = BlobClient.from_blob_url(image_url)
            # Upload frame as image
            image_blob_client.upload_blob(image_bytes, overwrite=True)
            print(f"Uploaded frame {frame_number} to {image_url}")
        frame_number += 1
    # Clean up temp file
    vidcap.release()
    if os.path.exists(video_temp):
        try:
            os.remove(video_temp)
            print(f"Video file '{video_temp}' deleted successfully.")
        except OSError as e:
            print(f"Error deleting file '{video_temp}': {e}")
    return frame_number
    
def get_uploaded_frames(video_sas_url, account_id = None, video_id = None):
    blob_service_client = None
    account_name = settings.ACCOUNT_NAME
    account_url = f'https://{account_name}.blob.core.windows.net'
    account_key = settings.AZURE_ACCOUNT_KEY
    container = settings.CONTAINER_NAME
    try:
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=account_key
        )
    except Exception as e:
       print(e)
       return 0
    for frame_number in range(9999):
        try:
            if blob_service_client:
                image_url = get_image_blob_url(video_sas_url, frame_number, video_id=video_id).strip('"')
                prefix=f"{account_url}/{container}/"
                blob_name=image_url.split('?')[0].replace(prefix,"")
                # print(f"blob_name={blob_name}")
                blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name) 
                exists = False
                blob_client.get_blob_properties()
                exists = True
                if exists:
                # image_url = get_image_blob_url(video_sas_url, frame_number, video_id=video_id).strip('"')
                # if BlobClient.from_blob_url(image_url).exists():
                   continue
        except Exception as e:
            print(e)
            break
        # print(image_url)
    return frame_number

    
def vectorize_extracted_frames(video_sas_url, frame_number = None, video_id = None):
    # frames = extract_and_upload_frames(video_sas_url)
    vision_credential = AzureKeyCredential(vision_api_key)
    analysis_client = ImageAnalysisClient(vision_endpoint, vision_credential)
    # Set up blob client for video
    video_blob_client = BlobClient.from_blob_url(video_sas_url)
    # Extract frames
    # frame_number = 0
    tuples = []
    while True:
        try:
            # Generate image blob URL
            image_url = get_image_blob_url(video_sas_url, frame_number, video_id = video_id)
            image_blob_client = BlobClient.from_blob_url(image_url)
            vector = vectorize_image(image_url, vision_api_key, vision_region)
            if vector:
                vector = np.pad(vector, (0, 1536 - len(vector)), mode='constant')
                print(f"Vectorized frame: {frame_number}, len={len(vector)}")
            description = analyze_image(analysis_client, image_url)
            if description:
                print(f"Analyzed frame: {frame_number}")
                tuples += [(vector, description)]
        except Exception as e:
            print(f"No such image: {image_url[74:80]}. Giving up...")
            raise
        break
        #frame_number += 1   
    return tuples

def vectorize_extracted_frames_and_upload(video_sas_url, frame_number, search_client, account_id, video_id = None):
    vector_descriptions = vectorize_extracted_frames(video_sas_url, frame_number, video_id)
    #search_client = get_search_client()
    # frame_number = 0
    source_sas_url = get_image_blob_url(video_sas_url, frame_number, video_id = video_id)
    for vector, description in vector_descriptions:
        print(f"processing {frame_number} ...")
        form_and_upload_document(search_client, account_id, frame_number, vector, description, source_sas_url, deep = False)
        # frame_number += 1

# access_token = os.getenv("AZURE_VIDEO_INDEXER_ACCESS_TOKEN", get_access_token())
# video_sas_url=video_sas_url.strip('"')
# print(video_sas_url)
# extract_and_upload_frames(video_sas_url)
# vision_credential = AzureKeyCredential(vision_api_key)
# analysis_client = ImageAnalysisClient(vision_endpoint, vision_credential)

@retry(stop=stop_after_attempt(5), wait=wait_fixed(60))
def vectorize_image(image_path, key, region):
    try:
        # API version and model version
        api_version = "2024-02-01"
        model_version = "2023-04-15"

        # Construct the request URL
        url = f"{vision_endpoint}/computervision/retrieval:vectorizeImage?api-version={api_version}&model-version={model_version}"

        # Set headers
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": key
        }

        # Set the payload with the SAS URL
        payload = {
            "url": image_path
        }

        # Make the POST request
        response = requests.post(url, headers=headers, json=payload)

        # Check the response
        if response.status_code == 200:
            result = response.json()
            # The vector is in the 'vector' field of the response
            vector = result.get("vector")
            
            # print("Vector embedding:", vector)
            return vector
        else:
            print("Error:", response.status_code, response.text)
            vector = [0.0] * 1024
            raise Exception(f"Error vectorizing image {image_path[74:80]}")

    except (requests.exceptions.Timeout, http.client.HTTPException) as e:
        print(f"Timeout/Error for {image_path[74:80]}. Retrying...")
        raise

@retry(stop=stop_after_attempt(5), wait=wait_fixed(60))
def analyze_image(client, image_url):
    description = "No description"
    try:
        # Define all available visual features for analysis
        features = [
            VisualFeatures.CAPTION,
            VisualFeatures.TAGS,
            VisualFeatures.OBJECTS,
            VisualFeatures.READ,
            VisualFeatures.SMART_CROPS,
            VisualFeatures.DENSE_CAPTIONS,
            VisualFeatures.PEOPLE
        ]
        
        # Analyze the image from the SAS URL
        result = client.analyze_from_url(
            image_url=image_url,
            visual_features=features,
            gender_neutral_caption=True        )
        # Explicitly cast to ImageAnalysisResult (for clarity)
        # print(str(result))
        result: ImageAnalysisResult = result
        if result is not None:
            captions = []
            captions += [ f"{result.caption.text}" if result.caption is not None else "No Caption"]
            captions += [ f"{caption.text}" for caption in result.dense_captions.list if result.dense_captions is not None]
            # Enhance result
            result.description = ",".join(captions)
            print(result.description)
            description = pformat(result.__dict__, depth=4, compact=False)
            print(f"1={description}")
            # description = prepare_json_string_for_load(description).replace('""','')
            # print(f"2={description}")
    except HttpResponseError as e:
        print(str(e))
        raise
    return description


@retry(stop=stop_after_attempt(5), wait=wait_fixed(60))
def upload(destination_client, document):
    try:
        upload_results = destination_client.upload_documents([document])
        error = ','.join([upload_result.error_message for upload_result in upload_results if upload_result.error_message]).strip(",")
        if error:
            print(error)
    except HttpResponseError as e:
        print(f"Error from upload: {e}")
        raise    

def prepare_json_string_for_load(text):
    import re
    text = text.replace("\"", "'")
    text = text.replace("{'", "{\"")
    text = text.replace("'}", "\"}")
    text = text.replace(" '", " \"")
    text = text.replace("' ", "\" ")
    text = text.replace(":'", ":\"")
    text = text.replace("':", "\":")
    text = text.replace(",'", ",\"")
    text = text.replace("',", "\",")
    return re.sub(r'\n\s*', '', text)

def geolocation(image_url):
    # Prepare headers and payload
    headers = {
        "Authorization": f"Bearer {perplexity_geo_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "image_url": image_url
    }

    # Send request to Perplexity API
    response = requests.post(perplexity_geo_api_url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        latitude = result.get("latitude", "")
        longitude = result.get("longitude", "")
        print(f"Estimated GPS coordinates: Latitude={latitude}, Longitude={longitude}")
        return latitude, longitude
    else:
        print("Error:", response.status_code, response.text)
        return "", ""
        
def read_image_from_blob(sas_url):
    """Reads an image from Azure Blob Storage using its SAS URL."""
    response = requests.get(sas_url)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    else:
        # raise Exception(f"Failed to fetch image. Status code: {response.status_code}")
        return None

def upload_image_to_blob(clipped_image, object_url):
    try:
        image_blob_client = BlobClient.from_blob_url(object_url)
        # Upload frame as image
        image_blob_client.upload_blob(clipped_image, overwrite=True)
    except Exception as e:
        print(f"Error uploading image to blob: {e}")
        return
    pass

def clip_image(image, bounding_box):
    # Extract bounding box parameters
    x, y, width, height = bounding_box

    # Clip the region using slicing
    clipped_image = image[y:y+height, x:x+width]

    return clipped_image

def to_string(bounding_box):
    return f"{bounding_box['x']},{bounding_box['y']},{bounding_box['w']},{bounding_box['h']}"


# Example usage
def form_and_upload_document(destination_client, account_id, frame_number, vector, description, source_sas_url, deep = False):
        destination_file = account_id + "-" + f"{frame_number:04d}"
        geotags = str(geolocation(source_sas_url))
        document = {}
        document['id'] = destination_file
        document['account_id'] = account_id
        document['vector'] =  vector.tolist()
        document['boundingbox'] = "0,0,1280,720"
        document['geotags'] = geotags
        document["description"] = description
        upload(destination_client, document)
        try:
            description_text = prepare_json_string_for_load(description).replace('""','')
            description_json = json.loads(description_text)
        except Exception as e:
            print(f"{frame_number}: parsing error: {e}")
        if description_json == None:
            print("Description could not be parsed.")
            return
        vision_credential = AzureKeyCredential(vision_api_key)
        analysis_client = ImageAnalysisClient(vision_endpoint, vision_credential)    
        if deep == True and description_json and description_json["_data"] and description_json["_data"]["denseCaptionsResult"] and description_json["_data"]["denseCaptionsResult"]["values"]:
            objectid = 0
            for item in description_json["_data"]["denseCaptionsResult"]["values"]:
                objectid += 1
                if objectid == 1:
                    continue
                box = item.get("boundingBox", None)
                print(f"{destination_file}: {box}")
                if box:
                    bounding_box = (box["x"], box["y"], box["w"], box["h"])
                    image = read_image_from_blob(source_sas_url)
                    if image.any() == False:
                       print(f"{destination_file} not found.")
                       continue

                    # Clip image
                    clipped = clip_image(image, bounding_box)
                    destination_file += f"-{objectid:04d}"
                    destination_sas_url = get_image_blob_url(source_sas_url, frame_number, folder='images', prefix=destination_file, include_name=False)
                    upload_image_to_blob(clipped, destination_sas_url)
                    object_vector = vectorize_image(destination_sas_url, vision_api_key, vision_region)
                    if object_vector:
                        object_vector = np.pad(object_vector, (0, 1536 - len(vector)), mode='constant')
                        object_description = analyze_image(analysis_client, destination_sas_url)
                        if object_description:
                            document = {}
                            document['id'] = destination_file
                            document['account_id'] = account_id
                            document['vector'] =  object_vector.tolist()
                            document['boundingbox'] = to_string(box)
                            document['geotags'] = geotags
                            document["description"] = object_description   
                            upload(destination_client, document)
                            print(f"uploaded {frame_number}-{objectid}")
                        else:
                            print("No object description")
                    else:
                        print("No object vector")
                else:
                    print("no objects detected")

def get_timestamps(access_token, video_id):
    insights = get_video_insights(access_token, video_id)
    #pprint(insights)
    timestamps=[]
    for keyframe in insights['videos'][0]['insights']['shots'][0]['keyFrames']:
        timestamps+=[(keyframe['instances'][0]['start'], keyframe['instances'][0]['end'])]
    print(timestamps)
    return timestamps
    
def get_search_client():
    return  SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_api_key)
    )

def copy_blob(source_sas_url: str, destination_sas_url: str, poll_interval: int = 2):
    status = None
    try:
        dest_blob = BlobClient.from_blob_url(destination_sas_url)
        # Start copy operation (server-side, async)
        copy_props = dest_blob.start_copy_from_url(source_sas_url)
        copy_id = copy_props['copy_id']

        print("Copy initiated.")
        print(f"Copy ID: {copy_props['copy_id']}")
        print(f"Copy Status: {copy_props['copy_status']}")

        # Poll until completion
        while True:
            props = dest_blob.get_blob_properties()
            status = props.copy.status
            print(f"Copy status: {status}")

            if status in ("success", "failed", "aborted"):
                print(f"Final status: {status}")
                break
            import time
            time.sleep(poll_interval)
    except Exception as e:
        print(f"Error during blob copy: {e}")
    return status

def get_destination_sas_url(video_sas_url: str, upload = True) -> str:
        blob_service_client = None
        account_name = settings.ACCOUNT_NAME
        account_url = f'https://{account_name}.blob.core.windows.net'
        account_key = settings.AZURE_ACCOUNT_KEY
        container = settings.CONTAINER_NAME
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=account_key
        )
        blob_client = blob_service_client.get_blob_client(container=container, blob="/")
        from azure.storage.blob import generate_container_sas, BlobSasPermissions
        permission = BlobSasPermissions(read=True, list=True)
        if upload == True:
            permission = BlobSasPermissions(read=True, write=True, create=True, list=True, add=True, delete_previous_version=True)
        print(f"permission={permission}")
        import datetime
        sas_token = generate_container_sas(
            account_name=account_name,
            container_name=container,
            account_key=settings.AZURE_ACCOUNT_KEY,
            permission=permission,
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        )
        # print(f"sas_token={sas_token}")
        video_sas_url = video_sas_url.split('?')[0] + "?" + sas_token
        parsed = urlparse(video_sas_url)
        path_parts = parsed.path.split('/')
        blob_name = path_parts[-1].split('.')[0]
        video_sas_url = video_sas_url.replace(blob_name, blob_name+"_indexed")
        return video_sas_url

def indexing_workflow(source_video_url, account_id = None, video_id = None):
    print(f"account_id={account_id}")
    if not account_id:
        account_id = settings.AZURE_VIDEO_INDEXER_ACCOUNT
    indexer_url = index_and_download_video(account_id = account_id, video_url = source_video_url)
    video_url = get_destination_sas_url(source_video_url)
    print(f"Destination SAS URL: {video_url}")
    status = copy_blob(indexer_url, video_url)
    print(f"status of copy blob: {status}")
    if not status or status != "success":
        return None
    # video_url = source_video_url
    if not video_url:
        return None
    frames = get_uploaded_frames(video_url, account_id, video_id)
    if frames == 0:
       frames = extract_and_upload_frames(video_url, video_id)
    # frames = 27
    client = get_search_client()
    for frame_number in range(frames):
        vectorize_extracted_frames_and_upload(video_url, frame_number, client, account_id, video_id)
        print(f"{frame_number} indexed")
    return video_url