#! /usr/bin/python
import logging
import requests
import sys
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import hdbscan
import matplotlib.pyplot as plt
from io import BytesIO
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from typing import Any, Callable, Set, Dict, List, Optional
import os
from django.conf import settings
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
match_threshold = 0.65
min_number_of_cluster_members = 2

vision_endpoint =  settings.AZURE_AI_VISION_ENDPOINT
vision_api_key = settings.AZURE_AI_VISION_API_KEY
object_uri = settings.SAMPLE_OBJECT_URI.strip('"')
scene_uri = settings.SAMPLE_SCENE_URI.strip('"')

perplexity_geo_api_key = settings.PERPLEXITY_GEO_API_KEY
perplexity_geo_api_url = settings.PERPLEXITY_GEO_API_URL

perplexity_api_key = settings.PERPLEXITY_CHAT_API_KEY
perplexity_api_url = settings.PERPLEXITY_CHAT_API_URL



# Step 1: Download images from SAS URLs
def download_image(url):
    response = requests.get(url)
    image_array = np.frombuffer(response.content, np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


# Step 2: Use OpenCV template matching to find object occurrences
def count_object_occurrences(scene, template, threshold=match_threshold):
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(scene_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    w, h = template_gray.shape[::-1]
    rects = [[pt[0], pt[1], pt[0] + w, pt[1] + h] for pt in zip(*locations[::-1])]
    rects, _ = cv2.groupRectangles(rects, groupThreshold=1, eps=0.5)
    return len(rects)

# Step 3: Count matches
def count_matches(scene_uri, object_uri):
    scene_img = download_image(scene_uri)
    object_img = download_image(object_uri)
    count = count_object_occurrences(scene_img, object_img)
    return count

# Load image from SAS URL
def load_image_from_sas(url):
    response = requests.get(url)
    image_array = np.frombuffer(response.content, np.uint8)
    logger.info(f"image_array_size={len(image_array.tolist())}")
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

def keypoints_and_descriptors(scene_img, object_img):
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(object_img, None)
    kp2, des2 = orb.detectAndCompute(scene_img, None)
    if des1 is None or des2 is None:
        return None, None, None, None
    return kp1, des1, kp2, des2

# Feature detection and matching
def get_matched_keypoints(scene_img, object_img):
    kp1,des1,kp2,des2 = keypoints_and_descriptors(scene_img, object_img)
    if des1 is None or des2 is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    logger.info(f"matched_pts={matched_pts}")
    return matched_pts
    
 # Extract matched descriptors using ORB
def get_matched_descriptors(scene_img, object_img):
    kp1,des1,kp2,des2 = keypoints_and_descriptors(scene_img, object_img)
    if des1 is None or des2 is None:
        return np.array([])

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_descriptors = np.array([des2[m.trainIdx] for m in matches])
    return matched_descriptors   

# Cluster matched keypoints using DBSCAN
def cluster_keypoints(points, eps=30, min_samples=min_number_of_cluster_members):
    if len(points) == 0:
        return []
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    return labels
    
# Cluster keypoints using HDBSCAN
def cluster_keypoints_hdbscan(points, min_cluster_size=min_number_of_cluster_members):
    if len(points) == 0:
        return np.array([])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(points)
    # if len(labels) > 0:
    #     plot_clusters(matched_points, labels)
    return labels
    
# Cluster descriptors using cosine similarity
def cluster_by_similarity(descriptors, min_cluster_size=min_number_of_cluster_members):
    if len(descriptors) == 0:
        return np.array([])

    # Normalize for cosine similarity
    descriptors = normalize(descriptors, norm='l2')

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',  # Euclidean on normalized vectors ≈ cosine similarity
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(descriptors)
    return labels
    
# Optional: visualize clusters
def plot_clusters(points, labels):
    plt.figure(figsize=(8, 6))
    for label in set(labels):
        mask = labels == label
        color = 'gray' if label == -1 else None
        plt.scatter(points[mask, 0], points[mask, 1], label=f"Cluster {label}", alpha=0.6, s=30, c=color)
    plt.title("HDBSCAN Clusters of Matched Keypoints")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

    
def count_multiple_matches(scene_uri, object_uri):
    logger.info(f"scene_uri: {scene_uri}")
    logger.info(f"object_uri: {object_uri}")
    scene_img = load_image_from_sas(scene_uri)
    logger.info(f"scene_img={scene_img}")
    object_img = load_image_from_sas(object_uri)
    logger.info(f"object_img={object_img}")
    # matched_points = get_matched_keypoints(scene_img, object_img)
    # labels = cluster_keypoints_hdbscan(matched_points)
    descriptors = get_matched_descriptors(scene_img, object_img)
    logger.info(descriptors)
    labels = cluster_by_similarity(descriptors)
    logger.info(f"len of labels={len(labels)} and labels={labels}")
    # Count valid clusters (excluding noise label -1 and 0)
    count = len([1 for label in labels if label == 1])
    # count = len(set(labels)) - (1 if -1 in labels else 0)
    return count


def agentic_retrieval(pattern_uri: Optional[str] = None, content_uri: Optional[str] = None, query_text: Optional[str] = None, account_id: Optional[str] = None, video_id: Optional[str] = None) -> str:
    if not pattern_uri:
        logger.info(f"No pattern uri for object to be detected found.")
        pattern_uri = get_object_uri(query_text, account_id, video_id)
    if not content_uri:
        logger.info(f"No content uri for scene to detect objects found.")
        content_uri = get_scene_uri(query_text, account_id, video_id)    
    count = count_multiple_matches(scene_uri, object_uri)
    return f"{count}"
    

        
def perplexity_retrieval(images_uri, query_text, account_number=2, frames=[], image_uri_template="https://saravinoteblogs.blob.core.windows.net/playground/vision/main/frames/frame{i}.jpg?sp=racwdlme&st=2025-09-14T01:02:24Z&se=2025-09-25T09:17:24Z&spr=https&sv=2024-11-04&sr=d&sig=LLB4NHzbOAqYEMUjLb0L3C39WR6cgrElqhaMtGkQg50%3D&sdd=3",pattern="(number)"):
    import requests
    import base64
    import os
    # sample_query_text = "Are there dedicated bicycle crossings in green color at street intersections in the attached set of aerial drone images?" 
    # API request payload
    headers = {
        "Authorization": f"Bearer {perplexity_api_key}",
        "accept": "application/json",
        "content-type": "application/json"
    }
    payload = {
        "model": "sonar-pro",
        "return_images": "true",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query_text}
                    # {"type": "zip_url", "zip_url": {"url": video_data_uri}}
                ]
            }
        ],
        "stream": False
    }
    if images_uri:
        for i in range(len(images_uri)):
            payload["messages"][0]["content"].append({"type": "image_url", "image_url":images_uri[i]})
            if i >= 20:
                break
    if frames and image_uri_template:
        for i in range(len(frames)):
            payload["messages"][0]["content"].append({"type": "image_url", "image_url":image_uri_template.replace(pattern, frames[i])})
            if i >= 20:
                break
    try:
        logger.info(f"perplexity_api_url = {perplexity_api_url}, perplexity_api_key = {perplexity_api_key}, query_text={query_text}")
        logger.info(payload)
        response = requests.post(perplexity_api_url, headers=headers, json=payload)
        logger.info(response.content)
        response.raise_for_status() # Raise an exception for bad status codes
        result = response.json()
        logger.info(result)
        logger.info(result["choices"][0]["message"]["content"])
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        logger.info(f"API Request failed: {e}")
        return "No comment."

def parse_bbox(s: str):
    import re
    """
    Parse a bounding box pattern from a string of the form:
    "{x: 0, y: 0, w: 74, h: 103}"
    
    Returns:
        tuple (x, y, w, h) if found, else None
    """
    pattern = r"\{x:\s*(\d+),\s*y:\s*(\d+),\s*w:\s*(\d+),\s*h:\s*(\d+)\}"
    match = re.search(pattern, s)
    if match:
        x, y, w, h = map(int, match.groups())
        return (x, y, w, h)
    pattern = r"\(x:\s*(\d+),\s*y:\s*(\d+),\s*width:\s*(\d+),\s*height:\s*(\d+)\)"
    match = re.search(pattern, s)
    if match:
        x, y, w, h = map(int, match.groups())
        return (x, y, w, h)
    return None


def get_object_uri(object_description, account_id, video_id = None, frame_number = None):
    query_text = f"Find the bounding box for a {object_description} in the saved images and cite the document url where it was found. Display the bounding box in the format {{x: , y: , w: , h: }}."
    query_text = f"Find {object_description} in saved images, cite your reference and from the description field of the reference and then find the bounding box for the asked item. Display the bounding box in the format {{x: , y: , w: , h: }}."
    sas_url_template = get_sas_url_template(account_id, video_id, upload=True)
    if frame_number:
        return get_sas_url_for_frame(account_id, sas_url_template, frame_number)
    #"""
    # ask an agent or search blob store
    from .myvideoanalyzer import ask_agent_for_url, ask_agent
    messages = ask_agent("scene-search-agent", query_text)
    if not messages:
        return None
    answer = None
    url = None
    for message in messages:
        if message.text_messages:
            answer = message.text_messages[-1].text.value
            for annotation in message.text_messages[-1].text.annotations:
                    if annotation.type == "url_citation":
                        logger.info(f"url={annotation.url_citation.url}")
                        url =  annotation.url_citation.url
                        break
    #"""
    # answer = "url=016477-0008,{x: 0, y: 0, w: 74, h: 103}"
    # url="016477-0008"
    if answer and url:
        logger.info(f"answer={answer}")
        bbox = parse_bbox(answer)
        logger.info(f"bbox={bbox}")
        if not bbox:
            return None
        # account_id = document.split('-')[0]
        frame_number = url.split('-')[1] # depends on the naming convention
        frame_number=str(int(frame_number))
        if not video_id:
            try:
               from .models import VideoEntity
               video_id = VideoEntity.objects.filter(account_id=account_id).last().id
            except Exception as e:
               logger.info(f"Request failed: {e}")
        # sas_url_template = get_sas_url_template(account_id, video_id)
        logger.info(f"sas_url_template={sas_url_template}")
        image_url = get_sas_url_for_frame(account_id, sas_url_template, frame_number)
        logger.info(f"image_url={image_url}")
        # object_url = f"{image_url}&x={bbox[0]}&y={bbox[1]}&w={bbox[2]}&h={bbox[3]}"
        # logger.info(f"object_url={object_url}")
        from datetime import datetime
        now = datetime.now()
        destination_file = now.strftime("%Y%m%d%H%M%S")
        logger.info(f"destination_file={destination_file}")
        from .myvideoindexer import read_image_from_blob
        image = read_image_from_blob(image_url)
        if image.any() == False:
            logger.info(f"{url} not found.")
            return None
        # Clip image
        x, y, width, height = bbox
        clipped_image = image[y:y+height, x:x+width]
        if clipped_image.any() == False:
            logger.info(f"Clipped image is empty.")
            return None
        _, buffer = cv2.imencode('.jpg', clipped_image)
        image_bytes = buffer.tobytes()
        from .myvideoindexer import get_image_blob_url
        object_url = get_image_blob_url(image_url, frame_number, folder='queries', prefix=destination_file, include_name=False)
        logger.info(f"object_url={object_url}")
        from .myvideoindexer import upload_image_to_blob
        upload_image_to_blob(image_bytes, object_url)
        logger.info(f"Uploaded clipped image to {object_url}")
        return object_url
    return None
    pass

def get_scene_uri(query_text, account_id, video_id, frame_number = None):
    sas_url_template = get_sas_url_template(account_id, video_id)
    if frame_number:
        return get_sas_url_for_frame(account_id, sas_url_template, frame_number)
    # ask an agent or search blob store
    from .myvideoanalyzer import ask_agent_for_url
    document = ask_agent_for_url("scene-search-agent", f"Find a saved image for {query_text} and cite the document url where it was found")
    if document:
        logger.info(f"document={document}")
        # account_id = document.split('-')[0]
        frame_number = document.split('-')[1]
        frame_number = str(int(frame_number))
        logger.info(f"frame_number={frame_number}")
        # frame_number="15"
        sas_url_template = get_sas_url_template(account_id, video_id)
        if frame_number:
            return get_sas_url_for_frame(account_id, sas_url_template, frame_number)
    return None
    pass

def get_sas_url_template(account_id, video_id= None, upload=False):
    from .models import VideoEntity
    from .serializers import VideoEntitySerializer
    from .myvideoindexer import get_image_blob_url, get_uploaded_frames
    if not video_id:
        video_id =  VideoEntity.objects.filter(account_id=account_id).last().id
        video_id = str(video_id)
    try:
        video_sas_url = VideoEntity.objects.get(pk=video_id).sas_url
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
        logger.info(f"permission={permission}")
        import datetime
        sas_token = generate_container_sas(
            account_name=account_name,
            container_name=container,
            account_key=settings.AZURE_ACCOUNT_KEY,
            permission=permission,
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        )
        # logger.info(f"sas_token={sas_token}")
        video_sas_url = video_sas_url.split('?')[0] + "?" + sas_token
        # logger.info(f"video_sas_url={video_sas_url}")
        sas_url_template = get_image_blob_url(video_sas_url, 0, folder='images', prefix='frame', include_name=False, video_id=video_id)
        # logger.info(f"sas_url_template={sas_url_template}")
        highest = get_uploaded_frames(video_sas_url, account_id = str(account_id), video_id = video_id)
        logger.info(f"highest={highest}")
        frames = []
        if highest and int(highest) > 0:
            logger.info(f"highest={highest}")
            frames = [str(0), str(int(highest/2)), str(highest-1)]
        # if frames_list:
        #     frames = frames_list.strip(',').split(',')
        logger.info(frames)
        if not frames:
            frames =  [str(num) for num in list(range(20))]
        sas_url_template = sas_url_template.replace("frame0", "frame(number)")
    except Exception as e:
        logger.info(f"Request failed: {e}")
        sas_url_template = None
    return sas_url_template
    pass

def get_sas_url_for_frame(account_id, sas_url_template, frame_number):
    try:
        sas_url = sas_url_template.replace("frame(number)", f"frame{frame_number}")
        return sas_url
    except Exception as e:
        logger.info(f"Request failed: {e}")
        return None

def ask_perplexity(query_text, account_id = "2", video_id = None, frames_list = None):
    logger.info(f"ask_perplexity called with {query_text}")
    from .models import VideoEntity
    from .serializers import VideoEntitySerializer
    from .myvideoindexer import get_image_blob_url, get_uploaded_frames
    if not video_id:
        video_id =  VideoEntity.objects.filter(account_id=account_id).last().id
        video_id = str(video_id)
    try:
        video_sas_url = VideoEntity.objects.get(pk=video_id).sas_url
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
        import datetime
        sas_token = generate_container_sas(
            account_name=account_name,
            container_name=container,
            account_key=settings.AZURE_ACCOUNT_KEY,
            permission=BlobSasPermissions(read=True, list=True),
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        )
        video_sas_url = video_sas_url.split('?')[0] + "?" + sas_token
        # logger.info(f"video_sas_url={video_sas_url}")
        sas_url_template = get_image_blob_url(video_sas_url, 0, folder='images', prefix='frame', include_name=False, video_id=video_id)
        # logger.info(f"sas_url_template={sas_url_template}")
        highest = get_uploaded_frames(video_sas_url, account_id = str(account_id), video_id = video_id)
        logger.info(f"highest={highest}")
        frames = []
        if highest and int(highest) > 0:
            logger.info(f"highest={highest}")
            frames = [str(0), str(int(highest/2)), str(highest-1)]
        if frames_list:
            frames = frames_list.strip(',').split(',')
        logger.info(frames)
        if not frames:
            frames =  [str(num) for num in list(range(20))]
        sas_url_template = sas_url_template.replace("frame0", "frame(number)")
        # logger.info(f"sas_url_template={sas_url_template}")
        return perplexity_retrieval(None, query_text, account_id, frames, sas_url_template, pattern="(number)")
    except Exception as e:
        logger.info(f"Request failed: {e}")
        return None
        
analyzer_functions: Set[Callable[..., Any]] = {
    download_image,
    count_object_occurrences,
    count_matches,
    load_image_from_sas,
    keypoints_and_descriptors,
    get_matched_keypoints,
    get_matched_descriptors,
    cluster_keypoints,
    cluster_keypoints_hdbscan,
    cluster_by_similarity,
    plot_clusters,
    count_multiple_matches,
    agentic_retrieval,
    get_object_uri,
    get_scene_uri,
    get_sas_url_template,
    get_sas_url_for_frame,
    ask_perplexity
}

image_user_functions: Set[Callable[..., Any]] = {
    agentic_retrieval,
    ask_perplexity
}