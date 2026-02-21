#! /usr/bin/python
import requests
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
match_threshold = 0.65
min_number_of_cluster_members = 2
# Azure Vision credentials
vision_endpoint=os.getenv("AZURE_AI_VISION_ENDPOINT")
vision_api_key = os.getenv("AZURE_AI_VISION_API_KEY")
# object_uri = os.getenv("AZURE_RED_CAR_2_SAS_URL").strip('"')
# scene_uri = os.getenv("AZURE_QUERY_SAS_URI").strip('"')
# object_uri = "https://saravinoteblogs.blob.core.windows.net/playground/vision/main/objects/001790-0008.jpg?sp=racwdyme&st=2025-12-21T05:16:25Z&se=2025-12-21T13:31:25Z&spr=https&sv=2024-11-04&sr=b&sig=EYIOmr%2BHSuSUxB%2FLnYgmfA6TB7nYntPQE3h53KCM7ow%3D"
# scene_uri = "https://saravinoteblogs.blob.core.windows.net/playground/vision/main/objects/010011-0003.jpg?sp=racwdyme&st=2025-12-21T05:14:40Z&se=2025-12-22T13:29:40Z&spr=https&sv=2024-11-04&sr=b&sig=tBiT3%2F5Ec7g3fxDR%2B4ZSZAWiRSnTn5QAa7kYefFG2IU%3D"

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
# print(f"Detected {count_matches()} occurrences of the object.")
#Output: Detected 1 occurrences of the object.

# Load image from SAS URL
def load_image_from_sas(url):
    response = requests.get(url)
    image_array = np.frombuffer(response.content, np.uint8)
    print(f"image_array_size={len(image_array.tolist())}")
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

def keypoints_and_descriptors(scene_img, object_img):
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(object_img, None)
    print(f"kp1={len(kp1) if kp1 is not None else 'None'}, des1={len(des1) if des1 is not None else 'None'}")
    kp2, des2 = orb.detectAndCompute(scene_img, None)
    print(f"kp2={len(kp2) if kp2 is not None else 'None'}, des2={len(des2) if des2 is not None else 'None'}")
    if des1 is None or des2 is None:
        return None, None, None, None
    return kp1, des1, kp2, des2

# Feature detection and matching
def get_matched_keypoints(scene_img, object_img):
    # orb = cv2.ORB_create(nfeatures=1000)
    # kp1, des1 = orb.detectAndCompute(object_img, None)
    # kp2, des2 = orb.detectAndCompute(scene_img, None)
    kp1,des1,kp2,des2 = keypoints_and_descriptors(scene_img, object_img)
    if des1 is None or des2 is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    print(f"matched_pts={matched_pts}")
    return matched_pts
    
 # Extract matched descriptors using ORB
def get_matched_descriptors(scene_img, object_img):
    # orb = cv2.ORB_create(nfeatures=1000)
    # kp1, des1 = orb.detectAndCompute(object_img, None)
    # kp2, des2 = orb.detectAndCompute(scene_img, None)
    kp1,des1,kp2,des2 = keypoints_and_descriptors(scene_img, object_img)
    if des1 is None or des2 is None:
        print(f"des1={des1} or des2={des2} is None")
        return np.array([])

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    print(f"Number of raw matches: {len(matches)}")
    matches = sorted(matches, key=lambda x: x.distance)
    print(f"Number of sorted matches: {len(matches)}")
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
    print(f"scene_uri: {scene_uri}")
    print(f"object_uri: {object_uri}")
    scene_img = load_image_from_sas(scene_uri)
    print(f"scene_img={len(scene_img) if scene_img is not None else 'None'}")
    object_img = load_image_from_sas(object_uri)
    print(f"object_img={len(object_img) if object_img is not None else 'None'}")
    # matched_points = get_matched_keypoints(scene_img, object_img)
    # labels = cluster_keypoints_hdbscan(matched_points)
    descriptors = get_matched_descriptors(scene_img, object_img)
    print(f"descriptors={descriptors}")
    labels = cluster_by_similarity(descriptors)
    print(f"len of labels={len(labels)} and labels={labels}")
    # Count valid clusters (excluding noise label -1 and 0)
    count = len([1 for label in labels if label == 1])
    count = len(set(labels)) - (1 if -1 in labels else 0)
    return count

# print(f"Estimated object instances: {count_multiple_matches(scene_uri, object_uri)}")
# for dbscan
#Output: Estimated object instances: 3
# for hdbscan based on similarity
# len of labels=24 and labels=[ 1 -1 -1  1 -1  1  0 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0 -1  1 -1]
# Estimated object instances: 2

# Statically defined user functions for fast reference
user_functions: Set[Callable[..., Any]] = {
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
    count_multiple_matches
}