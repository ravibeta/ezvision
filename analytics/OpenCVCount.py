#! /usr/bin/python
import requests
import cv2
import numpy as np
from io import BytesIO
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
import os

# Azure Vision credentials
vision_endpoint=os.getenv("AZURE_AI_VISION_ENDPOINT")
vision_api_key = os.getenv("AZURE_AI_VISION_API_KEY")
image_uri = os.getenv("AZURE_RED_CAR_2_SAS_URL").strip('"')
image_dataset_uri = os.getenv("AZURE_QUERY_SAS_URI").strip('"')

# Step 1: Download images from SAS URLs
def download_image(url):
    response = requests.get(url)
    image_array = np.frombuffer(response.content, np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

scene_img = download_image(image_dataset_uri)
object_img = download_image(image_uri)

# Step 2: Use OpenCV template matching to find object occurrences
def count_object_occurrences(scene, template, threshold=0.8):
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(scene_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    w, h = template_gray.shape[::-1]
    rects = [[pt[0], pt[1], pt[0] + w, pt[1] + h] for pt in zip(*locations[::-1])]
    rects, _ = cv2.groupRectangles(rects, groupThreshold=1, eps=0.5)
    return len(rects)

# Step 3: Count matches
count = count_object_occurrences(scene_img, object_img)
print(f"Detected {count} occurrences of the object.")
