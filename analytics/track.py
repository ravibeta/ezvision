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
object_uri = os.getenv("AZURE_RED_CAR_2_SAS_URL").strip('"')
scene_uri = os.getenv("AZURE_QUERY_SAS_URI").strip('"')

import torch

# Load a YOLOv5 pre-trained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_vehicles(frame):
    results = model(frame)
    # Keep only 'car', 'truck', 'bus', 'motorcycle' detections
    vehicle_labels = ['car', 'truck', 'bus', 'motorcycle']
    detections = results.pandas().xyxy[0]
    vehicles = detections[detections['name'].isin(vehicle_labels)]
    return vehicles

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
        
vehicles = detect_vehicles(read_image_from_blob(scene_uri))
print(vehicles)