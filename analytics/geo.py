import requests
import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image

# === Azure Computer Vision credentials ===
vision_api_key = os.getenv("AZURE_AI_VISION_API_KEY")
vision_endpoint = os.getenv("AZURE_AI_VISION_ENDPOINT")
computervision_client = ComputerVisionClient(vision_endpoint, CognitiveServicesCredentials(vision_api_key))

# === Azure Maps credentials ===
azure_maps_key = os.getenv("AZURE_MAPS_SUBSCRIPTION_KEY")

# === Load local image and get tags ===
image_path = "frame5.jpg"
with open(image_path, "rb") as img_stream:
    analysis = computervision_client.analyze_image_in_stream(
        img_stream,
        visual_features=["Tags"]
    )

tags = [tag.name for tag in analysis.tags if tag.confidence > 0.5]

# === Azure Maps Search API for landmark coordinates ===
def get_coordinates_from_azure_maps(landmark, azure_key):
    url = f"https://atlas.microsoft.com/search/address/json"
    params = {
        "api-version": "1.0",
        "subscription-key": azure_key,
        "query": landmark
    }
    response = requests.get(url, params=params)
    data = response.json()
    results = data.get("results", [])
    if results:
        position = results[0]["position"]
        return (position["lat"], position["lon"])
    return None
tags = ["circular plaza"]
# === Display matched coordinates ===
for tag in tags:
    coords = get_coordinates_from_azure_maps(tag, azure_maps_key)
    if coords:
        print(f"Landmark: {tag}, Latitude: {coords[0]}, Longitude: {coords[1]}")
    else:
        print(f"No match found for tag: {tag}")

"""
Output:
Landmark: outdoor, Latitude: 39.688359, Longitude: -84.235051
Landmark: text, Latitude: 17.9739757, Longitude: -76.7856201
Landmark: building, Latitude: 23.3531395, Longitude: -75.0597782
Landmark: car, Latitude: 18.5366554, Longitude: -72.4020263
Landmark: urban design, Latitude: 48.4732981, Longitude: 35.0019145
Landmark: metropolitan area, Latitude: 55.6033166, Longitude: 13.0013362
Landmark: urban area, Latitude: 8.448839, Longitude: -13.258005
Landmark: neighbourhood, Latitude: 54.8811412, Longitude: -6.2779797
Landmark: intersection, Latitude: 34.899284, Longitude: -83.392743
Landmark: vehicle, Latitude: 38.6151446, Longitude: -121.273215
Landmark: residential area, Latitude: 9.982962, Longitude: 76.2954466
Landmark: city, Latitude: 19.4326773, Longitude: -99.1342112
Landmark: traffic, Latitude: 23.5786896, Longitude: 87.1950397
Landmark: street, Latitude: 51.1250213, Longitude: -2.7313088
Landmark: aerial, Latitude: 34.95435, Longitude: -117.826011

# 
# Not even close to the nearest neigbhorhood: https://www.google.com/maps?q=42.3736,-71.1097
"""