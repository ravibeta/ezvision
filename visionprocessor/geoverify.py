#!/usr/bin/python
import cv2 
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import requests 

# Function to get GPS coordinates using Google Maps API 

def get_geolocation(landmark_name, api_key): 
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={landmark_name}&key={api_key}" 
    response = requests.get(url) 
    data = response.json() 
    if data["status"] == "OK": 
        location = data["results"][0]["geometry"]["location"] 
        return location["lat"], location["lng"] 
    return None 

def get_gps_data(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()

    if not exif_data:
        return None
    
    gps_info = {}
    for tag, value in exif_data.items():
        if TAGS.get(tag) == "GPSInfo":
            for key in value.keys():
                gps_info[GPSTAGS.get(key)] = value[key]
    
    return gps_info


def get_gps_location(gps_json_file, device_id):
    orig_bounds = None
	with open(gps_json_file, 'r') as f:
		orig_bounds = json.load(f)
    if orig_bounds and device_id in orig_bounds:
        return orig_bounds[device_id]
    return None
	
 
def convert_to_decimal(coord, ref):
    decimal = coord[0] + coord[1] / 60 + coord[2] / 3600
    if ref in ["S", "W"]:
        decimal *= -1
    return decimal


 
# Example usage
#image_path = "000001.jpg"
#gps_data = get_gps_data(image_path)
#if gps_data:
#   print(gps_data)
"""
{
    'GPSLatitudeRef': 'N',
    'GPSLatitude': (37, 25, 39.12),  # Degrees, Minutes, Seconds format
    'GPSLongitudeRef': 'W',
    'GPSLongitude': (122, 10, 11.64),  # Degrees, Minutes, Seconds format
    'GPSAltitudeRef': 0,
    'GPSAltitude': 30.5,  # Altitude in meters
    'GPSTimeStamp': (22, 14, 5),  # UTC time of capture (hours, minutes, seconds)
    'GPSDateStamp': '2025:05:25',  # Date of capture (YYYY:MM:DD)
}
"""
#else:
#   gps_data = get_gps_location("gps.json", "00001")
#latitude = convert_to_decimal((37, 25, 39.12), "N")
#longitude = convert_to_decimal((122, 10, 11.64), "W")
#print(f"Decimal Coordinates: Latitude {latitude}, Longitude {longitude}")
"""
Decimal Coordinates: Latitude 37.42753333333333, Longitude -122.1699
"""

