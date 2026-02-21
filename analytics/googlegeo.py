from google.cloud import vision
import io
import os
google_api_key=os.getenv("GOOGLE_CLOUD_API_KEY")
def detect_landmark(image_path):
    # Initialize the Vision API client
    client = vision.ImageAnnotatorClient()

    # Load image content
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Perform landmark detection
    response = client.landmark_detection(image=image)
    print(response)
    landmarks = response.landmark_annotations

    # Print detected landmarks and GPS coordinates
    for landmark in landmarks:
        print(f"Landmark: {landmark.description}")
        for location in landmark.locations:
            lat_lng = location.lat_lng
            print(f"Latitude: {lat_lng.latitude}, Longitude: {lat_lng.longitude}")

    if response.error.message:
        raise Exception(f"Vision API error: {response.error.message}")

# Replace with your image path
detect_landmark("frame5.jpg")
