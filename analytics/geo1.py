import requests
import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
from pprint import pprint

# === Azure Computer Vision credentials ===
vision_api_key = os.getenv("AZURE_AI_VISION_API_KEY")
vision_endpoint = os.getenv("AZURE_AI_VISION_ENDPOINT")
computervision_client = ComputerVisionClient(vision_endpoint, CognitiveServicesCredentials(vision_api_key))
scene_url = os.getenv("CIRCULAR_BUILDING_SAS_URL").strip('"')

def get_landmark_info(image_path_or_url):
    """
    Detects landmarks in an aerial image and returns detailed metadata.
    Supports both local file paths and image URLs.
    """
    visual_features = [VisualFeatureTypes.categories, VisualFeatureTypes.description, VisualFeatureTypes.tags]

    if image_path_or_url.startswith("http"):
        analysis = computervision_client.analyze_image(image_path_or_url, visual_features)
    else:
        with open(image_path_or_url, "rb") as image_stream:
            analysis = computervision_client.analyze_image_in_stream(image_stream, visual_features)

    # Extract landmark-related tags and descriptions
    landmark_tags = [tag.name for tag in analysis.tags if "landmark" in tag.name.lower()]
    description = analysis.description.captions[0].text if analysis.description.captions else "No description available"

    result = {
        "description": description,
        "landmark_tags": landmark_tags,
        "categories": [cat.name for cat in analysis.categories]
    }

    return result

# Example usage
if __name__ == "__main__":
    landmark_data = get_landmark_info(scene_url)
    pprint(landmark_data)


### output:
# {'categories': ['abstract_', 'others_', 'outdoor_', 'text_sign'],
#  'description': 'graphical user interface',
#  'landmark_tags': []}