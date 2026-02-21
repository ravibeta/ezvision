#! /usr/bin/python

#from azure.ai.vision import VisionClient
from azure.core.credentials import AzureKeyCredential
from azure.core.rest import HttpRequest, HttpResponse
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures, ImageAnalysisResult
from tenacity import retry, stop_after_attempt, wait_fixed
from pprint import pprint, pformat
from dotenv import load_dotenv  
import json  
import requests
import http.client, urllib.parse
import os

load_dotenv()  
search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
search_api_version = os.getenv("AZURE_SEARCH_API_VERSION")
search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")  
vision_api_key = os.getenv("AZURE_AI_VISION_API_KEY")
vision_api_version = os.getenv("AZURE_AI_VISION_API_VERSION")
vision_region = os.getenv("AZURE_AI_VISION_REGION")
vision_endpoint =  os.getenv("AZURE_AI_VISION_ENDPOINT")
credential = DefaultAzureCredential()
#search_credential = AzureKeyCredential(search_api_key)
vision_credential = AzureKeyCredential(vision_api_key)

# Initialize Azure clients
#vision_client = VisionClient(endpoint=vision_endpoint, credential=AzureKeyCredential(vision_api_key))
search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=AzureKeyCredential(search_api_key))
analysis_client = ImageAnalysisClient(vision_endpoint, vision_credential)

# Define SAS URL template
sas_template = "https://saravinoteblogs.blob.core.windows.net/playground/vision/main/main/{file}.jpg?sp=rle&st=2025-05-17T02:49:13Z&se=2025-05-31T10:49:13Z&spr=https&sv=2024-11-04&sr=d&sig=GqhakFHijZWlIUTsRvaVWIqA2EHA70tzaYMDK%2BhC31A%3D&sdd=3"

# Process images in batches of 10
batch_size = 10
initial_start = 17854
total_images = 17853  # Adjust this as needed


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
