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
sas_template = os.getenv("AZURE_SA_CONTAINER_SASURL")

# Process images in batches of 10
batch_size = 10
initial_start = 2040
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

@retry(stop=stop_after_attempt(5), wait=wait_fixed(60))
def analyze_image(client, image_url):
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
        result: ImageAnalysisResult = result
        if result is not None:
            captions = []
            captions += [ f"{result.caption.text}" if result.caption is not None else "No Caption"]
            captions += [ f"{caption.text}" for caption in result.dense_captions.list if result.dense_captions is not None]
            # Enhance result
            result.description = ",".join(captions)
            # print("Full ImageAnalysisResult object:")
            # pprint(result.__dict__, depth=4, compact=False)
            description =  pformat(result.__dict__, depth=4, compact=False)
            return description
    except HttpResponseError as e:
        print(str(e))
        raise
    return "No description"


for batch_start in range(initial_start, total_images + 1, batch_size):
    documents = []

    # Vectorize 100 images at a time
    batch_end = min(batch_start + batch_size, total_images + 1)
    for i in range(batch_start, batch_end):
        file_name = f"{i:06}"
        blob_url = sas_template.format(file=file_name)

        try:
            vector = vectorize_image(blob_url, vision_api_key, "eastus")
            if vector is not None:
               description = analyze_image(analysis_client, blob_url)
               documents += [
                  {"id": file_name, "description": description, "image_vector": vector}
               ]
        except Exception as e:
            print(f"Error processing {file_name}.jpg: {e}")

    # print(f"Vectorization complete for images {batch_start} to {min(batch_start + batch_size - 1, total_images)}")
    # Upload batch to Azure AI Search
    if len(documents) > 0:
        # [pprint(document, depth=4, compact=False) for document in documents]
        try:
            search_client.upload_documents(documents=documents)
            print(f"Uploaded {len(documents)} images {batch_start} to {batch_end} to {index_name}.")
        except Exception as e:
            print(f"Error {e} uploading {len(documents)} images {batch_start} to {batch_end} to {index_name}.")

print(f"Vectorized images successfully added to {index_name}!")
