import json
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.search.documents.models import (
    VectorizedQuery,
    VectorizableTextQuery
)
from dedup import ImageDeduplicator
from tenacity import retry, stop_after_attempt, wait_fixed
import os
import re
import sys
import time
search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
api_version = os.getenv("AZURE_SEARCH_API_VERSION")
search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "index00")
credential = AzureKeyCredential(search_api_key)
dest_index_name = os.getenv("AZURE_SEARCH_02_INDEX_NAME", "index02")
vision_api_key = os.getenv("AZURE_AI_VISION_API_KEY")
vision_api_version = os.getenv("AZURE_AI_VISION_API_VERSION")
vision_region = os.getenv("AZURE_AI_VISION_REGION")
vision_endpoint =  os.getenv("AZURE_AI_VISION_ENDPOINT")
source_url_template = os.getenv("AZURE_SOURCE_SAS_URI", "https://saravinoteblogs.blob.core.windows.net/playground/vision/main/main/{source_file}.jpg?sp=racwdlmep&st=2025-05-31T18:03:20Z&se=2025-08-01T02:03:20Z&spr=https&sv=2024-11-04&sr=d&sig=oPk6yFDyPlQDwJL4Wb4f7y5Vd1bJA%2BRSOenW%2FNbQIJo%3D&sdd=3")
destination_url_template = os.getenv("AZURE_DESTINATION_SAS_URI", "https://saravinoteblogs.blob.core.windows.net/playground/vision/main/test/{destination_file}.jpg?sp=racwdlmep&st=2025-06-17T23:36:20Z&se=2025-06-30T07:36:20Z&spr=https&sv=2024-11-04&sr=d&sig=fVpjwD5VoTL0QPbMn2q1vytBgEGwCenCYEnO4bX7730%3D&sdd=3")
#entry_id = "003190" # "003401" 
sys.path.insert(0, os.path.abspath(".."))
from visionprocessor.vectorizer import vectorize_image, analyze_image
page_size = 10
skip = 3184
total = 17833
deduplicator = ImageDeduplicator()
# Initialize SearchClient
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_api_key)
)
destination_client = SearchClient(
    endpoint=search_endpoint,
    index_name=dest_index_name,
    credential=AzureKeyCredential(search_api_key)
)
vision_credential = AzureKeyCredential(vision_api_key)
analysis_client = ImageAnalysisClient(vision_endpoint, vision_credential)

import cv2
import numpy as np
import requests
from io import BytesIO
from azure.storage.blob import BlobClient

def read_image_from_blob(sas_url):
    """Reads an image from Azure Blob Storage using its SAS URL."""
    response = None
    try:
        response = requests.get(sas_url)
    except Exception as e: 
        print(f"Error from requests.get: {e}")
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    else:
        # raise Exception(f"Failed to fetch image. Status code: {response.status_code}")
        return None

def upload_image_to_blob(clipped_image, sas_url):
    """Uploads the clipped image to Azure Blob Storage using its SAS URL."""
    _, encoded_image = cv2.imencode(".jpg", clipped_image)
    blob_client = BlobClient.from_blob_url(sas_url)
    blob_client.upload_blob(encoded_image.tobytes(), overwrite=True)
    # print("Clipped image uploaded successfully.")
    
def save_or_display(clipped_image, destination_file):
    cv2.imwrite(destination_file, clipped_image)
    cv2.imshow("Clipped Image", clipped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clip_image(image, bounding_box):
    # Extract bounding box parameters
    x, y, width, height = bounding_box

    # Clip the region using slicing
    clipped_image = image[y:y+height, x:x+width]

    return clipped_image

def prepare_json_string_for_load(text):
  text = text.replace("\"", "'")
  text = text.replace("{'", "{\"")
  text = text.replace("'}", "\"}")
  text = text.replace(" '", " \"")
  text = text.replace("' ", "\" ")
  text = text.replace(":'", ":\"")
  text = text.replace("':", "\":")
  text = text.replace(",'", ",\"")
  text = text.replace("',", "\",")
  return re.sub(r'\n\s*', '', text)
  
def to_string(bounding_box):
    return f"{bounding_box['x']},{bounding_box['y']},{bounding_box['w']},{bounding_box['h']}"
    

def is_duplicate_image(deduplicator, image):
    value = deduplicator.is_duplicate(image)
    return value
    
def is_visited(deduplicator, vector):
    value = deduplicator.is_visited(vector)
    return value
    
def is_existing(deduplicator, vector):
    start_time = time.time()
    value = deduplicator.is_existing(destination_client, vector)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for is_existing: {elapsed_time:.3f} seconds")
    return value

@retry(stop=stop_after_attempt(5), wait=wait_fixed(60))
def upload(document):
    try:
        upload_results = destination_client.upload_documents([document])
        error = ','.join([upload_result.error_message for upload_result in upload_results if upload_result.error_message]).strip(",")
        if error:
            print(error)
    except HttpResponseError as e:
        print(f"Error from upload: {e}")
        raise    
        
# Example usage
def shred(entry_id):
        source_file=entry_id
        source_sas_url = source_url_template.replace("{source_file}", source_file)
        print(entry_id)
        entry = search_client.get_document(key=entry_id) # , select=["id", "description"])
        id=entry['id']
        description_text=entry['description']
        tags = entry['tags']
        title = entry['title']
        description_json = None
        try:
            description_text = prepare_json_string_for_load(entry["description"]).replace('""','')
            description_json = json.loads(description_text)
        except Exception as e:
            print(description_text)
            print(f"{entry_id}: parsing error: {e}")
        if description_json == None:
            print("Description could not be parsed.")
            return
        if description_json and description_json["_data"] and description_json["_data"]["denseCaptionsResult"] and description_json["_data"]["denseCaptionsResult"]["values"]:
            objectid = 0            
            for item in description_json["_data"]["denseCaptionsResult"]["values"]:
                objectid += 1
                if objectid == 1:
                    continue
                destination_file=source_file+f"-{objectid:04d}"
                destination_sas_url = destination_url_template.replace("{destination_file}", destination_file)
                box = item.get("boundingBox", None)
                print(f"{destination_file}: {box}")
                if box:
                    bounding_box = (box["x"], box["y"], box["w"], box["h"])

                    # Read image from Azure Blob
                    image = read_image_from_blob(source_sas_url)
                    if image.any() == False:
                       print(f"{destination_file} not found.")
                       continue

                    # Clip image
                    clipped = clip_image(image, bounding_box)

                    # Upload clipped image to Azure Blob
                    upload_image_to_blob(clipped, destination_sas_url)
                    vector = vectorize_image(destination_sas_url, vision_api_key, "eastus")
                    vector = np.pad(vector, (0, 1536 - len(vector)), mode='constant')
                    print("checking existing")
                    if vector.any() and is_existing(deduplicator, vector) == False:
                        print(f"Match does not exist for {destination_file}.")
                    else:
                        print(f"Match exists for {destination_file}")
                else:
                    print("no objects detected")
for number in range(16004, 16005):
    entry_id = f"{number:06d}"
    shred(entry_id)
