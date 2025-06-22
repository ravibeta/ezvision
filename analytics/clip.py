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
source_url_template = os.getenv("AZURE_SOURCE_SAS_URI")
destination_url_template = os.getenv("AZURE_DESTINATION_SAS_URI")
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
    response = requests.get(sas_url)
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
    

def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)
   
def euclidean_distance(vec1, vec2):
    """Computes Euclidean distance between two vectors."""
    return np.linalg.norm(np.array(vec1) - np.array(vec2))
    
def is_closest_match(destination_client, vector):
    vector_query = VectorizedQuery(vector=vector,
                                  k_nearest_neighbors=3,
                                  exhaustive=True,
                                  fields = "vector")  

    results = search_client.search(  
        search_text=None,  
        vector_queries= [vector_query],
        select=["id", "description","vector"],  
        # select='id,description,vector',
        include_total_count=True,
        top=4
    ) 
    if results != None and results.get_count() > 0:
        best = 0
        for match in results:
            print(f"{match['id']} found." + ",".join([key for key in match.keys()]))
            match_vector = match["vector"]
            score = cosine_similarity(vector, match_vector)
            if score > best:
                best = score
                print("similarity > 80%")
            else:
                print("similarity < 80%")
        if best > 0.8:
           print("match found.")
           return True
    else:
        print("no match found.")
    return False

def is_visited_by_url(deduplicator, image_url):
    image = None
    try:
        image = read_image_from_blob(image_url)
    except Exception as e:
        print(f"Error: {e}")
    if image.any():
        value = deduplicator.is_duplicate(image)
        # print(f"is_duplicate={value}")
        return value
    # print("is not a duplicate")
    return False

def is_duplicate_image(deduplicator, image):
    value = deduplicator.is_duplicate(image)
    # print(f"is_duplicate={value}")
    return value
    
def is_visited(deduplicator, vector):
    value = deduplicator.is_visited(vector)
    # print(f"is_visited={value}")
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
        # Retrieve the first 10 entries from the index
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
                if objectid == 1 or objectid != 8:
                    continue
                destination_file=source_file+f"-{objectid:04d}"
                destination_sas_url = destination_url_template.replace("{destination_file}", destination_file)
                # print(f"Destination url={destination_sas_url}")
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
                    if vector.any() and is_closest_match(destination_client, vector) == False:
                        print("found no match for {destination_file}")
                    else:
                        print("found match for {destination_file}")
                else:
                    print("no objects detected")
                break
for number in range(1, 2):
    entry_id = f"{number:06d}"
    shred(entry_id)
