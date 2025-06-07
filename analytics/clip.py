import json
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import os
import re
search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
api_version = os.getenv("AZURE_SEARCH_API_VERSION")
search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "index00")
credential = AzureKeyCredential(search_api_key)
entry_id = "003184" # "003401" 

# Initialize SearchClient
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_api_key)
)


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
        raise Exception(f"Failed to fetch image. Status code: {response.status_code}")


def upload_image_to_blob(clipped_image, sas_url):
    """Uploads the clipped image to Azure Blob Storage using its SAS URL."""
    _, encoded_image = cv2.imencode(".jpg", clipped_image)
    blob_client = BlobClient.from_blob_url(sas_url)
    blob_client.upload_blob(encoded_image.tobytes(), overwrite=True)
    print("Clipped image uploaded successfully.")
    
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
    

# Example usage
def shred():
        source_file=entry_id
        source_sas_url = f"https://saravinoteblogs.blob.core.windows.net/playground/vision/main/main/{source_file}.jpg?sp=racwdlmep&st=2025-05-31T18:03:20Z&se=2025-08-01T02:03:20Z&spr=https&sv=2024-11-04&sr=d&sig=oPk6yFDyPlQDwJL4Wb4f7y5Vd1bJA%2BRSOenW%2FNbQIJo%3D&sdd=3"
        # Retrieve the first 10 entries from the index
        entry = search_client.get_document(key=entry_id) # , select=["id", "description"])
        for key in entry.keys():
            print(key)
        print(f"id={entry['id']}")
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
                destination_sas_url = f"https://saravinoteblogs.blob.core.windows.net/playground/vision/main/objects/{destination_file}.jpg?sp=racwdlmep&st=2025-06-07T17:05:48Z&se=2025-07-06T01:05:48Z&spr=https&sv=2024-11-04&sr=d&sig=oVKpehwqkh%2F4E%2Bt%2F5nb4aso1lR0tueSo%2FTNqzTrNvTs%3D&sdd=3"
                box = item.get("boundingBox", None)
                if box:
                    print(f"x={box['x']}")
                    print(f"y={box['y']}")
                    print(f"w={box['w']}")
                    print(f"h={box['h']}")
                    bounding_box = (box["x"], box["y"], box["w"], box["h"])

                    # Read image from Azure Blob
                    image = read_image_from_blob(source_sas_url)

                    # Clip image
                    clipped = clip_image(image, bounding_box)

                    # Upload clipped image to Azure Blob
                    print(f"destination_file={destination_file}")
                    # upload_image_to_blob(clipped, destination_sas_url)
                else:
                    print("no objects detected")
                break

shred()
