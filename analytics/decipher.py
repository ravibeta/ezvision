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


# Initialize SearchClient
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_api_key)
)

# Retrieve the first 10 entries from the index
search_results = search_client.search("*", select=["id", "description", "vector"], top=10)

# Process entries and shred descriptions
flat_list = []

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

for entry in search_results:
    entry_id = entry["id"]
    width = 0
    height = 0
    tags = ""
    title = ""
    description_text = prepare_json_string_for_load(entry["description"]).replace('""','')

    description_json = json.loads(description_text)

    if description_json and description_json["description"]:
        title = description_json["description"]
    if description_json and description_json["_data"] and description_json["_data"]["tagsResult"] and description_json["_data"]["tagsResult"]["values"]:
        tags = ','.join([tag["name"] for tag in description_json["_data"]["tagsResult"]["values"]]).strip(",")
    if description_json and description_json["_data"] and description_json["_data"]["denseCaptionsResult"] and description_json["_data"]["denseCaptionsResult"]["values"]:
        for item in description_json["_data"]["denseCaptionsResult"]["values"]:
            text = item.get("text", "")
            bounding_box = item.get("boundingBox", {
                "x": 0,
                "y": 0,
                "w": 0,
                "h": 0
            })
            flat_list.append({
                "id": entry_id,
                "text": text,
                "bounding_box": to_string(bounding_box),
                "tags" : tags,
                "title": title
            })
    else:
        print(f"Nothing found in entry with id:{id}")
      

# Print the flattened list
print(len(flat_list))