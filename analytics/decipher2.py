import json
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from collections import deque
import os
import re
search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
api_version = os.getenv("AZURE_SEARCH_API_VERSION")
search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "index00")
dest_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "index00")
credential = AzureKeyCredential(search_api_key)

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
    
page_size = 10
skip = 0
total = 17833
while True:
    # Retrieve the first 10 entries from the index
    search_results = search_client.search("*", select=["id", "description", "vector"], top=page_size, skip = skip, include_total_count=True)
    # Process entries and shred descriptions
    flat_list = []
    if search_results.get_count() == 0  or skip > 10670:
        break
    for entry in search_results:
        entry_id = entry["id"]
        width = 0
        height = 0
        tags = ""
        title = ""
        objects = []
        description_text = prepare_json_string_for_load(entry["description"]).replace('""','')
        description_json = None
        try:
            description_json = json.loads(description_text)
        except Exception as e:
            print(f"{entry_id}: parsing error: {e}")
        if description_json == None:
            continue
        if description_json and description_json["description"]:
            title = description_json["description"]
        if description_json and description_json["_data"] and description_json["_data"]["tagsResult"] and description_json["_data"]["tagsResult"]["values"]:
            tags = ','.join([tag["name"] for tag in description_json["_data"]["tagsResult"]["values"]]).strip(",")
        if description_json and description_json["_data"] and description_json["_data"]["denseCaptionsResult"] and description_json["_data"]["denseCaptionsResult"]["values"]:
            for item in description_json["_data"]["denseCaptionsResult"]["values"]:
                text = item.get("text", "")
                objects += [text]
                # bounding_box = item.get("boundingBox", {
                    # "x": 0,
                    # "y": 0,
                    # "w": 0,
                    # "h": 0
                # })
                # flat_list.append({
                    # "id": entry_id,
                    # "text": text,
                    # "bounding_box": to_string(bounding_box),
                    # "tags" : tags,
                    # "title": title
                # })
        # else:
            # print(f"Nothing found in entry with id:{id}")
        flat_list.append({
        "id": entry_id,
        "objects": ','.join(objects).strip(","),
        "tags" : tags,
        "title": title
        })
          

    if len(flat_list) != 0:
        merge_results = destination_client.merge_documents(flat_list)
        error = ','.join([merge_result.error_message for merge_result in merge_results if merge_result.error_message]).strip(",")
        if error:
            print(error)
        # if len([merge_result.succeeded for merge_result in merge_results if merge_result.succeeded]) == page_size:
        else:
                print(f"success in merging entries with id: {skip} to {skip + page_size}")
    skip += page_size