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
target_id = "012112" # "003184" # "003401" 

# Initialize SearchClient
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=index_name,
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
    
# Retrieve the first 10 entries from the index
entry = search_client.get_document(key=target_id) # , select=["id", "description"])
for key in entry.keys():
    print(key)
print(f"id={entry['id']}")
print(f"description={entry['description']}")
print(f"objects={entry['objects']}")
print(f"tags={entry['tags']}")
print(f"title={entry['title']}")
# # Process entries and shred descriptions
# if entry:
    # entry_id = entry["id"]
    # width = 0
    # height = 0
    # tags = ""
    # title = ""
    # if (entry_id == "003401"):
        # print(f"description={entry["description"]}")
    # description_text = prepare_json_string_for_load(entry["description"]).replace('""','')
    # description_json = None
    # try:
        # description_json = json.loads(description_text)
    # except Exception as e:
        # print(f"{entry_id}: parsing error: {e}")
    # if description_json == None:
        # print("Exiting")