import json
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import os
import re
search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
api_version = os.getenv("AZURE_SEARCH_API_VERSION")
search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "index007")
credential = AzureKeyCredential(search_api_key)
target_id = "2-0026" 

# Initialize SearchClient
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_api_key)
)

# Retrieve the first 10 entries from the index
entry = search_client.get_document(key=target_id) # , select=["id", "description"])
print(f"id={entry["id"]}")
print(f"location={entry.get("location")}")
#"""
entry["location"] = "42.37194286224507, -71.11863940498394"
merge_results = search_client.merge_documents([entry])
if merge_results:
	print(f"{merge_results[0].succeeded}")
	print(f"{merge_results[0].error_message}")
	print(f"location={entry.get("location")}")
else:
	print(f"Merge failed for document with key: {target_id}")
#"""