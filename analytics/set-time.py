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


# Initialize SearchClient
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_api_key)
)

from datetime import datetime, timedelta

def get_timestamp(i):
    """
    Prints ten consecutive timestamps in ISO 8601 format (yyyy-mm-ddThh:mm:ss)
    with a 2-second interval between each.
    """
    # Get the current time to use as the starting point
    current_time = datetime.now()

	# Calculate the timestamp for this iteration:
	# start time + (iteration number * interval duration)
    timestamp = current_time + timedelta(seconds=i * 2)

	# Format the datetime object into the required string format (ISO 8601)
    formatted_timestamp = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

	# Print the result
    return formatted_timestamp

for i in range(0,27):
	target_id = f"2-{i:04d}"
	# Retrieve the first 10 entries from the index
	entry = search_client.get_document(key=target_id) # , select=["id", "description"])
	print(f"id={entry["id"]}")
	entry["created"] = get_timestamp(i)
	merge_results = search_client.merge_documents([entry])
	if merge_results:
		print(f"{merge_results[0].succeeded}")
		print(f"{merge_results[0].error_message}")
		print(f"created={entry.get("created")}")
	else:
		print(f"Merge failed for document with key: {target_id}")
	#"""