import numpy as np
from azure.search.documents import SearchClient, IndexDocumentsBatch
from azure.core.credentials import AzureKeyCredential
import os
# Azure AI Search configurations
search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  
index_name = os.getenv("AZURE_SEARCH_NEW_INDEX_NAME")
destination_name = os.getenv("AZURE_SEARCH_1024_INDEX_NAME")
api_version = os.getenv("AZURE_SEARCH_API_VERSION")
search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
credential = AzureKeyCredential(search_api_key)


# Initialize search clients
source_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=AzureKeyCredential(search_api_key))
destination_client = SearchClient(endpoint=search_endpoint, index_name=destination_name, credential=AzureKeyCredential(search_api_key))

# Retrieve documents from source index
source_documents = source_client.search("*", select=["id", "description", "vector"])

# Process documents in batches
batch_size = 10
buffer = []
start = 0
end = 0

for doc in source_documents:
    original_vector = np.array(doc["vector"])

    # Expand vector to 1536 dimensions (zero-padding)
    # expanded_vector = np.pad(original_vector, (0, 1536 - len(original_vector)), mode='constant')
    truncated_vector = original_vector[:1024]
    # Prepare document for destination index
    buffer.append({
        "id": doc["id"],
        "description": doc["description"],
        "vector": truncated_vector.tolist()
    })
    end += 1
    # Upload in batches of 10
    if len(buffer) == batch_size:
        destination_client.upload_documents(buffer)
        print(f"Uploaded {len(buffer)} documents from {start} to {end}.")
        buffer = []  # Reset batch list
        start = end

# Upload remaining documents
if buffer:
    destination_client.upload_documents(buffer)
    print(f"Uploaded final batch of {len(buffer)} documents from {start} to {end}.")

print("Vector transformation and upload completed successfully.")
