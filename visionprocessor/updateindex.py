from azure.search.documents import SearchClient, IndexDocumentsBatch
from azure.core.credentials import AzureKeyCredential

# Azure AI Search configurations
AZURE_SEARCH_ENDPOINT = "https://your-search-service.search.windows.net"
AZURE_SEARCH_INDEX = "your-index-name"
AZURE_SEARCH_API_KEY = "your-api-key"

# Initialize SearchClient
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

# Retrieve all documents from the index
documents = search_client.search("*", select=["id"])

# Prepare update requests
update_batch = [{"@search.action": "merge", "id": doc["id"], "color_field": "green"} for doc in documents]

# Update documents in batches
if update_batch:
    search_client.index_documents(update_batch)
    print(f"Updated {len(update_batch)} documents with 'green'.")
else:
    print("No documents found to update.")
