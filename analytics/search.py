import requests
import json
import sys
import os
# Add the parent folder to the module search path
sys.path.insert(0, os.path.abspath(".."))
from visionprocessor.vectorizer import vectorize_image

# Azure AI Search configurations
search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")  
vision_api_key = os.getenv("AZURE_AI_VISION_API_KEY")

# Query string for red cars
query_text = "Find red cars in drone images"
blob_url = "https://saravinoteblogs.blob.core.windows.net/playground/vision/query/RedCar4.jpg?sp=racwdle&st=2025-05-26T23:54:09Z&se=2025-05-27T07:54:09Z&spr=https&sv=2024-11-04&sr=d&sig=9RRmmtlBnEiFsOGHJ2d%2ByEkBz2gxXOrQEc%2B4uf%2Fd6ao%3D&sdd=2"
vector = vectorize_image(blob_url, vision_api_key, "eastus")
print(f"len={len(vector)}")
# Vector search payload
search_payload = {
    "search": query_text,
    "vector": vector,  # Example query vector
    "vectorFields": "vector",
    "searchFields": "description",
    "select": "id, description",
    "top": 10
}

# Headers for Azure Search API
headers = {
    "Content-Type": "application/json",
    "api-key": search_api_key
}

# Send search request to Azure AI Search
response = requests.post(
    f"{search_endpoint}/indexes/{index_name}/docs/search?api-version=2023-07-01",
    headers=headers,
    data=json.dumps(search_payload)
)

# Parse response
search_results = response.json()

# Count occurrences of "red car" in descriptions
red_car_count = sum(1 for item in search_results.get("value", []) if "red car" in item["description"].lower())

print(f"Total red cars found in drone images: {red_car_count}")
