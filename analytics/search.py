import requests
import json
import sys
import os
import numpy as np
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
blob_url = os.getenv("AZURE_QUERY_SAS_URI")
vector = vectorize_image(blob_url, vision_api_key, "eastus")
vector = np.pad(vector, (0, 1536 - len(vector)), mode='constant')
# print(f"len={len(vector)}")
# Vector search payload
body =     {
        "count": True,
        "select": "id,description,vector",
        "vectorQueries": [
            {
                "vector": vector.tolist(),
                "k": 5,
                "fields": "vector",
                "kind": "vector",
                "exhaustive": True
            }
        ]
    }

# Headers for Azure Search API
headers = {
    "Content-Type": "application/json",
    "api-key": search_api_key
}

# Send search request to Azure AI Search
response = requests.post(
    f"{search_endpoint}/indexes/{index_name}/docs/search?api-version=2024-07-01",
    headers=headers,
    data=json.dumps(body)
)

# Parse response
search_results = response.json()
print(len(search_results))
print(search_results)
ids = ",".join([item["id"] for item in search_results.get("value", [])]).strip(",")
print(ids)
# output: 
# RedCar3: 015644,015643,012669,008812,011600
# RedCar4: 014076,014075,014077,014074,014543
# Count occurrences of "red car" in descriptions
red_car_count = sum(1 for item in search_results.get("value", []) if "red car" in item["description"].lower())

print(f"Total red cars found in drone images: {red_car_count}")
