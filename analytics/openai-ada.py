#! /usr/bin/python
import json
import sys
import os
import requests
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient 
from azure.search.documents.models import VectorizedQuery  
sys.path.insert(0, os.path.abspath(".."))
from visionprocessor.vectorizer import vectorize_image
search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
api_version = os.getenv("AZURE_SEARCH_API_VERSION")
search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
vision_api_key = os.getenv("AZURE_AI_VISION_API_KEY")
vision_api_version = os.getenv("AZURE_AI_VISION_API_VERSION")
vision_region = os.getenv("AZURE_AI_VISION_REGION")
vision_endpoint =  os.getenv("AZURE_AI_VISION_ENDPOINT")
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_gpt_deployment = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4o-mini")
azure_openai_gpt_model = os.getenv("AZURE_OPENAI_GPT_MODEL", "gpt-4o-mini")
azure_openai_embedding_api = os.getenv("AZURE_OPENAI_EMBEDDING_API")
azure_openai_embedding_api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
azure_openai_embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
credential = AzureKeyCredential(search_api_key)

search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://your-openai-resource.openai.azure.com
OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_DEPLOYMENT = "text-embedding-ada-002"

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")  # e.g. https://your-search-resource.search.windows.net
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
SEARCH_INDEX = "drone-images-index"
VECTOR_FIELD = "imageVector"

# === STEP 1: Embed the query using text-embedding-ada-002 ===
def get_text_embedding(query):
    url = azure_openai_embedding_api
    print(azure_openai_embedding_api_key)
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_openai_embedding_api_key
    }
    payload = {
        "input": query
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

# === STEP 2: Search Azure AI Search with the query vector ===
def search_drone_images(query_text, top_k=5):
    embedding = get_text_embedding(query_text)
    print(f"len of embedding={len(embedding)}")
    client = SearchClient(endpoint=search_endpoint,
                          index_name=index_name,
                          credential=credential)
    vectorized_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k, fields="vector", exhaustive=True)
    results = client.search(
        search_text=None,
        vector_queries=[vectorized_query],
        top=top_k,
        semantic_configuration_name = "mysemantic",
        include_total_count=True,
    )

    print(f"\n🔍 Top {top_k} matches for: \"{query_text}\"")
    for i, result in enumerate(results):
        print(f"{i+1}. ID: {result['id']} | Score: {result['@search.score']:.4f} | IMG: {result['id']}.jpg")

# === EXAMPLE USAGE ===
if __name__ == "__main__":
    # query = "How many red cars are parked near the building with a circular roof structure?"
    # query = "Do bicycles have dedicated green street crossings at intersections?"
    # query = "Are there tall buildings with lots of storeys?"
    # query = "Is there a lake or water body in the aerial images?"
    # query = "Are there blue or white or black cars in parking lots or roofs?"
    query = "Are there buildings with lots of fans on the roofs?"
    search_drone_images(query)

"""
Answer 1:
015614.jpg
015584.jpg
015587.jpg
015612.jpg
015581.jpg

Answer 2:
Top 5 matches for: "Do bicycles have dedicated green street crossings at intersections?"
1. ID: 015614 | Score: 0.5055 | IMG: 015614.jpg
2. ID: 015584 | Score: 0.5053 | IMG: 015584.jpg
3. ID: 015595 | Score: 0.5050 | IMG: 015595.jpg
4. ID: 015602 | Score: 0.5048 | IMG: 015602.jpg
5. ID: 015612 | Score: 0.5048 | IMG: 015612.jpg

Answer 3:
Top 5 matches for: "Are there tall buildings with lots of storeys?"
1. ID: 015595 | Score: 0.5024 | IMG: 015595.jpg
2. ID: 003488 | Score: 0.5023 | IMG: 003488.jpg
3. ID: 015614 | Score: 0.5023 | IMG: 015614.jpg
4. ID: 015584 | Score: 0.5022 | IMG: 015584.jpg
5. ID: 015606 | Score: 0.5021 | IMG: 015606.jpg

Answer 4:
Top 5 matches for: "Is there a lake or water body in the aerial images?"
1. ID: 015614 | Score: 0.5080 | IMG: 015614.jpg
2. ID: 015877 | Score: 0.5078 | IMG: 015877.jpg
3. ID: 015440 | Score: 0.5075 | IMG: 015440.jpg
4. ID: 015435 | Score: 0.5075 | IMG: 015435.jpg
5. ID: 016028 | Score: 0.5074 | IMG: 016028.jpg

"""