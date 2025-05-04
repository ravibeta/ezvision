from dotenv import load_dotenv,dotenv_values
import json
import os
import requests
import http.client, urllib.parse
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv  
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import (
    RawVectorQuery,
)
from azure.search.documents.indexes.models import (  
 
    ExhaustiveKnnParameters,  
    ExhaustiveKnnVectorSearchAlgorithmConfiguration,
    HnswParameters,  
    HnswVectorSearchAlgorithmConfiguration,
    SimpleField,
    SearchField,  
    SearchFieldDataType,  
    SearchIndex,  
    VectorSearch,  
    VectorSearchAlgorithmKind,  
    VectorSearchProfile,  
)

from IPython.display import Image, display
load_dotenv()  
service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
api_version = os.getenv("AZURE_SEARCH_API_VERSION")
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")  
aiVisionApiKey = os.getenv("AZURE_AI_VISION_API_KEY")  
aiVisionRegion = os.getenv("AZURE_AI_VISION_REGION")
aiVisionEndpoint = os.getenv("AZURE_AI_VISION_ENDPOINT")
credential = AzureKeyCredential(key)

search_client = SearchClient(endpoint=service_endpoint, 
                             index_name=index_name, 
                             credential=credential)


def generate_embeddings(text, aiVisionEndpoint, aiVisionApiKey):  
    url = f"{aiVisionEndpoint}/computervision/retrieval:vectorizeText"  
  
    params = {  
        "api-version": "2023-02-01-preview"  
    }  
  
    headers = {  
        "Content-Type": "application/json",  
        "Ocp-Apim-Subscription-Key": aiVisionApiKey  
    }  
  
    data = {  
        "text": text  
    }  
  
    response = requests.post(url, params=params, headers=headers, json=data)  
  
    if response.status_code == 200:  
        embeddings = response.json()["vector"]  
        return embeddings  
    else:  
        print(f"Error: {response.status_code} - {response.text}")  
        return None  
    
query = "truck"

vector_text = generate_embeddings(query, aiVisionEndpoint, aiVisionApiKey)

vector_query = RawVectorQuery(vector=vector_text,
                              k=3, 
                              fields="image_vector")  

# Perform vector search  
results = search_client.search(  
    search_text=None,  
    vector_queries= [vector_query],
    select=["description"]  
)  

FILE_PATH = os.getenv("FILE_PATH") or 'data/frames/main' 
DIR_PATH = os.path.join(os.getcwd(), FILE_PATH)
for result in results:
    print(f"{result['description']}")
    display(Image(DIR_PATH + "/" + result["description"]))
    print("\n") 