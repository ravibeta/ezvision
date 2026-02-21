#! /usr/bin/python
import json
import sys
import os
import requests
import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient  

from azure.search.documents.models import (
    QueryCaptionResult,
    QueryType,
    VectorizedQuery,
    VectorQuery,
)
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
red_car_sas_url = os.getenv("AZURE_RED_CAR_SAS_URL").strip('"')
credential = AzureKeyCredential(search_api_key)

search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)
vector = vectorize_image(red_car_sas_url, vision_api_key, "eastus")
vector = np.pad(vector, (0, 1536 - len(vector)), mode='constant')
# vector_query = VectorQuery(vector=vector,
                              # k=1000,
                              # fields = "image_vector")  
vector_query = {
    "vector": vector.tolist(),
    "k": 1000,  # retrieve up to 1000 matches, adjust as needed
    "fields": "vector",
    "kind": "vector",
    "exhaustive": True
}
results = search_client.search(
    search_text="",
    vector_queries= [vector_query],
    select=["id"],
    include_total_count=True,
    top=1000,
)
if results:
    if results.get_count() != 1000
        print(f"Number of results: {results.get_count()}")
    items=[]
    for result in results:
        if result:
        line = result["id"]
        id = int(line)
        match = False
        for item in items:
           if item < id and id-item < 1000:
              match = True
              break
           if item > id and item-id < 1000:
              match = True
              break
        if match == False:
           items+=[id]
    for item in items:
        print(f"{item:06d}")
"""
Sample output:
008812 # invalid match no red car, all others have red cars
011214
015644
006475
010111
007526
017760
012911
014182
016672 # only red car, full precision
004559
003267
000601
001680
"""