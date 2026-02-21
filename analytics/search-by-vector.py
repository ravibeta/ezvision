#! /usr/bin/python
import json
import sys
import os
import re
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizableTextQuery
)
search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  
src_index_name = "index02" # os.getenv("AZURE_SEARCH_INDEX_NAME", "index007")
dest_index_name = "index02" # os.getenv("AZURE_SEARCH_DEST_INDEX_NAME", "index02")
api_version = os.getenv("AZURE_SEARCH_API_VERSION")
search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
credential = AzureKeyCredential(search_api_key)
# target_id = "2-0015" 
target_id = "006312-0010" # scene
target_id = "010011-0003" # object
target_id = "003804-0010" # parking lot
target_id = "009239-0009" # Another parking lot
target_id = "011546-0009" # Truck parking lot
target_id = "016293-0003" # park with trees
target_id = "016848-0007" # roof with circular structure
target_id = "009239-0009" # neighborhood with parking lot
target_id = "012657-0010" # parallel row parking
target_id = "010011-0003" # parking garage curved
# target_id = "007463-0003"
match_count = 10

# Initialize SearchClient
src_search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=src_index_name,
    credential=credential
)

target_id = "015387-0009" # scene with roof with circular structure
target_id = "003802-0010" # rows_of parked_cars
src_image = src_search_client.get_document(key=target_id) 
print(f"id={src_image.get("id")}")
print(f"description={src_image.get("description")}")
vector = src_image.get("vector")

# target_id = "015387-0010" # object with roof with circular structure
# src_image = src_search_client.get_document(key=target_id) 
# print(f"id={src_image.get("id")}")
# print(f"description={src_image.get("description")}")
# vector = src_image.get("vector")

dest_search_client = SearchClient(endpoint=search_endpoint, index_name=dest_index_name, credential=credential)
# filter = f"id ne null and startswith(cast(id,Edm.String), '{prefix}')",
# filter = f"id ne null and id ge '{prefix}-0001' and id lt '{prefix}-0010'",
search_text = "parking garage"
# exhaustive means non-hnsw search and weight=0.5 means half-text and half-vector.
vector_query = VectorizableTextQuery(text=search_text, exhaustive=True, k_nearest_neighbors=50, fields="vector", weight=0.5)
vector_hnsw_query = VectorizableTextQuery(text=search_text, k_nearest_neighbors=50, fields="vector", weight=0.5)
vector_object_query = VectorizedQuery(vector=vector, k_nearest_neighbors=50, fields="vector", weight=1.0)
vector_object_exhaustive_query = VectorizedQuery(vector=vector, exhaustive=True, k_nearest_neighbors=50, fields="vector", weight=1.0)

semantic_configuration = True

# vector query alone
results = dest_search_client.search(
    vector_queries=[vector_object_exhaustive_query],
    select=["id", "description","vector"],
    include_total_count=True,
    top=100,
)
# vector and semantic hybrid
# results = dest_search_client.search(
#     search_text=search_text,
#     vector_queries=[vector_query],
#     query_type=QueryType.SEMANTIC,
#     select=["id", "description","vector"],
#     filter = f"description ne null and search.ismatch('{search_text}', 'description')",
#     semantic_configuration_name="mysemantic",
#     query_caption=QueryCaptionType.EXTRACTIVE,
#     query_answer=QueryAnswerType.EXTRACTIVE,
#     include_total_count=True,
#     top=10,
# )
import time
time.sleep(1)
# query_text = "green street crossing mark for bicycles"
# search_query = odata_filter = search("id: 006312-0003") as per ai agent
# odata_filter = "id eq '008333'"
# odata_filter = "search.in(title, 'aerial', ' ')"
# odata_filter = "search.in(tags, 'urban design')"
# odata_filter = "tags eq 'building,urban design,car,house,land vehicle,vehicle,outdoor,city,aerial,truck'"
# odata_filter = "search.ismatch('urban*', 'tags')"
#only-for-collection-fields odata_filter = "tags/any(g: search.in(g, 'urban', ' '))"

vectors = []
ids = []
# print(repr(results))
if results:
    print(f"Number of results: {results.get_count()}")
    for result in results:
         if result:
            # print(repr(result))
            print(f"{result['id']}")
            ids.append(result['id'])
            # if "vector" in result:
            #     print(f"vector length: {len(result['vector'])}")
            vectors.append(result['vector'])
            # print("\n") 
            # break
else:
    print(f"No Results found")
            
            

import numpy as np

def closest_to_centroid(vectors):
    """
    vectors: list or array of shape (10, 1536)
    returns: (index_of_closest_vector, vector_itself, distance)
    """
    # Convert to numpy array
    X = np.array(vectors)  # shape: (10, 1536)

    # Compute centroid
    centroid = np.mean(X, axis=0)

    # Compute Euclidean distances to centroid
    distances = np.linalg.norm(X - centroid, axis=1)

    # Find index of closest vector
    idx = np.argmin(distances)

    return idx, X[idx], distances[idx]


idx = closest_to_centroid(vectors)
print(idx)
print(f"Closest vector index: {idx[0]}, Distance to centroid: {idx[2]}")
count = 0
print(f"Rewinding results: {results.get_count()}")
for id in ids:
    print(count)
    if count == idx[0].item():
        print(f"Closest to centroid is id: {id}")
        break
    count += 1
print("finished")

# References: from studies in search3.py