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
target_id = "010011-0003" # parking garage curved
match_count = 10

# Initialize SearchClient
src_search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=src_index_name,
    credential=credential
)

src_image = src_search_client.get_document(key=target_id) 
print(f"id={src_image.get("id")}")
# print(f"location={src_image.get("location")}")
# print(f"vector={len(src_image.get("vector"))}")
# print(f"description = {src_image.get("description")}")
vector = src_image.get("vector")

def prepare_json_string_for_load(text):
  text = text.replace("\"", "'")
  text = text.replace("{'", "{\"")
  text = text.replace("'}", "\"}")
  text = text.replace(" '", " \"")
  text = text.replace("' ", "\" ")
  text = text.replace(":'", ":\"")
  text = text.replace("':", "\":")
  text = text.replace(",'", ",\"")
  text = text.replace("',", "\",")
  return re.sub(r'\n\s*', '', text)

dest_search_client = SearchClient(endpoint=search_endpoint, index_name=dest_index_name, credential=credential)
# # vector_query = VectorizedQuery(vector=src_image.get("vector"),kind="vector", k_nearest_neighbors=match_count, fields="vector", exhaustive=True)  #weight=0.5  
# # results = dest_search_client.search(  
# #     search_text=None,  
# #     vector_queries= [vector_query],
# #     select=["id"],
# #     include_total_count=True,
# #     top=10  
# # ) 

# # # print(descriptions)
# filter = f"id ne null and startswith(cast(id,Edm.String), '{prefix}')",
# filter = f"id ne null and id ge '{prefix}-0001' and id lt '{prefix}-0010'",
# Invalid expression: Unsupported function call: contains. Note that you can achieve similar functionality by using the search.ismatch() function. See the documentation for Azure Search filters for more details: https://aka.ms/azsearchodataexpr
# # results = dest_search_client.search(
# #     search_text=None,
# #     filter = f"description ne null and search.ismatch('parking garage', 'description')",
# #     # filter = f"id ne null and search.ismatch('{prefix}', 'id')",
# #     select = ['id','description'],
# #     include_total_count = True,
# #     top = 10
# # )
search_text = "parking garage"
semantic_configuration = True
results = dest_search_client.search(
    search_text=search_text, 
    query_type="semantic" if semantic_configuration else "simple",
    select = ["id", "description"],
    filter = f"description ne null and search.ismatch('parking garage', 'description')",
    semantic_configuration_name='mysemantic' if semantic_configuration else None,
    ## order_by="cast(search.score,Edm.String) desc",
    include_total_count=True,
    top=10,
)
import time
time.sleep(1)
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