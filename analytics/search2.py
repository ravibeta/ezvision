#! /usr/bin/python
import json
import sys
import os
import requests
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import (
#    RawVectorQuery,
    VectorizableTextQuery
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
credential = AzureKeyCredential(search_api_key)

search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)
# search_query = odata_filter = search("id: 006312-0003") as per ai agent
query_text = "green street crossing mark for bicycles"
odata_filter = "id eq '008333'"
odata_filter = "search.in(title, 'aerial', ' ')"
odata_filter = "search.in(tags, 'urban design')"
odata_filter = "tags eq 'building,urban design,car,house,land vehicle,vehicle,outdoor,city,aerial,truck'"
odata_filter = "search.ismatch('urban*', 'tags')"
#only-for-collection-fields odata_filter = "tags/any(g: search.in(g, 'urban', ' '))"
"""
# Each of the search cases commented out is valid and successful but we will keep it simple
# start with a vector search
blob_url = "https://saravinoteblogs.blob.core.windows.net/playground/vision/query/RedCar4.jpg?sp=racwdle&st=2025-05-26T23:54:09Z&se=2025-05-27T07:54:09Z&spr=https&sv=2024-11-04&sr=d&sig=9RRmmtlBnEiFsOGHJ2d%2ByEkBz2gxXOrQEc%2B4uf%2Fd6ao%3D&sdd=2"
vector = vectorize_image(blob_url, vision_api_key, "eastus")
print(f"len={len(vector)}")
print("search_client created")

vector_query = RawVectorQuery(vector=vector,
                              k=3,
                              fields = "image_vector")  

results = search_client.search(  
    search_text=None,  
    vector_queries= [vector_query],
    select=["id", "description"]  
)   
# and simple text multimodal
results =  search_client.search(query_type='simple',
    search_text="green crosswalk for bicycles at street intersection" ,
    select='id,description',
    include_total_count=True,
    top=10)
"""    
# and effect of alternate jargon
results =  search_client.search(
    # query_type='simple',
    # search_text=query_text,
    select='id,description',
    filter=odata_filter,
    include_total_count=True,
    top=10)
"""    
# and semantic search
results =  search_client.search(query_type='semantic', semantic_configuration_name='my-semantic-config',
    search_text="green crossing for bicycles at street intersection", 
    select='id,description', query_caption='extractive')

# and vectorizable text query
query="Do bicycles have a dedicated crossing at street intersections?"
vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="image_vector")

# Set up the search results and the chat thread.
# Retrieve the selected fields from the search index related to the question.
# Search results are limited to the top 5 matches. Limiting top can help you stay under LLM quotas.
results = search_client.search(
    search_text=query,
    vector_queries= [vector_query],
    select=["id", "description"],
    include_total_count=True,
    top=5,
)
# returns Message: Field 'image_vector' does not have a vectorizer defined in it's vector profile.
"""
print(repr(results))
if results:
    print(f"Number of results: {results.get_count()}")
    for result in results:
         if result:
            # print(repr(result))
            print(f"{result['id']}")
            # print("\n") 
            # break
            
            
"""
<iterator object azure.core.paging.ItemPaged at 0x1df4252ee40>
Number of results: 1
015969
<iterator object azure.core.paging.ItemPaged at 0x227467dee40>
Number of results: 7893
000688
000735
000720
000766
000741
000771
000775
000820
000824
000791
"""