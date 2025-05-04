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

# Create a search index 
index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)  
fields = [  
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),  
    SearchField(name="description", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),  
    SearchField(
        name="image_vector",  
        hidden=True,
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
        searchable=True,
        vector_search_dimensions=1024,  
        vector_search_profile="myHnswProfile"
    ),  
]  
  
# Configure the vector search configuration  
vector_search = VectorSearch(  
    algorithms=[  
        HnswVectorSearchAlgorithmConfiguration(  
            name="myHnsw",  
            kind=VectorSearchAlgorithmKind.HNSW,  
            parameters=HnswParameters(  
                m=4,  
                ef_construction=400,  
                ef_search=1000,  
                metric="cosine",  
            ),  
        ),  
            ExhaustiveKnnVectorSearchAlgorithmConfiguration(  
            name="myExhaustiveKnn",  
            kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,  
            parameters=ExhaustiveKnnParameters(  
                metric="cosine",  
            ),  
        ), 
    ],  
   profiles=[  
        VectorSearchProfile(  
            name="myHnswProfile",  
            algorithm="myHnsw",  
        ),  
        VectorSearchProfile(  
            name="myExhaustiveKnnProfile",  
            algorithm="myExhaustiveKnn",  
        ),  
    ],  
)  
  
# Create the search index with the vector search configuration  
index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)  
result = index_client.create_or_update_index(index)  
print(f"{result.name} created") 