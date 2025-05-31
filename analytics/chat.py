# Import libraries
import json
import os
import requests
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.identity import get_bearer_token_provider
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import (
    RawVectorQuery,
    VectorizableTextQuery
)
from openai import AzureOpenAI
search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  
index_name = os.getenv("AZURE_SEARCH_NEW_INDEX_NAME")
api_version = os.getenv("AZURE_SEARCH_API_VERSION")
search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
vision_api_key = os.getenv("AZURE_AI_VISION_API_KEY")
vision_api_version = os.getenv("AZURE_AI_VISION_API_VERSION")
vision_region = os.getenv("AZURE_AI_VISION_REGION")
vision_endpoint =  os.getenv("AZURE_AI_VISION_ENDPOINT")
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
credential = AzureKeyCredential(openai_api_key)

# Set up the Azure OpenAI client
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://search.azure.com/.default")
openai_client = AzureOpenAI(
     api_version="2024-06-01",
     azure_endpoint=openai_endpoint,
     azure_ad_token_provider=token_provider
 )

deployment_name = "text-embedding-ada-002"

# Set up the Azure Azure AI Search client
search_client = SearchClient(
     endpoint=search_endpoint,
     index_name=index_name,
     credential=DefaultAzureCredential() # AzureKeyCredential(search_api_key)
 )

# Provide instructions to the model
GROUNDED_PROMPT="""
You are an AI assistant that helps users learn from the images found in the source material.
Answer the query using only the sources provided below.
Use bullets if the answer has multiple points.
If the answer is longer than 3 sentences, provide a summary.
Answer ONLY with the facts listed in the list of sources below. Cite your source when you answer the question
If there isn't enough information below, say you don't know.
Do not generate answers that don't use the sources below.
Query: {query}
Sources:\n{sources}
"""
# Provide the search query. 
# It's hybrid: a keyword search on "query", with text-to-vector conversion for "vector_query".
# The vector query finds 50 nearest neighbor matches in the search index
query="Do bicycles have a dedicated crossing at street intersections?"
vector_query = VectorizableTextQuery(text=query, fields="vector", Exhaustive=True, KNearestNeighborsCount=50)
vector_query.Text = query
vector_query.fields = "vector"
vector_query.Exhaustive = True
vector_query.KNearestNeighborsCount = 50

# Set up the search results and the chat thread.
# Retrieve the selected fields from the search index related to the question.
# Search results are limited to the top 5 matches. Limiting top can help you stay under LLM quotas.
results = search_client.search(
    search_text=query,
    vector_queries= [vector_query],
    select=["id", "description"],
    include_total_count = True,
    semantic_configuration_name = "mysemantic",
    top=5
)
references = []
if results != None:
    # print(f"Number of results: {results.get_count()}")
    references = [f'ID: {document["id"]}' for document in results]
    print("Number of references: {len(references)}")
# Use a unique separator to make the sources distinct. 
# We chose repeated equal signs (=) followed by a newline because it's unlikely the source documents contain this sequence.
sources_formatted = "=================\n".join(references)
response = openai_client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": GROUNDED_PROMPT.format(query=query, sources=sources_formatted)
        }
    ],
    model=deployment_name
)

print(response.choices[0].message.content)