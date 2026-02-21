import os
import sys
import requests
import numpy as np
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from openai import AzureOpenAI
sys.path.insert(0, os.path.abspath(".."))
from visionprocessor.vectorizer import vectorize_image, analyze_image


# ENVIRONMENT VARIABLES
vision_endpoint=os.getenv("AZURE_AI_VISION_ENDPOINT")
vision_api_key = os.getenv("AZURE_AI_VISION_API_KEY")
search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")  
search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
image_uri = os.getenv("AZURE_RED_CAR_SAS_URL").strip('"')
image_dataset_uri = os.getenv("AZURE_QUERY_SAS_URI").strip('"')
embedding_api = os.getenv("AZURE_EMBEDDING_ENDPOINT")
embedding_api_key = os.getenv("AZURE_EMBEDDING_API_KEY")
chat_api = os.getenv("AZURE_CHAT_ENDPOINT")
chat_api_key = os.getenv("AZURE_CHAT_API_KEY")
model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"

# STEP 1: Embed the query using Florence or CLIP
def embed_query(text):
    # Replace with your embedding endpoint or model
    headers = {"api-key": embedding_api_key, "Content-Type": "application/json"}
    response = requests.post(embedding_api, json={"input": text}, headers=headers)
    return response.json()["data"][0]["embedding"]

# STEP 2: Search Azure AI Search with the query vector
def search_similar_images(vector):
    client = SearchClient(endpoint=search_endpoint,
                          index_name=search_index_name,
                          credential=AzureKeyCredential(search_api_key))
    results = client.search(
        search_text=None,
        vector=vector,
        vector_fields="vector",
        top=20,
        vector_search_mode="nearest"
    )
    return [doc["imageUrl"] for doc in results]

# STEP 3: Analyze images for red cars
def count_red_cars(image_urls):
    client = ImageAnalysisClient(endpoint=vision_endpoint,
                                 credential=AzureKeyCredential(vision_api_key))
    count = 0
    for url in image_urls:
        result = client.analyze_from_url(
            image_url=url,
            visual_features=[VisualFeatures.OBJECTS]
        )
        for obj in result.objects:
            if obj.name.lower() == "car" and "red" in obj.tags:
                count += 1
    return count
    
def ask_chat(image_vector, dataset_vector, description, query):
    client = AzureOpenAI(
      azure_endpoint = chat_api, 
      api_key=chat_api_key,  
      api_version="2023-05-15"
    )
    data = [{"object_vector": image_vector.tolist(), "image_vector": dataset_vector.tolist()}]
    import json
    content = json.dumps(data)
    conversation=[{"role": "system", "content": "You are an AI assistant that answers questions about objects detected in an image when both are made available to you as vectors. You must find the number of instances of the given object in the image."}]
    conversation.append({"role": "assistant", "content": content})
    conversation.append({"role": "user", "content": query})
    response = client.chat.completions.create(
    model=deployment,
    messages=conversation,
    temperature=0,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop = [' END']
    )
    return (response.choices[0].message.content).strip()
    
# MAIN
query = "How many red trucks are there in the given image?"
# query_vector = embed_query(query)
#query_vector = np.pad(query_vector, (0, 1536 - len(query_vector)), mode='constant')
image_vector = vectorize_image(image_uri, vision_api_key, "eastus")
print(f"image_vector_len={len(image_vector)}")
image_vector = np.pad(image_vector, (0, 1536 - len(image_vector)), mode='constant')
dataset_vector = vectorize_image(image_dataset_uri, vision_api_key, "eastus")
dataset_vector = np.pad(dataset_vector, (0, 1536 - len(dataset_vector)), mode='constant')
print(f"dataset_vector_len={len(dataset_vector)}")
analysis_client = ImageAnalysisClient(vision_endpoint, AzureKeyCredential(vision_api_key))
description = analyze_image(analysis_client, image_dataset_uri)
print(description)
# image_urls = search_similar_images(query_vector) 
red_car_count = ask_chat(image_vector, dataset_vector, description, query)
print(f"Result: {red_car_count}")
