#! /usr/bin/python
import json
import sys
import os
sys.path.insert(0, os.path.abspath(".."))
import re
import numpy as np
from PIL import Image
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
vision_api_key = os.getenv("AZURE_AI_VISION_API_KEY")
# import dbscan
object_id = "010011-0006"
object_id = "005071-0006"
scene_id = "010011-0003"
object_uri = f"https://saravinoteblogs.blob.core.windows.net/playground/vision/main/objects/{object_id}.jpg?sp=racwdlme&st=2025-12-21T09:05:18Z&se=2025-12-23T17:20:18Z&spr=https&sv=2024-11-04&sr=c&sig=3mGonRsk%2Bm2CLTuJieOAUqFLYuQIWliZQnOHcJxR56o%3D"
object_uri = f"https://sadronevideo.blob.core.windows.net/input/stock/aerial-view-white-car.jpg?sp=racwdl&st=2025-12-21T19:41:48Z&se=2025-12-24T03:56:48Z&spr=https&sv=2024-11-04&sr=c&sig=meBFxTQ5N9z%2BKKzg2tGWvbt7KFzTxt9DvBPod5OejzU%3D"
object_uri = f"https://sadronevideo.blob.core.windows.net/input/stock/white-car-zooming.jpg?sp=racwdl&st=2025-12-21T19:41:48Z&se=2025-12-24T03:56:48Z&spr=https&sv=2024-11-04&sr=c&sig=meBFxTQ5N9z%2BKKzg2tGWvbt7KFzTxt9DvBPod5OejzU%3D"
object_uri = f"https://sadronevideo.blob.core.windows.net/input/stock/white-car-zooming-rotated.jpg?sp=racwdl&st=2025-12-21T19:41:48Z&se=2025-12-24T03:56:48Z&spr=https&sv=2024-11-04&sr=c&sig=meBFxTQ5N9z%2BKKzg2tGWvbt7KFzTxt9DvBPod5OejzU%3D"
scene_uri = f"https://saravinoteblogs.blob.core.windows.net/playground/vision/main/objects/{scene_id}.jpg?sp=racwdlme&st=2025-12-21T09:05:18Z&se=2025-12-23T17:20:18Z&spr=https&sv=2024-11-04&sr=c&sig=3mGonRsk%2Bm2CLTuJieOAUqFLYuQIWliZQnOHcJxR56o%3D"

src_search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=src_index_name,
    credential=credential
)
# object = src_search_client.get_document(key=object_id) 
# scene = src_search_client.get_document(key=scene_id) 
#print(f"object_description={object.get("description")}")
#print(f"scene_description={scene.get("description")}")
# object_vector = object.get("vector")
# scene_vector = scene.get("vector")
# print(f"object_vector_length={len(object_vector)}")
# print(f"scene_vector_length={len(scene_vector)}")

count = 0
import dbscan
count = dbscan.count_multiple_matches(scene_uri, object_uri)
print(f"Estimated object instances: {count}")

"""
len of labels=0 and labels=[]
Estimated object instances: 0
"""