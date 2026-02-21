#! /usr/bin/python
import json
import sys
import os
sys.path.insert(0, os.path.abspath(".."))
import re
import numpy as np
from PIL import Image, ImageDraw
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
object_id = "001790-0008"
scene_id = "010011-0003"
scene_ids = [ # "010011-0003",
"000301-0002",
"000313-0004",
"001164-0004",
"001861-0007",
"001866-0004",
"001873-0010",
"002150-0005",
"002789-0004",
"003101-0002",
"003112-0006",
"004399-0004",
"005062-0008",
"005076-0007",
"005344-0002",
"005352-0004",
"005362-0005",
"006303-0002",
"006312-0003",
"007458-0004",
"007463-0004",
"007488-0002",
"007787-0003",
"007795-0003",
"008740-0006",
"008758-0002",
"008769-0002",
"009036-0002",
"009045-0002",
"009055-0004",
"009660-0005",
"009671-0003",
"009677-0003",
"009696-0002",
"010000-0002",
"010017-0004",
"010020-0004",
"010703-0003",
"010717-0002",
"011023-0002",
"011041-0004",
"012168-0005",
"012657-0010",
"012777-0007",
"013388-0009",
"013390-0004",
"013659-0007",
"014750-0009",
"014958-0002",
"014961-0002",
]

src_search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=src_index_name,
    credential=credential
)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed_image(clipped_image, count=0):
    """
    Replace this with your actual embedding call.
    Must return a 1536-dim vector.
    """
    destination_sas_url=f"https://sadronevideo.blob.core.windows.net/output/frame-{count:04d}.jpg?sp=racwdl&st=2025-12-21T07:44:33Z&se=2025-12-23T15:59:33Z&spr=https&sv=2024-11-04&sr=c&sig=2YBieQS2eckGUw%2FL0H1Cqe3UbxU0LjPDl9JM0l5CGIg%3D"
    upload_image_to_blob(clipped_image, destination_sas_url)
    from visionprocessor.vectorizer import vectorize_image
    vector = vectorize_image(destination_sas_url, vision_api_key, "eastus")
    vector = np.pad(vector, (0, 1536 - len(vector)), mode='constant')
    return vector
    # raise NotImplementedError("Connect to your embedding model here.")

def download_and_read_image(image_url):
    from azure.storage.blob import BlobClient
    image_blob_client = BlobClient.from_blob_url(image_url)
    download_stream = image_blob_client.download_blob()
    import io
    image_stream = io.BytesIO(download_stream.readall())
    image_bytes = image_stream.getvalue()
    image_path = f"scene.jpg"
    with open(image_path, 'wb') as f:
        f.write(image_bytes)
    image = Image.open(image_path).convert("RGB")
    return image

def clip_image(image, x, y, width, height):
    clipped_image = image[y:y+height, x:x+width]
    return clipped_image

def upload_image_to_blob(clipped_image, sas_url):
    import cv2
    import numpy as np
    from io import BytesIO
    from azure.storage.blob import BlobClient

    """Uploads the clipped image to Azure Blob Storage using its SAS URL."""
    # _, encoded_image = cv2.imencode(".jpg", clipped_image)
    blob_client = BlobClient.from_blob_url(sas_url)
    # blob_client.upload_blob(encoded_image.tobytes(), overwrite=True)
    blob_client.upload_blob(clipped_image, overwrite=True)

def non_max_suppression(boxes, scores, iou_threshold=0.3):
    """
    Standard NMS to avoid double-counting overlapping detections.
    boxes: list of (x, y, w, h)
    scores: list of similarity scores
    """
    if not boxes:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def find_object_occurrences(object, scene, scene_uri,
                            similarity_threshold=0.90,
                            step=30):
    """
    Sliding-window vector similarity search.
    """

    # 1. Fetch object entry
    obj = object
    obj_vec = obj["vector"]
    # 'width': 556, 'height': 702
    obj_w = 58
    obj_h = 87

    # 2. Load scene image
    download_and_read_image(scene_uri)
    scene_img = None
    with Image.open("scene.jpg") as im:
        W, H = im.size
        scene_img = im.convert("RGB")
    # scene_img = Image.open("scene.jpg").convert("RGB")
    # W = 556
    # H = 702

    detections = []
    scores = []
    frame = 0
    # 3. Slide window across scene
    for y in range(0, H - obj_h, step):
        for x in range(0, W - obj_w, step):

            crop = scene_img.crop((x, y, x + obj_w, y + obj_h))
            import cv2
            import io
            img_byte_arr = io.BytesIO()
            # Save the image to the buffer in the specified format
            crop.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()
            local_only = True
            if local_only:
                image_path = f"frame-{frame:04d}.jpg"
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
            crop_vec = embed_image(image_bytes, frame)
            frame += 1

            sim = cosine_similarity(obj_vec, crop_vec)

            if sim >= similarity_threshold:
                detections.append((x, y, obj_w, obj_h))
                scores.append(sim)

    print(f"detections={len(detections)} before NMS and scores={len(scores)}")
    # 4. Apply NMS to avoid duplicates
    keep_indices = non_max_suppression(detections, scores)
    final_detections = [detections[i] for i in keep_indices]

    # 5. Limit to max 50 occurrences
    bboxes = final_detections[:50]
    print(f"final_detections={len(bboxes)} after NMS")
    img = Image.open("scene.jpg")
    draw = ImageDraw.Draw(img)

    W, H = img.size  # image width and height

    for (x, y, w, h) in bboxes:
        # Convert bottom-left origin → top-left origin
        top_left_y = H - (y + h)

        # Compute rectangle corners
        x1 = x
        y1 = top_left_y
        x2 = x + w
        y2 = top_left_y + h

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    # Save or show the result
    img.save(f"scene_{scene_id}_30.jpg")
    return bboxes

for scene_id in scene_ids:
    # print(f"Processing scene_id={scene_id}")
    object_uri = f"https://saravinoteblogs.blob.core.windows.net/playground/vision/main/objects/{object_id}.jpg?sp=racwdlme&st=2025-12-21T09:05:18Z&se=2025-12-23T17:20:18Z&spr=https&sv=2024-11-04&sr=c&sig=3mGonRsk%2Bm2CLTuJieOAUqFLYuQIWliZQnOHcJxR56o%3D"
    scene_uri = f"https://saravinoteblogs.blob.core.windows.net/playground/vision/main/objects/{scene_id}.jpg?sp=racwdlme&st=2025-12-21T09:05:18Z&se=2025-12-23T17:20:18Z&spr=https&sv=2024-11-04&sr=c&sig=3mGonRsk%2Bm2CLTuJieOAUqFLYuQIWliZQnOHcJxR56o%3D"

    object = src_search_client.get_document(key=object_id) 
    scene = src_search_client.get_document(key=scene_id) 
    #print(f"object_description={object.get("description")}")
    #print(f"scene_description={scene.get("description")}")
    object_vector = object.get("vector")
    scene_vector = scene.get("vector")
    #print(f"object_vector_length={len(object_vector)}")
    #print(f"scene_vector_length={len(scene_vector)}")
    bboxes = []
    bboxes = find_object_occurrences(object, scene, scene_uri)
    # count = dbscan.count_multiple_matches(scene_uri, object_uri)
    print(f"Estimated object instances in {scene_id}: {bboxes}")
    # break

"""
object_vector_length=1536
scene_vector_length=1536
detections=43 before NMS and scores=43
final_detections=27 after NMS
Estimated object instances: [(0, 600, 58, 87), (150, 30, 58, 87), (240, 120, 58, 87), (480, 300, 58, 87), (360, 270, 58, 87), (360, 360, 58, 87), (180, 240, 58, 87), (120, 240, 58, 87), (390, 600, 58, 87), (120, 570, 58, 87), (240, 600, 58, 87), (270, 540, 58, 87), (180, 0, 58, 87), (90, 480, 58, 87), (0, 30, 58, 87), (210, 570, 58, 87), (420, 330, 58, 87), (90, 270, 58, 87), (480, 240, 58, 87), (0, 450, 58, 87), (480, 420, 58, 87), (150, 90, 58, 87), (0, 90, 58, 87), (360, 420, 58, 87), (0, 510, 58, 87), (210, 180, 58, 87), (210, 90, 58, 87)]
"""