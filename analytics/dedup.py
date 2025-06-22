# import numpy as np

# def cosine_similarity(vec1, vec2):
    # """Computes cosine similarity between two vectors."""
    # dot_product = np.dot(vec1, vec2)
    # norm_vec1 = np.linalg.norm(vec1)
    # norm_vec2 = np.linalg.norm(vec2)
    # return dot_product / (norm_vec1 * norm_vec2)

# def euclidean_distance(vec1, vec2):
    # """Computes Euclidean distance between two vectors."""
    # return np.linalg.norm(np.array(vec1) - np.array(vec2))

# Example query vector
# query_vector = np.random.rand(1536)  # Replace with actual vector

# Example retrieved vector (from Azure AI Search)
# retrieved_vector = np.random.rand(1536)  # Replace with actual vector from search results

# Compute distances
# cosine_sim = cosine_similarity(query_vector, retrieved_vector)
# euclidean_dist = euclidean_distance(query_vector, retrieved_vector)

# print(f"Cosine Similarity: {cosine_sim}")
# print(f"Euclidean Distance: {euclidean_dist}")

# from PIL import Image
# import imagehash

# def perceptual_hash(image_path):
    # img = Image.open(image_path)
    # return imagehash.phash(img)

# print(perceptual_hash("image.jpg"))

import cv2
import imagehash
import numpy as np
from PIL import Image
from collections import deque
from azure.search.documents.models import (
    VectorizedQuery,
    VectorizableTextQuery
)
class ImageDeduplicator:
    def __init__(self, buffer_size=100):
        """Initialize a ring buffer for tracking image hashes."""
        self.buffer_size = buffer_size
        self.hash_buffer = deque(maxlen=buffer_size)
        self.vector_buffer = deque(maxlen=buffer_size)

    def compute_hash(self, image):
        """Compute perceptual hash of an image."""
        return imagehash.phash(Image.fromarray(image))

    def is_duplicate(self, image):
        """Check if the image is a duplicate."""
        img_hash = self.compute_hash(image)
        if img_hash in self.hash_buffer:
            return True
        self.hash_buffer.append(img_hash)
        return False
    
    def is_visited(self, vector):
        index = 0
        for existing in reversed(self.vector_buffer):
            # print(existing)
            score = self.cosine_similarity(existing, vector)
            if score > 0.97:
                # print(f"found a match at {index} out of {len(self.vector_buffer)} with score {score}")
                return True
            index += 1    
        self.vector_buffer.append(vector)
        return False
        
    def is_existing(self, external_vector_client, vector):
        vector_query = VectorizedQuery(vector=vector,
                                  k_nearest_neighbors=3,
                                  exhaustive=False,
                                  fields = "vector") 
        results = external_vector_client.search(  
        search_text=None,  
        vector_queries= [vector_query],
        select=["id", "description","vector"],  
        # select='id,description,vector',
        include_total_count=True,
        top=4
        )
        if results != None and results.get_count() > 0:
            best = 0
            id = None
            for match in results:
                # print(f"{match['id']} found." + ",".join([key for key in match.keys()]))
                match_vector = match["vector"]
                score = self.cosine_similarity(vector, match_vector)
                # print(f"score={score}")
                if score > best:
                    id = match['id']
                    best = score
                else:
                    continue
            matches = ','.join([match['id'] for match in results]).strip(',')
            print(f"matches: {matches}")
            if best > 0.8:
               print(f"match found with score {best} for {id}.")
               return True
        else:
            print("no match found.")
        return False
        
    def get_hash_buffer_len(self):
        return len(self.hash_buffer)

    def get_vector_buffer_len(self):
        return len(self.vector_buffer) 

    def cosine_similarity(self, vec1, vec2):
        """Computes cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def euclidean_distance(self, vec1, vec2):
        """Computes Euclidean distance between two vectors."""
        value = np.linalg.norm(np.array(vec1) - np.array(vec2))   
        print(f"Euclidean={value}")
        return value        

# def process_image_stream(image_stream):
    # """Process a stream of images and eliminate duplicates."""
    # deduplicator = ImageDeduplicator()
    # unique_images = []

    # for image in image_stream:
        # if not deduplicator.is_duplicate(image):
            # unique_images.append(image)

    # return unique_images

# # Example usage
# image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]  # Replace with actual image paths
# image_stream = [cv2.imread(img) for img in image_paths]

# unique_images = process_image_stream(image_stream)

# print(f"Unique images count: {len(unique_images)}")
