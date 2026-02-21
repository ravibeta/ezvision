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

src_image = src_search_client.get_document(key=target_id) 
print(f"id={src_image.get("id")}")
print(f"location={src_image.get("location")}")
print(f"vector={len(src_image.get("vector"))}")
print(f"description = {src_image.get("description")}")
vector = src_image.get("vector")

# # from sentence_transformers import SentenceTransformer, util

# # def get_most_similar_sentence(input_string, sentences):
# #     """
# #     Accepts an input string and a list of sentences.
# #     Computes semantic similarity scores and returns the most similar sentence.
# #     """
# #     # Load a pretrained model for semantic similarity
# #     model = SentenceTransformer('all-MiniLM-L6-v2')
    
# #     # Encode input string and sentences into embeddings
# #     input_embedding = model.encode(input_string, convert_to_tensor=True)
# #     sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    
# #     # Compute cosine similarity scores
# #     similarity_scores = util.cos_sim(input_embedding, sentence_embeddings)[0]
    
# #     # Rank sentences by similarity
# #     ranked_indices = similarity_scores.argsort(descending=True)
    
# #     # Return the sentence with the highest similarity
# #     most_similar_sentence = sentences[ranked_indices[0]]
    
# #     return most_similar_sentence, similarity_scores[ranked_indices[0]].item()

# Example usage
# # if __name__ == "__main__":
# #     input_str = "Artificial intelligence in drones"
# #     sentence_list = [
# #         "Drones are used for aerial photography.",
# #         "AI improves drone navigation and analytics.",
# #         "Cooking recipes can be automated with AI.",
# #         "The weather today is sunny."
# #     ]
    
# #     best_sentence, score = get_most_similar_sentence(input_str, sentence_list)
# #     print("Most similar sentence:", best_sentence)
# #     print("Similarity score:", score)

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
# # # print(f"'{target_id.split('-')[0]}'")
# # # prefix = f"{target_id.split('-')[0]}"
# # # descriptions = []
# # # for i in range(1,11):
# # #     object_id = f"{prefix}-{i:04d}"
# # #     print(object_id)
# # #     try:
# # #         object_image = dest_search_client.get_document(key=object_id) 
# # #         description_text = object_image.get("description")
# # #         try:
# # #             description = json.loads(prepare_json_string_for_load(description_text))
# # #         except Exception as ie:
# # #             print(f"Could not parse description as JSON: {ie}")
# # #             continue
# # #         description = object_image.get("_data").get("description")
# # #         if description:
# # #             descriptions.append(",".join(description))
# # #             print(f"id={object_image.get("id")}, description={description}")
# # #         if "parking" in descriptions:
# # #             print(f"Found parking in description for {object_id}")
# # #         print("No description.")
# # #     except Exception as e:
# # #         print(f"Document not found: {object_id} due to : {e}")
# # #         if 'Not Found' in str(e):
# # #             continue
# # #         break

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
search_text = "building with cars parked inside the building and on the top-level roof"
search_text = "parking garage"
# search_text = "parking garage with cars parked inside the building and on the top-level roof"
vector_query = VectorizableTextQuery(text=search_text, exhaustive=True, k_nearest_neighbors=50, fields="vector", weight=0.5)

vector_hnsw_query = VectorizableTextQuery(text=search_text, k_nearest_neighbors=50, fields="vector", weight=0.5)
vector_object_query = VectorizedQuery(vector=vector, k_nearest_neighbors=50, fields="vector", weight=1.0)
vector_object_exhaustive_query = VectorizedQuery(vector=vector, exhaustive=True, k_nearest_neighbors=50, fields="vector", weight=1.0)

# Set up the search results and the chat thread.
# Retrieve the selected fields from the search index related to the question.
# Search results are limited to the top 5 matches. Limiting top can help you stay under LLM quotas.
semantic_configuration = True
# results = dest_search_client.search(
#     search_text=search_text, 
#     query_type="semantic" if semantic_configuration else "simple",
#     select = ["id", "description"],
#     filter = f"description ne null and search.ismatch('parking garage', 'description')",
#     semantic_configuration_name='mysemantic' if semantic_configuration else None,
#     ## order_by="cast(search.score,Edm.String) desc",
#     include_total_count=True,
#     top=10,
# )

# vector query alone
results = dest_search_client.search(
    # search_text=search_text,
    vector_queries=[vector_object_exhaustive_query],
    # query_type=QueryType.SEMANTIC,
    select=["id", "description","vector"],
    # filter = f"description ne null and search.ismatch('{search_text}', 'description')",
    # semantic_configuration_name="mysemantic",
    # query_caption=QueryCaptionType.EXTRACTIVE,
    # query_answer=QueryAnswerType.EXTRACTIVE,
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
#     top=10,
# )
# results = dest_search_client.search(
#     # search_text=search_text, 
#     # query_type="semantic" if semantic_configuration else "simple",
#     vector_queries= [vector_query],
#     select = ["id", "description"],
#     ## select=", ".join(["id", "description"]),
#     ## filter = f"description ne null and search.ismatch('parking garage', 'description')",
#     # semantic_configuration_name='mysemantic' if semantic_configuration else None,
#     ## order_by="cast(search.score,Edm.String) desc",
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
# # and effect of alternate jargon
# results =  search_client.search(
#     # query_type='simple',
#     # search_text=query_text,
#     select='id,description',
#     filter=odata_filter,
#     include_total_count=True,
#     top=10)
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


### The following is for object search just like in playground:
# from azure.ai.projects import AIProjectClient
# from azure.identity import DefaultAzureCredential
# from azure.ai.agents.models import ListSortOrder

# project = AIProjectClient(
#     credential=DefaultAzureCredential(),
#     endpoint="https://found-vision-1.services.ai.azure.com/api/projects/droneimage")

# agent = project.agents.get_agent("asst_JI9VWjdav3To7jjGUROejGkV")

# thread = project.agents.threads.create()
# print(f"Created thread, ID: {thread.id}")

# message = project.agents.messages.create(
#     thread_id=thread.id,
#     role="user",
#     content="Hi object-search-agent"
# )

# run = project.agents.runs.create_and_process(
#     thread_id=thread.id,
#     agent_id=agent.id)

# if run.status == "failed":
#     print(f"Run failed: {run.last_error}")
# else:
#     messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)

#     for message in messages:
#         if message.text_messages:
#             print(f"{message.role}: {message.text_messages[-1].text.value}")