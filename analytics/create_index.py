import requests
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
import os

project_endpoint = os.environ["AZURE_PROJECT_ENDPOINT"]
project_api_key = os.environ["AZURE_PROJECT_API_KEY"]
agent_model = os.getenv("AZURE_AGENT_MODEL", "gpt-4o-mini")
search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
api_version = os.getenv("AZURE_SEARCH_API_VERSION")
search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
credential = AzureKeyCredential(search_api_key)
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "index00")
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_gpt_deployment = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4o-mini")
azure_openai_gpt_model = os.getenv("AZURE_OPENAI_GPT_MODEL", "gpt-4o-mini")
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
azure_openai_embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
chat_agent_name = os.getenv("AZURE_CHAT_AGENT_NAME", "chat-agent-in-a-team")
search_agent_name = os.getenv("AZURE_SEARCH_AGENT_NAME", "search-agent-in-a-team")
api_version = "2025-05-01-Preview"
agent_max_output_tokens=10000
vectorizer_name = "vectorizer-1748574121417"
semantic_configuration_name = "mysemantic1"
vector_dimension_size=1536
vector_search_profile_name = "myExhaustiveKnnProfile1"
new_index_name = "index05"

from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField, SearchFieldDataType,
    SimpleField, SearchableField, VectorSearch, VectorSearchAlgorithmConfiguration, VectorSearchProfile,
    HnswParameters, ExhaustiveKnnParameters, VectorSearchAlgorithmMetric, HnswAlgorithmConfiguration, ExhaustiveKnnAlgorithmConfiguration,
    AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters, VectorSearchAlgorithmKind,
    # AzureOpenAIParameters, VectorSearchVectorizer,
    SemanticSearch, SemanticConfiguration, SemanticPrioritizedFields, SemanticField, BM25SimilarityAlgorithm
)

# 1. Create the vectorizer
# vectorizer_url = f"{search_endpoint}/vectorizers/{vectorizer_name}?api-version=2023-11-01-preview"
# vectorizer_payload = {
    # "name": vectorizer_name,
    # "kind": "azureOpenAI",
    # "azureOpenAIParameters": {
        # "resourceUri": azure_openai_endpoint,
        # "deploymentId": azure_openai_embedding_deployment,
        # "modelName": azure_openai_embedding_model,
    # }
# }
# vectorizer_response = requests.put(vectorizer_url, json=vectorizer_payload, headers=headers)
# print("Vectorizer:", vectorizer_response.status_code, vectorizer_response.json())

vectorizer_config = AzureOpenAIVectorizer(
    vectorizer_name = vectorizer_name, 
    parameters = AzureOpenAIVectorizerParameters(
        resource_url=azure_openai_endpoint,
        api_key = azure_openai_api_key,
        deployment_name=azure_openai_embedding_deployment,
        model_name=azure_openai_embedding_model
    )
)    
print(f"Vectorizer config '{vectorizer_name}' created for index.")

# 2. Create the vector search with the vectorizer
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            parameters = HnswParameters(metric=VectorSearchAlgorithmMetric.COSINE, m=4, ef_construction=400, ef_search=1000),
            name="myHnsw1",
            kind=VectorSearchAlgorithmKind.HNSW,
        ),
        ExhaustiveKnnAlgorithmConfiguration(
            parameters = ExhaustiveKnnParameters(metric=VectorSearchAlgorithmMetric.COSINE),
            name="myExhaustiveKnn1",
            kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="myHnswProfile1",
            algorithm_configuration_name="myHnsw1",
            vectorizer=vectorizer_name
        ),
        VectorSearchProfile(
            name=vector_search_profile_name,
            algorithm_configuration_name="MyExhaustiveKnn1",
            vectorizer=vectorizer_name
        )
    ],
    vectorizers = [vectorizer_config]
    # vectorizers=None  # Already created if using REST APIs for latest features
)
print(f"VectorSearch with '{vectorizer_name}' created for index.")

semantic_search = SemanticSearch(
    default_configuration_name=semantic_configuration_name,
    configurations=[
        SemanticConfiguration(
            name=semantic_configuration_name,
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="description"),
                prioritized_content_fields=[
                    SemanticField(field_name="id"),
                    SemanticField(field_name="description")
                ],
                prioritized_keywords_fields=[
                    SemanticField(field_name="id"),
                    SemanticField(field_name="description")
                ]
            ),
            ranking_order="BoostedRerankerScore",
            flighting_opt_in=False
        )
    ]
)

similarity_algorithm = BM25SimilarityAlgorithm()
print(f"Semantic configuration '{semantic_configuration_name}' created for index.")
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True, retrievable=True, stored=True),
    SearchableField(name="accountid", type=SearchFieldDataType.String, searchable=True, filterable=True,
                    retrievable=True, stored=True, sortable=True, facetable=True),
    SearchableField(name="description", type=SearchFieldDataType.String, searchable=True, filterable=True,
                    retrievable=True, stored=True, sortable=True, facetable=True),
    SearchField(name="vector", type="SearchFieldDataType.Collection(Edm.Single)", searchable=True, retrievable=True,
                stored=True, vector_search_dimensions=vector_dimension_size, vector_search_profile_name=vector_search_profile_name),
    SearchableField(name="objects", type=SearchFieldDataType.String, analyzer_name="standard.lucene",
                    searchable=True, filterable=True, retrievable=True, stored=True,
                    sortable=True, facetable=True),
    SearchableField(name="tags", type=SearchFieldDataType.String, analyzer_name="standard.lucene",
                    searchable=True, filterable=True, retrievable=True, stored=True,
                    sortable=True, facetable=True),
    SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="standard.lucene",
                    searchable=True, filterable=True, retrievable=True, stored=True,
                    sortable=True, facetable=True),
]

# 3. create the index with fields, vectorizer and semantic configuration
index = SearchIndex(
    name=new_index_name,
    fields=fields,
    semantic_search=semantic_search,
    vector_search=vector_search,
    similarity=similarity_algorithm
)

index_client = SearchIndexClient(
    endpoint=search_endpoint,
    credential=AzureKeyCredential(search_api_key)
)

index_client.create_or_update_index(index)
print(f"Index '{new_index_name}' created with vector and semantic search.")
"""
Output:
Vectorizer config 'vectorizer-1748574121417' created for index.
vectorizer is not a known attribute of class <class 'azure.search.documents.indexes._generated.models._models_py3.VectorSearchProfile'> and will be ignored
vectorizer is not a known attribute of class <class 'azure.search.documents.indexes._generated.models._models_py3.VectorSearchProfile'> and will be ignored
VectorSearch with 'vectorizer-1748574121417' created for index.
prioritized_content_fields is not a known attribute of class <class 'azure.search.documents.indexes._generated.models._models_py3.SemanticPrioritizedFields'> and will be ignored
prioritized_keywords_fields is not a known attribute of class <class 'azure.search.documents.indexes._generated.models._models_py3.SemanticPrioritizedFields'> and will be ignored
Semantic configuration 'mysemantic1' created for index.
Index 'index05' created with vector and semantic search.
"""