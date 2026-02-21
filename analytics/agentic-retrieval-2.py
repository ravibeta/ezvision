#!/usr/bin/python
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.agents import AgentsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.projects import AIProjectClient
import os
load_dotenv(override=True)

project_endpoint = os.environ["AZURE_PROJECT_ENDPOINT"]
project_api_key = os.environ["AZURE_PROJECT_API_KEY"]
agent_model = os.getenv("AZURE_AGENT_MODEL", "gpt-4o-mini")
search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
api_version = os.getenv("AZURE_SEARCH_API_VERSION")
search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
credential = AzureKeyCredential(search_api_key)
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://search.azure.com/.default")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "index00")
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_gpt_deployment = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4o-mini")
azure_openai_gpt_model = os.getenv("AZURE_OPENAI_GPT_MODEL", "gpt-4o-mini")
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
azure_openai_embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
chat_agent_name = os.getenv("AZURE_CHAT_AGENT_NAME", "chat-agent-in-a-team")
search_agent_name = os.getenv("AZURE_SEARCH_AGENT_NAME", "search-agent-in-a-team")
print(f"Agent Name={search_agent_name}")
print(f"Index Name={index_name}")
api_version = "2025-05-01-Preview"
agent_max_output_tokens=10000

from azure.search.documents.indexes.models import KnowledgeAgent, KnowledgeAgentAzureOpenAIModel, KnowledgeAgentTargetIndex, KnowledgeAgentRequestLimits, AzureOpenAIVectorizerParameters
from azure.search.documents.indexes import SearchIndexClient
index_client = SearchIndexClient(endpoint=search_endpoint, credential=AzureKeyCredential(search_api_key))  
index_client.close()
index_client = SearchIndexClient(endpoint=search_endpoint, credential=AzureKeyCredential(search_api_key))  

for index_agent in index_client.list_agents():
    print(index_agent.name)
    # index_agent.clear()
agent = index_client.get_agent(search_agent_name)
if  agent is None:
    print("no agent found")
    model =  KnowledgeAgentAzureOpenAIModel( 
            azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                resource_url=azure_openai_endpoint, 
                deployment_name=azure_openai_gpt_deployment, 
                model_name=azure_openai_gpt_model,
                api_key = azure_openai_api_key                 
            ) 
        )
    agent = KnowledgeAgent( 
        name=search_agent_name, 
        models=[ 
            model
        ], 
        target_indexes=[ 
            KnowledgeAgentTargetIndex( 
                index_name=index_name, 
                default_include_reference_source_data=True, 
                default_reranker_threshold=2.5 
            ) 
        ], 
        request_limits=KnowledgeAgentRequestLimits( 
            max_output_size=10000 
        ) 
    )
    r = index_client.create_or_update_agent(agent=agent) 
    print(f"AI Knowledge agent '{search_agent_name}' created or updated successfully: {r}")     
print(agent) 

instructions = ["""
You are an AI assistant that answers questions about the stored and indexed drone images and objects in search index index02.
The data source is an Azure AI Search resource where the schema has JSON description field, a vector field and an id field and this id field must be cited in your answer.
If you do not find a match for the query, respond with "I don't know", otherwise cite references with the value of the id field.
"""]
messages = [
    {
        "role":"system",
        "content": instructions
    }
]


from azure.ai.agents.models import FunctionTool, ToolSet, ListSortOrder
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import KnowledgeAgentRetrievalRequest, KnowledgeAgentMessage, KnowledgeAgentMessageTextContent, KnowledgeAgentIndexParams

agent_client = KnowledgeAgentRetrievalClient(endpoint=search_endpoint, agent_name=search_agent_name, credential=AzureKeyCredential(search_api_key))
# query_text = "How many parking lots are empty when compared to all the parking lots?" 
query_text = "How many red cars can be found near the building with a roof that has a circular structure?"
query_text = "How many parking can be found near the building with a roof that has a circular structure?"
query_text = "How far is a split in the road from the building with a roof that has a circular structure?"
query_text = "Is there a green car?"
query_text = "How far is a skyscraper from the lake?"
query_text = "Find the image with the tallest building among all."
query_text = "Which is the tallest building among all the buildings in images?"
query_text = "List all types of vehicles found on the roads or parking lots from the images"
query_text = "How many red cars can be found near the building with a roof that has a circular structure?"
query_text = "How many different colors of cars can be found on the roads near the building with a roof that has a circular structure?"
query_text = "How many storeys are there in buildings close to parking lots?"
query_text = "Is there a parking lot building with cars on the roof?"
query_text = "Are there triangular buildings for parking?"
messages.append({
    "role": "user",
    "content": query_text
})
# print(agent_client)
print([msg["content"] for msg in messages if msg["role"] != "system"])
retrieval_result = agent_client.retrieve(
    retrieval_request=KnowledgeAgentRetrievalRequest(
        messages=[KnowledgeAgentMessage(role=msg["role"], content=[KnowledgeAgentMessageTextContent(text=msg["content"])]) for msg in messages if msg["role"] != "system"],
        target_index_params=[KnowledgeAgentIndexParams(index_name=index_name, reranker_threshold=2.5, include_reference_source_data=True)] # add filter_add_on here
    )
)
# print(f"Response={retrieval_result.response[0].content[0]}")
print(f"Result={retrieval_result.response[0].content[0].text}")
"""
response = retrieval_result.response[0].content[0].text
print(response)
messages.append({
    "role": "assistant",
    "content": response
})
print("References List:")
print([r.as_dict() for r in retrieval_result.references])
"""
"""
Agent Name=search-agent-in-a-team
Index Name=index00
{'additional_properties': {}, 'name': 'search-agent-in-a-team', 'models': [<azure.search.documents.indexes._generated.models._models_py3.KnowledgeAgentAzureOpenAIModel object at 0x000001AF20311160>], 'target_indexes': [<azure.search.documents.indexes._generated.models._models_py3.KnowledgeAgentTargetIndex object at 0x000001AF20311A90>], 'request_limits': <azure.search.documents.indexes._generated.models._models_py3.KnowledgeAgentRequestLimits object at 0x000001AF203117F0>, 'e_tag': None, 'encryption_key': None, 'description': None}
search-agent-in-a-team:[<azure.search.documents.indexes._generated.models._models_py3.KnowledgeAgentAzureOpenAIModel object at 0x000001AF21517750>]
<KnowledgeAgentRetrievalClient [endpoint='https://srch-vision-01.search.windows.net', agent='search-agent-in-a-team']>
['How many red cars can be found?']
[]
References List:
[]
"""