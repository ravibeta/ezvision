from azure.search.documents.indexes import SearchIndexClient 

from azure.search.documents.indexes.models import ( 
    KnowledgeAgent, 
    KnowledgeAgentAzureOpenAIModel, 
    KnowledgeAgentRequestLimits, 
    KnowledgeAgentTargetIndex 
)

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import os

load_dotenv(override=True)

project_endpoint = os.environ["PROJECT_ENDPOINT"]
agent_model = os.getenv("AGENT_MODEL", "gpt-4o-mini")
search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
api_version = os.getenv("AZURE_SEARCH_API_VERSION")
search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential, "https://search.azure.com/.default")
index_name = os.getenv("AZURE_SEARCH_NEW_INDEX_NAME", "index01")
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_gpt_deployment = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4o-mini")
azure_openai_gpt_model = os.getenv("AZURE_OPENAI_GPT_MODEL", "gpt-4o-mini")
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
azure_openai_embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
agent_name = os.getenv("AZURE_SEARCH_AGENT_NAME", "image-search-agent")
api_version = "2025-05-01-Preview"

agent=KnowledgeAgent( 
    name=agent_name, 
    target_indexes=[ 
        KnowledgeAgentTargetIndex( 
            index_name=index_name, default_include_reference_source_data=True, 
            default_reranker_threshold=2.5 
        ) 
    ], 
    models=[ 
        KnowledgeAgentAzureOpenAIModel( 
            azure_open_ai_parameters=AzureOpenAIVectorizerParameters( 
                resource_url=azure_openai_endpoint, 
                deployment_name=azure_openai_gpt_deployment, 
                model_name=azure_openai_gpt_model, 
            ) 
        ) 
    ], 
    request_limits=KnowledgeAgentRequestLimits( 
        max_output_size=agent_max_output_tokens 
    ) 
)

index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential) 
index_client.create_or_update_agent(agent) 

instructions = """
A Q&A agent that can answer questions about the drone images stored in Azure AI Search.
Sources have a JSON description and vector format with a ref_id that must be cited in the answer.
If you do not have the answer, respond with "I don't know".
"""

messages = [
    {
        "role": "system",
        "content": instructions
    }
] 

from azure.search.documents.agent import KnowledgeAgentRetrievalClient 
from azure.search.documents.agent.models import KnowledgeAgentRetrievalRequest, KnowledgeAgentMessage, KnowledgeAgentMessageTextContent, KnowledgeAgentIndexParams 

agent_client = KnowledgeAgentRetrievalClient(endpoint=search_endpoint, agent_name=agent_name, credential=credential)

messages.append({ 
  "role": "user", 
  "content": 
""" 
How many red cars could be found? 
""" 

}) 

# retrieval_result = agent_client.knowledge_retrieval.retrieve( 
   # messages[KnowledgeAgentMessage( 
        # role=msgp["role"], 
        # content=[KnowledgeAgentMessageTextContent(text=msg["content"])]) 
        # for msg in messages if msg["role"] != "system"], 
        # Target_index_params=[KnowedgeAgentIndexParams(index_name=index_name, reranker_threshold=3, include_reference_source_data=True)], 
   # )
# ) 
retrieval_result = agent_client.retrieve(
    retrieval_request=KnowledgeAgentRetrievalRequest(
        messages=[KnowledgeAgentMessage(role=msg["role"], content=[KnowledgeAgentMessageTextContent(text=msg["content"])]) for msg in messages if msg["role"] != "system"],
        target_index_params=[KnowledgeAgentIndexParams(index_name=index_name, reranker_threshold=2.5)]
    )
)

messages.append({ 
   "role": "assistant", 
   "content": retrieval_result.response[0].content[0].text 
}) 

print(messages)
import textwrap

print("Response")
print(textwrap.fill(retrieval_result.response[0].content[0].text, width=120))
import json
print("Activity")
print(json.dumps([a.as_dict() for a in retrieval_result.activity], indent=2))
print("Results")
print(json.dumps([r.as_dict() for r in retrieval_result.references], indent=2))