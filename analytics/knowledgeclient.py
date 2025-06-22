from azure.search.documents.indexes import SearchIndexClient 

from azure.search.documents.indexes.models import ( 
    KnowledgeAgent, 
    KnowledgeAgentAzureOpenAIModel, 
    KnowledgeAgentRequestLimits, 
    KnowledgeAgentTargetIndex,
    AzureOpenAIVectorizerParameters
)
from azure.ai.agents.models import AzureAISearchTool, AzureAISearchQueryType, MessageRole, ListSortOrder


# The search_tool object can now be used within an Azure AI project,
# typically as part of an agent or flow, to perform search operations
# against the specified Azure AI Search index.
# For example, if you are building an agent, this tool could be invoked
# when the agent needs to retrieve information from your search index.
from azure.ai.agents import AgentsClient
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
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
index_name = os.getenv("AZURE_SEARCH_02_INDEX_NAME", "index02")
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_gpt_deployment = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4o-mini")
azure_openai_gpt_model = os.getenv("AZURE_OPENAI_GPT_MODEL", "gpt-4o-mini")
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
azure_openai_embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
agent_name = os.getenv("AZURE_SEARCH_AGENT_NAME", "objects-search-agent")
api_version = "2025-05-01-Preview"
agent_max_output_tokens=10000

"""
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
"""
agents_client = AgentsClient(endpoint=project_endpoint, credential=DefaultAzureCredential())
index_client = SearchIndexClient(endpoint=search_endpoint, credential=AzureKeyCredential(search_api_key)) 
instructions = """
You are an AI assistant that answers questions about the stored and indexed drone images and objects in search index index02.
The data source is an Azure AI Search resource where the schema has JSON description field, a vector field and an id field and this id field must be cited in your answer.
If you do not find a match for the query, respond with "I don't know", otherwise cite references with the value of the id field.
"""

connection_id = os.getenv("AI_AZURE_AI_CONNECTION_ID","https://srch-vision-01.search.windows.net")
# Initialize agent AI search tool and add the search index connection id

# Initialize the AzureAISearchTool
# You can specify optional parameters like query_type, filter, and top_k
search_tool = AzureAISearchTool(
    index_connection_id=connection_id,
    index_name=index_name,
    query_type=AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,
    filter="",  # Optional filter expression
    top_k=3  # Number of results to return
)
# ai_search_tool = AzureSearchToolset(search_endpoint, index_name, search_api_key)

# agents_client.create_agent(agent) 
# agent = agents_client.create_agent(
    # model=agent_model, # azure_openai_embedding_model,
    # name=agent_name,
    # instructions=instructions,
    # tools=search_tool.definitions,
    # tool_resources=search_tool.resources
# )
existing_agents = agents_client.list_agents()
print(f"Agents other than {agent_name}:")
print(",".join([f"{agent.name}:{agent.model}" for agent in existing_agents]))
agent = agents_client.get_agent(agent_id="asst_JI9VWjdav3To7jjGUROejGkV")

# Create a thread for the conversation
thread = agents_client.threads.create()

# Send a user message (the query text)
query_text = "How many red cars can be found?"
query_text = "How many parkings spots are vacant near a building with circular roof?"
query_text = "Find images with a building that has a circular roof."
query_text = "How many times did the drone capturing the images fly over the building with distinct circular roof structure that you mentioned in the image with id 000552?"
query_text = "How many images are similar to the one with the id 000552 that has a building with distinct circular roof structure?"
query_text = "Find images with green bicycle crossing signs at street intersections."
query_text = "How many bicycle crossing signs can be found at street intersections."
query_text = "How far apart is abuilding with distinct circular roof structure from the nearest water body such as a lake?"
query_text = "Find images with big parking lots."
message = agents_client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content=query_text,
)
# Run the agent to process the query
run = agents_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

# Check run status
if run.status == "failed":
    print(f"Run failed: {run.last_error}")

# Retrieve and print all messages in the thread (including agent's answer)
messages = agents_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
for message in messages:
    print(",".join([key for key in message.keys()]))
    print(f"Role: {message.role}, Content: {message.content}, Metadata: {message.metadata}")


# messages = [
    # {
        # "role": "system",
        # "content": instructions
    # }
# ] 

# from azure.search.documents.agent import KnowledgeAgentRetrievalClient 
# from azure.search.documents.agent.models import KnowledgeAgentRetrievalRequest, KnowledgeAgentMessage, KnowledgeAgentMessageTextContent, KnowledgeAgentIndexParams 

# agent_client = KnowledgeAgentRetrievalClient(endpoint=search_endpoint, agent_name=agent_name, credential=credential)

# messages.append({ 
  # "role": "user", 
  # "content": 
# """ 
# How many red cars could be found? 
# """ 

# }) 

# retrieval_result = agent_client.knowledge_retrieval.retrieve( 
   # messages[KnowledgeAgentMessage( 
        # role=msgp["role"], 
        # content=[KnowledgeAgentMessageTextContent(text=msg["content"])]) 
        # for msg in messages if msg["role"] != "system"], 
        # Target_index_params=[KnowedgeAgentIndexParams(index_name=index_name, reranker_threshold=3, include_reference_source_data=True)], 
   # )
# ) 
# retrieval_result = agents_client.retrieve(
    # retrieval_request=KnowledgeAgentRetrievalRequest(
        # messages=[KnowledgeAgentMessage(role=msg["role"], content=[KnowledgeAgentMessageTextContent(text=msg["content"])]) for msg in messages if msg["role"] != "system"],
        # target_index_params=[KnowledgeAgentIndexParams(index_name=index_name, reranker_threshold=2.5)]
    # )
# )

# messages.append({ 
   # "role": "assistant", 
   # "content": retrieval_result.response[0].content[0].text 
# }) 

# print(messages)
# import textwrap

# print("Response")
# print(textwrap.fill(retrieval_result.response[0].content[0].text, width=120))
# import json
# print("Activity")
# print(json.dumps([a.as_dict() for a in retrieval_result.activity], indent=2))
# print("Results")
# print(json.dumps([r.as_dict() for r in retrieval_result.references], indent=2))