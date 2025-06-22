#!/usr/bin/python 
# azure-ai-agents==1.0.0 
# azure-ai-projects==1.0.0b11 
# azure-ai-vision-imageanalysis==1.0.0 
# azure-common==1.1.28 
# azure-core==1.34.0 
# azure-identity==1.22.0 
# azure-search-documents==11.6.0b12 
# azure-storage-blob==12.25.1 
# azure_ai_services==0.1.0 
from dotenv import load_dotenv 
from azure.identity import DefaultAzureCredential, get_bearer_token_provider 
from azure.ai.agents import AgentsClient 
from azure.core.credentials import AzureKeyCredential 
from azure.ai.projects import AIProjectClient 
from azure.ai.agents.models import AzureAISearchTool, AzureAISearchQueryType, MessageRole, ListSortOrder
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
search_connection_id = os.environ["AI_AZURE_AI_CONNECTION_ID"] # resource id of AI Search resource
api_version = "2025-05-01-Preview" 
agent_max_output_tokens=10000 
 
from azure.search.documents.indexes.models import KnowledgeAgent, KnowledgeAgentAzureOpenAIModel, KnowledgeAgentTargetIndex, KnowledgeAgentRequestLimits, AzureOpenAIVectorizerParameters 
from azure.search.documents.indexes import SearchIndexClient 
from azure.ai.projects import AIProjectClient 
project_client = AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential()) 

instructions = """ 
You are an AI assistant that answers questions about the stored and indexed drone images and objects in search index index02. 
The data source is an Azure AI Search resource where the schema has JSON description field, a vector field and an id field and this id field must be cited in your answer. 
If you do not find a match for the query, respond with "I don't know", otherwise cite references with the value of the id field. 
""" 
messages = [ 
    { 
        "role":"system", 
        "content": instructions 
    } 
] 

search_tool = AzureAISearchTool(
    index_connection_id=search_connection_id,
    index_name=index_name,
    query_type=AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,
    filter="",  # Optional filter expression
    top_k=5  # Number of results to return
)
agent = None
for existing_agent in list(project_client.agents.list_agents()):
    if existing_agent.name == search_agent_name:
        print(existing_agent.id)
        agent = existing_agent
if agent == None:
    agent = project_client.agents.create_agent( 
        model=azure_openai_gpt_model, 
        # deployment=azure_openai_gpt_deployment, 
        name=search_agent_name, 
        instructions=instructions,
        tools=search_tool.definitions,
        tool_resources=search_tool.resources,
        top_p=1
    ) 
# agent = project_client.agents.get_agent("asst_lsH8uwS4hrg4v1lRpXm6sdtR")
print(f"AI agent '{search_agent_name}' created or retrieved successfully:{agent}") 
from azure.ai.agents.models import FunctionTool, ToolSet, ListSortOrder 
from azure.search.documents.agent import KnowledgeAgentRetrievalClient 
from azure.search.documents.agent.models import KnowledgeAgentRetrievalRequest, KnowledgeAgentMessage, KnowledgeAgentMessageTextContent, KnowledgeAgentIndexParams 
 
agent_client = KnowledgeAgentRetrievalClient(endpoint=search_endpoint, agent_name=search_agent_name, credential=credential) 

# query_text = "How many parking lots are empty when compared to all the parking lots?" 
query_text = "How many red cars can be found near the building with a roof that has a circular structure?"

messages.append({ 
    "role": "user", 
    "content": query_text
    
    #"How many parking lots are empty when compared to all the parking lots?"
    
}) 
 
thread = project_client.agents.threads.create() 
retrieval_results = {} 
 
def agentic_retrieval() -> str: 
    #    Searches drone images about objects detected and their facts. 
    #    The returned string is in a JSON format that contains the reference id. 
    #    Be sure to use the same format in your agent's response 
    #    You must refer to references by id number 
    # Take the last 5 messages in the conversation 
    messages = project_client.agents.messages.list(thread.id, limit=5, order=ListSortOrder.DESCENDING) 
    # Reverse the order so the most recent message is last 
    messages = list(messages) 
    messages.reverse() 
    retrieval_result = agent_client.retrieve( 
        retrieval_request=KnowledgeAgentRetrievalRequest( 
            messages=[KnowledgeAgentMessage(role=msg["role"], content=[KnowledgeAgentMessageTextContent(text=msg["content"])]) for msg in messages if msg["role"] != "system"], 
            target_index_params=[KnowledgeAgentIndexParams(index_name=index_name, reranker_threshold=2.5, include_reference_source_data=True)] # add filter_add_on here 
        ) 
    ) 
 
    # Associate the retrieval results with the last message in the conversation 
    last_message = messages[-1] 
    retrieval_results[last_message.id] = retrieval_result 
 
    # Return the grounding response to the agent 
    return retrieval_result.response[0].content[0].text 
 
# https://learn.microsoft.com/en-us/azure/ai-services/agents/how-to/tools/function-calling 
functions = FunctionTool({ agentic_retrieval }) 
toolset = ToolSet() 
toolset.add(functions)
toolset.add(search_tool)
project_client.agents.enable_auto_function_calls(toolset) 
 
from azure.ai.agents.models import AgentsNamedToolChoice, AgentsNamedToolChoiceType, FunctionName 
 
message = project_client.agents.messages.create( 
    thread_id=thread.id, 
    role="user", 
    content = query_text
    # "How many red cars can be found near a building with a roof that has a circular structure?"
    # content= "How many parking lots are empty when compared to all the parking lots?" 
 
) 
 
run = project_client.agents.runs.create_and_process( 
    thread_id=thread.id, 
    agent_id=agent.id, 
    tool_choice=AgentsNamedToolChoice(type=AgentsNamedToolChoiceType.FUNCTION, function=FunctionName(name="agentic_retrieval")), 
    toolset=toolset) 
if run.status == "failed": 
    raise RuntimeError(f"Run failed: {run.last_error}") 
output = project_client.agents.messages.get_last_message_text_by_role(thread_id=thread.id, role="assistant").text.value 
 
print("Agent response:", output.replace(".", "\n")) 
 
import json 
 
retrieval_result = retrieval_results.get(message.id) 
if retrieval_result is None: 
    raise RuntimeError(f"No retrieval results found for message {message.id}") 
 
print("Retrieval activity") 
print(json.dumps([activity.as_dict() for activity in retrieval_result.activity], indent=2)) 
print("Retrieval results") 
print(json.dumps([reference.as_dict() for reference in retrieval_result.references], indent=2))