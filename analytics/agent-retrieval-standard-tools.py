#!/usr/bin/python
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.agents import AgentsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.projects import AIProjectClient
import os
import sys
load_dotenv(override=True)
sys.path.insert(0, os.path.abspath(".."))
from visionprocessor.vectorizer import vectorize_image, analyze_image
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
vision_api_key = os.getenv("AZURE_AI_VISION_API_KEY")
vision_api_version = os.getenv("AZURE_AI_VISION_API_VERSION")
vision_region = os.getenv("AZURE_AI_VISION_REGION")
vision_endpoint =  os.getenv("AZURE_AI_VISION_ENDPOINT")
vision_credential = AzureKeyCredential(vision_api_key)
bing_connection_resource_id = os.getenv("AZURE_BNG_SEARCH_RESOURCE_ID")
from azure.ai.vision.imageanalysis import ImageAnalysisClient
analysis_client = ImageAnalysisClient(vision_endpoint, vision_credential)


print(f"Agent Name={search_agent_name}")
print(f"Index Name={index_name}")
api_version = "2025-05-01-Preview"
agent_max_output_tokens=10000

from azure.search.documents.indexes.models import KnowledgeAgent, KnowledgeAgentAzureOpenAIModel, KnowledgeAgentTargetIndex, KnowledgeAgentRequestLimits, AzureOpenAIVectorizerParameters
from azure.search.documents.indexes import SearchIndexClient
from azure.ai.agents.models import FileSearchTool
# index_client = SearchIndexClient(endpoint=search_endpoint, credential=AzureKeyCredential(search_api_key))  
# index_client.close()
index_client = SearchIndexClient(endpoint=search_endpoint, credential=AzureKeyCredential(search_api_key))  
from azure.ai.projects import AIProjectClient 
from azure.ai.projects.models import Index
from azure.ai.agents.models import (
    AgentThreadCreationOptions,
    ThreadMessageOptions,
    MessageTextContent,
    ListSortOrder
)
project_client = AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential()) 
"""
print([f"{index.name} {index.id} {index.version} {index.type} {index.description} {index.tags}" for index in project_client.indexes.list()])
pindex = project_client.indexes.create_or_update(name=index_name, version="1",  body={
"name": index_name,
"type": "AzureSearch",
"version": 1,
"ConnectionName": search_endpoint,
"IndexName": index_name
})
print(pindex)
"""

for index_agent in index_client.list_agents():
    print(index_agent.name)
file_agent_name = "file-agent-in-a-team"
file_agent_instructions = "Search files and documents to find relevant information."
bing_agent_name = "web-agent-in-a-team"
bing_agent_instructions = "Search the web to find relevant information".

instructions = ["""
You are an AI assistant that answers questions about the stored and indexed drone images and objects in search index index02.
The data source is an Azure AI Search resource where the schema has JSON description field, a vector field and an id field and this id field must be cited in your answer.
If you do not find a match for the query, respond with "I don't know", otherwise cite references with the value of the id field.
"""]


def get_image_json():
     sas_url_template = "https://sadronevideo.blob.core.windows.net/vi-rendered-wfmat1ysct-0e4129//images/{frame}.jpg?sp=racwdl&st=2025-07-12T19:32:14Z&se=2025-07-16T03:47:14Z&spr=https&sv=2024-11-04&sr=c&sig=frSTDOQIWY%2B7FTlmrPW%2BDfI0ctLTVfJ4KOs60qEgGhs%3D"
     for i in range(26,27):
        sas_url = sas_url_template.replace("{frame}", f"frame{i}")
        desc = analyze_image(analysis_client, sas_url)
        if desc:
            print(f"writing frame{i}.json")
            with open(f"frame{i}.json", 'w', encoding="utf-8") as fout:
                fout.write(desc)

def create_connected_agent(name, instructions, tools):
    return project_client.agents.create_agent(
        model=azure_openai_gpt_model, 
        # deployment=azure_openai_gpt_deployment, 
        name=name, 
        instructions=instructions,
        tools=tools.definitions,
        tool_resources=tools.resources,
        top_p=1
    )

def get_file_agent(name, instructions):
    file_search_agent = None
    for agent in project_client.agents.list_agents():
        print(f"{agent.name} matches {agent.name == name}")
        if agent.name == name:
            file_search_agent = agent
            break
    # file_search_agent = [ agent for agent in project_client.agents.list_agents() if agent.name == name][0] # get_agent(agent_id=name)
    if not file_search_agent:
        print("Do you want me to create A File Search Agent?")
        return None
    return file_search_agent
  
def get_bing_agent(name, instructions):
    bing_search_agent = None
    for agent in project_client.agents.list_agents():
        print(f"{agent.name} matches {agent.name == name}")
        if agent.name == name:
            bing_search_agent = agent
            break
    # file_search_agent = [ agent for agent in project_client.agents.list_agents() if agent.name == name][0] # get_agent(agent_id=name)
    if not bing_search_agent:
        print("Do you want me to create A Bing Search Agent?")
        # 2. Bing Agent
        bing_search_tool = BingGroundingTool(    
            connection_id=bing_connection_resource_id)
        bing_search_agent = create_connected_agent(
            name=name,
            instructions=instructions,
            tools=[bing_search_tool]
        )
    return bing_search_agent
        
"""       
        # 2. File Search Agent
        file_search_tool = FileSearchTool(    
            vector_store_id=index_name,  # This is your Azure AI Search index name
            vector_field="vector",      # The field storing embeddings
            endpoint=search_endpoint,
            api_key=search_api_key)
        file_search_agent = create_connected_agent(
            name="file_search_agent",
            instructions="Search files and documents to find relevant information.",
            tools=[file_search_tool]
        )
    return file_search_agent
"""
# file_agent = get_file_agent(file_agent_name, file_agent_instructions)    
agent = get_bing_agent(bing_agent_name, bing_agent_instructions)
def get_response(agent, instructions, query):
    messages = [
        {
            "role":"assistant",
            "content": instructions
        }
    ]
    run = project_client.agents.create_thread_and_process_run(agent_id = agent.id, 
        thread = AgentThreadCreationOptions(messages = [
        ThreadMessageOptions(role="assistant", content=instructions),
        ThreadMessageOptions(role="user", content=query)]),)
    print(run)
    if run.status == "failed":
        print(f"Run error: {run.last_error}")

    # List all messages in the thread, in ascending order of creation
    messages = project_client.agents.messages.list(
        thread_id=run.thread_id, ## "thread_gX5kKqSaPvtR5ISQSkTCZVdk"
        order=ListSortOrder.ASCENDING,
    )
    for msg in messages:
        last_part = msg.content[-1]
        if isinstance(last_part, MessageTextContent):
            print(f"{msg.role}: {last_part.text.value}")
    return last_part.text.value

    # index_agent.clear()
# agent = index_client.get_agent(search_agent_name)
# if  agent is None:
    # print("no agent found")
    # model =  KnowledgeAgentAzureOpenAIModel( 
            # azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                # resource_url=azure_openai_endpoint, 
                # deployment_name=azure_openai_gpt_deployment, 
                # model_name=azure_openai_gpt_model,
                # api_key = azure_openai_api_key                 
            # ) 
        # )
    # agent = KnowledgeAgent( 
        # name=search_agent_name, 
        # models=[ 
            # model
        # ], 
        # target_indexes=[ 
            # KnowledgeAgentTargetIndex( 
                # index_name=index_name, 
                # default_include_reference_source_data=True, 
                # default_reranker_threshold=2.5 
            # ) 
        # ], 
        # request_limits=KnowledgeAgentRequestLimits( 
            # max_output_size=10000 
        # ) 
    # )
    # r = index_client.create_or_update_agent(agent=agent) 
    # print(f"AI Knowledge agent '{search_agent_name}' created or updated successfully: {r}")     
# print(agent) 


from azure.ai.agents.models import FunctionTool, ToolSet, ListSortOrder

# from azure.search.documents.agent import KnowledgeAgentRetrievalClient
# from azure.search.documents.agent.models import KnowledgeAgentRetrievalRequest, KnowledgeAgentMessage, KnowledgeAgentMessageTextContent, KnowledgeAgentIndexParams

# agent_client = KnowledgeAgentRetrievalClient(endpoint=search_endpoint, agent_name=search_agent_name, credential=AzureKeyCredential(search_api_key))
# # query_text = "How many parking lots are empty when compared to all the parking lots?" 
# query_text = "How many red cars can be found near the building with a roof that has a circular structure?"
# query_text = "How many parking can be found near the building with a roof that has a circular structure?"
# query_text = "How far is a split in the road from the building with a roof that has a circular structure?"
# query_text = "Is there a green car?"
# query_text = "How far is a skyscraper from the lake?"
# query_text = "Find the image with the tallest building among all."
# query_text = "Which is the tallest building among all the buildings in images?"
# query_text = "List all types of vehicles found on the roads or parking lots from the images"
# query_text = "How many red cars can be found near the building with a roof that has a circular structure?"
# query_text = "How many different colors of cars can be found on the roads near the building with a roof that has a circular structure?"
# query_text = "How many storeys are there in buildings close to parking lots?"
# query_text = "Is there a parking lot building with cars on the roof?"
# query_text = "Are there triangular buildings for parking?"
# messages.append({
    # "role": "user",
    # "content": query_text
# })
# # print(agent_client)
# print([msg["content"] for msg in messages if msg["role"] != "system"])
# retrieval_result = agent_client.retrieve(
    # retrieval_request=KnowledgeAgentRetrievalRequest(
        # messages=[KnowledgeAgentMessage(role=msg["role"], content=[KnowledgeAgentMessageTextContent(text=msg["content"])]) for msg in messages if msg["role"] != "system"],
        # target_index_params=[KnowledgeAgentIndexParams(index_name=index_name, reranker_threshold=2.5, include_reference_source_data=True)] # add filter_add_on here
    # )
# )
# # print(f"Response={retrieval_result.response[0].content[0]}")
# print(f"Result={retrieval_result.response[0].content[0].text}")

if agent:
    query_text = "Are there dedicated bicycle crossings in green color at street intersections?"
    query_text = "lone red car in a scene"
    #response = get_response(agent, file_agent_instructions, query_text)
    print(response)
else:
    print("No agent found.")


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
