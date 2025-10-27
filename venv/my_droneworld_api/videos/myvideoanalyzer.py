#!/usr/bin/python
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.agents import AgentsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.projects import AIProjectClient
from azure.search.documents.indexes.models import (
    KnowledgeAgent, 
    KnowledgeAgentAzureOpenAIModel, 
    # KnowledgeAgentTargetIndex, 
    KnowledgeAgentRequestLimits, 
    AzureOpenAIVectorizerParameters, 
    KnowledgeAgentOutputConfiguration,
    KnowledgeAgentOutputConfigurationModality,
    KnowledgeSourceReference,
    SearchIndexKnowledgeSource,
    SearchIndexKnowledgeSourceParameters
    # ConnectedAgentReference
)
from azure.search.documents.indexes import SearchIndexClient
from azure.ai.agents.models import (
    FunctionTool,
    ToolSet,
    ListSortOrder,
    ConnectedAgentTool,
    RequiredFunctionToolCall,
    RunStepAzureAISearchToolCall,
    SubmitToolOutputsAction,
    ToolOutput,
)
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import KnowledgeAgentRetrievalRequest, KnowledgeAgentMessage, KnowledgeAgentMessageTextContent # , KnowledgeAgentIndexParams
from django.conf import settings
from typing import Any, Callable, Set, Dict, List, Optional
import os, time, sys
sys.path.insert(0, os.path.abspath("."))
load_dotenv(override=True)

vision_api_key = settings.AZURE_AI_VISION_API_KEY
vision_api_version = settings.VISION_API_VERSION
vision_region = settings.AZURE_AI_VISION_REGION
vision_endpoint =  settings.AZURE_AI_VISION_ENDPOINT
api_version = settings.SEARCH_API_VERSION
model_version = settings.MODEL_VERSION
connection_id = settings.AZURE_SEARCH_CONNECTION_ID
search_endpoint = settings.AZURE_SEARCH_SERVICE_ENDPOINT
search_api_key  = settings.AZURE_SEARCH_ADMIN_KEY
index_name = "index007" # /versions/1" # settings.AZURE_SEARCH_INDEX_NAME
video_indexer_endpoint = settings.AZURE_VIDEO_INDEXER_URL
video_indexer_region = settings.AZURE_VIDEO_INDEXER_REGION
video_indexer_account_id = settings.AZURE_VIDEO_INDEXER_ACCOUNT
perplexity_geo_api_key = settings.PERPLEXITY_GEO_API_KEY
perplexity_geo_api_url = settings.PERPLEXITY_GEO_API_URL
search_location = settings.AZURE_SEARCH_LOCATION or "eastus"
search_subscription = settings.AZURE_SEARCH_SUBSCRIPTION
search_resource_group = settings.AZURE_SEARCH_RESOURCE_GROUP
search_service_name = settings.AZURE_SEARCH_SERVICE_NAME
search_resource_id = settings.AZURE_SEARCH_RESOURCE_ID or f"/subscriptions/{search_subscription}/resourceGroups/{search_resource_group}/providers/Microsoft.Search/searchServices/{search_service_name}"
object_uri = settings.SAMPLE_OBJECT_URI.strip('"')
scene_uri = settings.SAMPLE_SCENE_URI.strip('"')


project_endpoint = settings.AZURE_PROJECT_ENDPOINT
project_api_key = settings.AZURE_PROJECT_API_KEY
agent_model = settings.AZURE_AGENT_MODEL
embedding_model = settings.AZURE_EMBEDDING_MODEL
azure_openai_endpoint = settings.AZURE_OPENAI_ENDPOINT
azure_openai_api_key = settings.AZURE_OPENAI_API_KEY
azure_openai_gpt_deployment = settings.AZURE_OPENAI_GPT_DEPLOYMENT
azure_openai_gpt_model = settings.AZURE_OPENAI_GPT_MODEL
azure_openai_embedding_deployment = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
azure_openai_embedding_model = settings.AZURE_OPENAI_EMBEDDING_MODEL

fn_agent_name = settings.AZURE_FN_AGENT_NAME
chat_agent_name = settings.AZURE_CHAT_AGENT_NAME
search_agent_name = settings.AZURE_SEARCH_AGENT_NAME

perplexity_api_key = settings.PERPLEXITY_CHAT_API_KEY
perplexity_api_url = settings.PERPLEXITY_CHAT_API_URL

credential = AzureKeyCredential(search_api_key)
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://search.azure.com/.default")


agent_max_output_tokens=10000

def delete_all_threads_for_agent(agent_name):
    project_client = AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential()) 
    agents_client = AgentsClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
    )
    agent = None
    for entry in agents_client.list_agents():
        if entry.name == agent_name:
            agent = entry
            break;
    if not agent:
        print(f"cannot find threads to delete for agent: {agent_name}")
        return
    # Find the agent ID for the given name
    agent_id = agent.id

    # List all threads associated with this agent
    threads = agents_client.threads.list() # agent_id=agent_id)
    # Iterate and delete each thread
    for thread in threads:
        agents_client.threads.delete(thread_id=thread.id)
        print(f"Deleted thread ID: {thread.id}")
    return
        
# all agentic_framework requires a knowledgeAgent that can automate query decomposition and rewriting, refer: https://learn.microsoft.com/en-us/azure/search/search-agentic-retrieval-how-to-pipeline?tabs=search-perms

# def knowledge_base_search(query_text, account_id):
def run_connected_agent(query_text, account_id):
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=AzureKeyCredential(search_api_key))  
    agent = None
    for index_agent in index_client.list_agents():
        print(f"Agent={index_agent.name}")
        if index_agent.name == search_agent_name:
            # index_client.delete_agent(agent=index_agent)
            # print(f"{search_agent_name} deleted successfully")
            agent = index_agent
    print(f"Found Agent={agent}")
    if  agent is None:
        print("no agent found")
        # create a KnowledgeSource first then a search index agent with that knowledge source
        knowledge_source = None
        for source in index_client.list_knowledge_sources():
            if source.name == index_name:
                print(f"Found Knowledge source: {source}")
                # index_client.delete_knowledge_source(knowledge_source=source.name)
                knowledge_source = source
                
        if not knowledge_source:
            print(f"creating a knowledge_source with name: {index_name}")
            knowledge_source = SearchIndexKnowledgeSource(
                name=index_name, 
                search_index_parameters=SearchIndexKnowledgeSourceParameters(
                    search_index_name=index_name,
                    source_data_select="id,account_id,description,location,created"))
            index_client.create_knowledge_source(knowledge_source=knowledge_source, api_version=api_version)
        
        retrieval_text="You are an aerial drone image analyst. If an account_id is provided to you along with the query, use it select only those images from the index that have account_id matching and then respond based on those images and their associated vectors and fields."
        model =  KnowledgeAgentAzureOpenAIModel( 
                azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                    resource_url=azure_openai_endpoint, 
                    deployment_name=azure_openai_gpt_deployment, 
                    model_name=azure_openai_gpt_model,
                    api_key = azure_openai_api_key                 
                ) 
            )
        # connected_agent_specialized_tasks = ConnectedAgentReference(
            # name=fn_agent_name
        # )
        output_cfg = KnowledgeAgentOutputConfiguration(
            modality=KnowledgeAgentOutputConfigurationModality.ANSWER_SYNTHESIS,
            include_activity=True,
        )
        agent = knowledge_agent = KnowledgeAgent(
            name=search_agent_name,
            models=[
                model
            ],
            knowledge_sources=[
                KnowledgeSourceReference(
                    name=index_name,
                    include_references=True,
                    include_reference_source_data=False,
                    always_query_soure=True,
                    max_sub_queries=10,
                    reranker_threshold=2.5
                )
            ], 
            request_limits=KnowledgeAgentRequestLimits( 
                max_output_size=10000 
            ),
            retrieval_instructions=retrieval_text,
            output_configuration=output_cfg
            # connected_agents = [connected_agent_specialized_tasks]
        )
        r = index_client.create_or_update_agent(agent=agent) 
        print(f"AI Knowledge agent '{search_agent_name}' created or updated successfully: {r}")
    print(agent)


    instructions = ["""
    You are an AI assistant that answers questions about the stored and indexed drone images and objects in Azure AI Search index index007/versions/1.
    The data source is an Azure AI Search resource where the schema has JSON description field, a vector field and an id field and this id field must be cited in your answer.
    If you do not find a match for the query, respond with "I don't know", otherwise cite references with the value of the id field.
    """]
    messages = [
        {
            "role":"system",
            "content": instructions
        }
    ]

    agent_client = KnowledgeAgentRetrievalClient(endpoint=search_endpoint, agent_name=search_agent_name, credential=AzureKeyCredential(search_api_key))
    messages.append({
        "role": "user",
        "content": query_text
    })
    print([msg["content"] for msg in messages if msg["role"] != "system"])
    # retrieval_result = agent_client.retrieve(
        # retrieval_request=KnowledgeAgentRetrievalRequest(
            # messages=[KnowledgeAgentMessage(role=msg["role"], content=[KnowledgeAgentMessageTextContent(text=msg["content"])]) for msg in messages if msg["role"] != "system"],
            # target_index_params=[KnowledgeAgentIndexParams(index_name=index_name, reranker_threshold=2.5, include_reference_source_data=True,filter_add_on=f"startwith(id,{account_id})")] # add filter_add_on here using the syntax at https://learn.microsoft.com/en-us/azure/search/search-query-odata-filter
        # )
    # )
    
    retrieval_request = KnowledgeAgentRetrievalRequest(
        messages=[
            KnowledgeAgentMessage(
                role=msg["role"],
                content=[KnowledgeAgentMessageTextContent(text=msg["content"])]
            )
            for msg in messages if msg["role"] != "system"
        ]
    )
    retrieval_result = agent_client.retrieve(retrieval_request=retrieval_request, api_version=api_version)
    print([response for response in retrieval_result.response])
    print(f"Result={retrieval_result.response[0].content[0].text}")
    return retrieval_result.response[0].content[0].text


def knowledge_base_search(query_text, account_id):
    answer = None
    project_client = AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential()) 
    agents_client = AgentsClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
    )
    connected_agent_name = "master-agent-in-a-team"
    from .analyzer_functions import analyzer_functions, image_user_functions
    from azure.ai.agents.models import (
        AzureAISearchTool,
        AzureAISearchQueryType,
        ConnectedAgentTool
    )
    from azure.ai.projects.models import ConnectionType
    # from azure.ai.projects.models import (
        # ConnectionResource,
        # ConnectionType
    # )
    # image_user_functions: Set[Callable[..., Any]] = {
        # agentic_retrieval
    # }

    search_connection_name = index_name
    existing_connection = None
    connected_agent = None
    ai_search_agent = None
    for entry in project_client.agents.list_agents():
        print(f"Listing Agent: id: {entry.id}, name: {entry.name}, model: {entry.model}")
        if entry.name == fn_agent_name:
            connected_agent = entry
        if entry.name == connected_agent_name:
            ai_search_agent = entry
        # Listing Agent: id: asst_FNwlA7fqvDf4WdbVtsIHYQZS, name: master-agent-in-a-team, model: gpt-4o-mini
        # Listing Agent: id: asst_v2Hj4CJ5wEW2gqGwG44YtbD4, name: fn-agent-in-a-team, model: gpt-4o-mini
        # Listing Agent: id: asst_ilwEdVRNApUDmqa2EB3sSBKp, name: file-agent-in-a-team, model: gpt-4o-mini
        # Listing Agent: id: asst_lsH8uwS4hrg4v1lRpXm6sdtR, name: chat-agent-in-a-team, model: gpt-4o-mini
        # Listing Agent: id: asst_JI9VWjdav3To7jjGUROejGkV, name: object-search-agent, model: gpt-4o-mini
    for deployment in project_client.deployments.list():
        print(f"Deployment: type:{deployment.type}, name:{deployment.name}")
        if "id" in deployment and deployment.id:
            print(f"Deployment_id: {deployment.id}")
            # Deployment: type:ModelDeployment, name:gpt-4o-mini
            # Deployment: type:ModelDeployment, name:text-embedding-ada-002
    model_deployment_id = "/subscriptions/656e67c6-f810-4ea6-8b89-636dd0b6774c/resourceGroups/rg-ctl-2/providers/Microsoft.CognitiveServices/accounts/found-vision-1/projects/droneimage/deployments/gpt-4o-mini"
    for conn in project_client.connections.list():
        # print(f"Connection: id:{conn.id}, name:{conn.name}, type: {conn.type}, target: {conn.target}, is_default: {conn.is_default}")
        # Connection:id:/subscriptions/656e67c6-f810-4ea6-8b89-636dd0b6774c/resourceGroups/rg-ctl-2/providers/Microsoft.CognitiveServices/accounts/found-vision-1/projects/droneimage/connections/srchvision01, name:srchvision01, type: ConnectionType.AZURE_AI_SEARCH, target: https://srch-vision-01.search.windows.net/, is_default: True
        # Connection: id:/subscriptions/656e67c6-f810-4ea6-8b89-636dd0b6774c/resourceGroups/rg-ctl-2/providers/Microsoft.CognitiveServices/accounts/found-vision-1/projects/droneimage/connections/LogicApps_Tool_Connection_fnagentaction_7461, name:LogicApps_Tool_Connection_fnagentaction_7461, type: ConnectionType.CUSTOM, target: _, is_default: True
        if conn.name == search_connection_name and conn.connection_type == ConnectionType.AZURE_AI_SEARCH:
            existing_connection = conn
            print(f"Found existing knowledge source connection id: {conn.name}")
            break
    """        
    if not existing_connection:
        print(f"Creating connection to knowledge source by name: {search_connection_name}")
        created_connection = project_client.connections.create(
            name=search_connection_name,
            connection_type=ConnectionType.AZURE_AI_SEARCH,
            target=search_endpoint,
            auth_type="ApiKey",
            credentials={"key": api_key},
            metadata={"description": "Programmatic connection for Azure AI Search index007"}
        )
        existing_connection = created_connection
    search_connection_id = existing_connection.id
    """
    search_connection_id = connection_id
    # print(f"connection-id: {search_connection_id}")
    # Initialize search tool definition
    ai_search_tool = AzureAISearchTool(
        index_connection_id=search_connection_id,
        index_name=index_name,
        query_type=AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,
        top_k=3,
        filter = "account_id eq '" + account_id + "'"
        # filter="startwith(account_id,'" + account_id + "')"
    )
    print(f"ai_search_tool created for_agent")
    connected_agent_instructions = "If the search over the Azure AI search index does not provide a conclusive answer for the user query, then answer the question by finding a suitable function, passing the question to the function, evaluating it and relaying the response from the function. If you can't find a suitable function, default to the ask_perplexity function included in your tools."
    connected_agent_tool = ConnectedAgentTool(
        id=connected_agent.id, 
        name="connected_agent", 
        description=connected_agent_instructions
    )
    # Initialize search call
    instructions = "You are a drone aerial image analytics assistant that answers the caller's question by searching an azure search index or delegating to connected agent, evaluating the responses and synthesizing a comprehensive response back to the caller. If you can't find a suitable answer, reply with I do not know."
    # query_text = f"How many objects given by its image URI {object_uri} are found in the image given by its image URI {scene_uri}?"
    with agents_client:
        agent = None
        for entity in agents_client.list_agents():
            if entity.name == connected_agent_name:  
                agent = entity
        if  agent is None:
            print("no agent found")
            agent = agents_client.create_agent(
                model="gpt-4o-mini",
                name=connected_agent_name,
                instructions=instructions,
                tools=ai_search_tool.definitions + connected_agent_tool.definitions,
                tool_resources=ai_search_tool.resources + connected_agent_tool.resources,
                top_p=1
            )
            # """
            #print(f"Created agent, ID: {agent.id}")
        print(f"Agent found, ID: {agent.id}") 
        thread = agents_client.threads.create()
        print(f"Created thread, ID: {thread.id}")

        message = agents_client.messages.create(
            thread_id=thread.id,
            role="user",
            content=query_text,
        )
        print(f"Created message, ID: {message.id}")

        run = agents_client.runs.create(thread_id=thread.id, agent_id=agent.id)
        print(f"Created run, ID: {run.id}")

        while run.status in ["queued", "in_progress", "requires_action"]:
            time.sleep(1)
            run = agents_client.runs.get(thread_id=thread.id, run_id=run.id)

            if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                if not tool_calls:
                    print("No tool calls provided - cancelling run")
                    agents_client.runs.cancel(thread_id=thread.id, run_id=run.id)
                    break

                tool_outputs = []
                for tool_call in tool_calls:
                    print(f"Tool Call id: {tool_call.id}, type:{tool_call.type}")
                    if isinstance(tool_call, RunStepAzureAISearchToolCall):
                        #print("Is an instance of RequiredFunctionToolCall")
                        try:
                            #print(f"Executing tool call: {tool_call}")
                            output = ai_search_tool.execute(tool_call)
                            print(f"Output={output}")
                            answer = output
                            tool_outputs.append(
                                ToolOutput(
                                    tool_call_id=tool_call.id,
                                    output=output,
                                )
                            )
                        except Exception as e:
                            print(f"Error executing tool_call {tool_call.id}: {e}")
                    else:
                        print(f"{tool_call} skipped.")

                print(f"Tool outputs: {tool_outputs}")
                if tool_outputs:
                    agents_client.runs.submit_tool_outputs(thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs)
                else:
                    print(f"No tool output.")
            else:
                print(f"Waiting: {run}")

            print(f"Current run status: {run.status}")

        print(f"Run completed with status: {run.status} and details {run}")

        # Delete the agent when done
        # agents_client.delete_agent(agent.id)
        # print("Deleted agent")

        # Fetch and log all messages
        
        messages = agents_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
        for msg in messages:
            print(f"msg={msg}")
            if msg.text_messages:
                last_text = msg.text_messages[-1]
                print(f"{msg.role}: {last_text.text.value}")
                answer = last_text.text.value
        print(f"answer={answer}")
        for entry in project_client.agents.list_agents():
            print(f"Listing Agent: id: {entry.id}, name: {entry.name}, model: {entry.model}")
        return answer
        

def synthesize_from_agents(query_text, account_id):
    # Query the Search-backed agent
    knowledge_search_answer = knowledge_base_search(query_text, account_id)
    delegated_answer = run_function_tools(query_text, account_id)
    # Synthesize by prompting the composite agent
    synthesis_prompt = "Combine and summarize insights from the following responses to form a cohesive answer."
    synthesis_response = f"""
    
    [Search Agent Output]:
    {knowledge_search_answer}

    [Connected Agent Output]:
    {delegated_answer}
    """
    return synthesis_response
    
    

def run_function_tools(query_text, account_id):
    answer = None
    project_client = AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential()) 
    agents_client = AgentsClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
    )
    from .analyzer_functions import analyzer_functions, image_user_functions
    # image_user_functions: Set[Callable[..., Any]] = {
        # agentic_retrieval
    # }

    # Initialize function tool with user functions
    functions = FunctionTool(functions=image_user_functions)
    instructions = "You are a drone aerial image analytics assistant that answers the question by finding a suitable function, passing the question to the function, evaluating it and relaying the response from the function. If you can't find a suitable function, default to the ask_perplexity function included in your tools."
    # query_text = f"How many objects given by its image URI {object_uri} are found in the image given by its image URI {scene_uri}?"
    agent = None
    for entity in agents_client.list_agents():
        if entity.name == fn_agent_name:  
            agent = entity
    with agents_client:
        # agent = agents_client.get_agent("asst_v2Hj4CJ5wEW2gqGwG44YtbD4") # fn_agent_name
        if  agent is None:
            print("no agent found")
            # 
            # Create an agent and run user's request with function calls
            # agent = agents_client.get_agent(agent_id="asst_qyMFcz1BnU0BS0QUmhxAAyFk")
            # """
            agent = agents_client.create_agent(
                model=agent_model,
                name=fn_agent_name,
                instructions=instructions,
                tools=functions.definitions,
                tool_resources=functions.resources,
                top_p=1
            )
            # """
            #print(f"Created agent, ID: {agent.id}")
        print(f"Agent found, ID: {agent.id}") 
        thread = agents_client.threads.create()
        print(f"Created thread, ID: {thread.id}")

        message = agents_client.messages.create(
            thread_id=thread.id,
            role="user",
            content=query_text,
        )
        #print(f"Created message, ID: {message.id}")

        run = agents_client.runs.create(thread_id=thread.id, agent_id=agent.id)
        #print(f"Created run, ID: {run.id}")

        while run.status in ["queued", "in_progress", "requires_action"]:
            time.sleep(1)
            run = agents_client.runs.get(thread_id=thread.id, run_id=run.id)

            if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                if not tool_calls:
                    print("No tool calls provided - cancelling run")
                    agents_client.runs.cancel(thread_id=thread.id, run_id=run.id)
                    break

                tool_outputs = []
                for tool_call in tool_calls:
                    if isinstance(tool_call, RequiredFunctionToolCall):
                        #print("Is an instance of RequiredFunctionToolCall")
                        try:
                            #print(f"Executing tool call: {tool_call}")
                            output = functions.execute(tool_call)
                            print(f"Output={output}")
                            answer = output
                            tool_outputs.append(
                                ToolOutput(
                                    tool_call_id=tool_call.id,
                                    output=output,
                                )
                            )
                        except Exception as e:
                            print(f"Error executing tool_call {tool_call.id}: {e}")
                    else:
                        print(f"{tool_call} skipped.")

                print(f"Tool outputs: {tool_outputs}")
                if tool_outputs:
                    agents_client.runs.submit_tool_outputs(thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs)
                else:
                    print(f"No tool output.")
            else:
                print(f"Waiting: {run}")

            print(f"Current run status: {run.status}")

        print(f"Run completed with status: {run.status} and details {run}")

        # Delete the agent when done
        # agents_client.delete_agent(agent.id)
        # print("Deleted agent")

        # Fetch and log all messages
        """
        messages = agents_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
        for msg in messages:
            if msg.text_messages:
                last_text = msg.text_messages[-1]
                print(f"{msg.role}: {last_text.text.value}")
                return last_text.text.value
        """
        # print(f"answer={answer}")
        return answer
        


"""
search-agent-in-a-team
 
You are an AI assistant that answers questions about the stored and indexed drone images and objects in search index index02. 
The data source is an Azure AI Search resource where the schema has JSON description field, a vector field and an id field and this id field must be cited in your answer. 
If you do not find a match for the query, respond with "I don't know", otherwise cite references with the value of the id field. 

gpt-4o-mini (version:2024-07-18)

index007/versions/1

specialized_tasks => asst_v2Hj4CJ5wEW2gqGwG44YtbD4
file_agent => asst_ilwEdVRNApUDmqa2EB3sSBKp

"""