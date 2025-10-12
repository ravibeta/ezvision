#!/usr/bin/python
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.agents import AgentsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.projects import AIProjectClient
from azure.search.documents.indexes.models import KnowledgeAgent, KnowledgeAgentAzureOpenAIModel, KnowledgeAgentTargetIndex, KnowledgeAgentRequestLimits, AzureOpenAIVectorizerParameters
from azure.search.documents.indexes import SearchIndexClient
from azure.ai.agents.models import (
    FunctionTool,
    ToolSet,
    ListSortOrder,
    RequiredFunctionToolCall,
    SubmitToolOutputsAction,
    ToolOutput,
)
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import KnowledgeAgentRetrievalRequest, KnowledgeAgentMessage, KnowledgeAgentMessageTextContent, KnowledgeAgentIndexParams
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
search_endpoint = settings.AZURE_SEARCH_SERVICE_ENDPOINT
search_api_key  = settings.AZURE_SEARCH_ADMIN_KEY
index_name = settings.AZURE_SEARCH_INDEX_NAME
video_indexer_endpoint = settings.AZURE_VIDEO_INDEXER_URL
video_indexer_region = settings.AZURE_VIDEO_INDEXER_REGION
video_indexer_account_id = settings.AZURE_VIDEO_INDEXER_ACCOUNT
perplexity_geo_api_key = settings.PERPLEXITY_GEO_API_KEY
perplexity_geo_api_url = settings.PERPLEXITY_GEO_API_URL

object_uri = settings.SAMPLE_OBJECT_URI.strip('"')
scene_uri = settings.SAMPLE_SCENE_URI.strip('"')


project_endpoint = settings.AZURE_PROJECT_ENDPOINT
project_api_key = settings.AZURE_PROJECT_API_KEY
agent_model = settings.AZURE_AGENT_MODEL
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

# all agentic_framework requires a knowledgeAgent that can automate query decomposition and rewriting, refer: https://learn.microsoft.com/en-us/azure/search/search-agentic-retrieval-how-to-pipeline?tabs=search-perms

def knowledge_base_search(query_text, account_id):
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

    agent_client = KnowledgeAgentRetrievalClient(endpoint=search_endpoint, agent_name=search_agent_name, credential=AzureKeyCredential(search_api_key))
    messages.append({
        "role": "user",
        "content": query_text
    })
    print([msg["content"] for msg in messages if msg["role"] != "system"])
    retrieval_result = agent_client.retrieve(
        retrieval_request=KnowledgeAgentRetrievalRequest(
            messages=[KnowledgeAgentMessage(role=msg["role"], content=[KnowledgeAgentMessageTextContent(text=msg["content"])]) for msg in messages if msg["role"] != "system"],
            target_index_params=[KnowledgeAgentIndexParams(index_name=index_name, reranker_threshold=2.5, include_reference_source_data=True,filter_add_on=f"startwith(id,{account_id})")] # add filter_add_on here using the syntax at https://learn.microsoft.com/en-us/azure/search/search-query-odata-filter
        )
    )
    print(f"Result={retrieval_result.response[0].content[0].text}")
    return retrieval_result.response[0].content[0].text




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
    with agents_client:
        agent = agents_client.get_agent("asst_v2Hj4CJ5wEW2gqGwG44YtbD4") # fn_agent_name
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
        #print(f"Created thread, ID: {thread.id}")

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
        
