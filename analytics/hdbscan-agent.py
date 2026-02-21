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
from typing import Any, Callable, Set, Dict, List, Optional
import os, time, sys
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    FunctionTool,
    ListSortOrder,
    RequiredFunctionToolCall,
    SubmitToolOutputsAction,
    ToolOutput,
)
from user_functions import fetch_weather, user_functions
sys.path.insert(0, os.path.abspath("."))
load_dotenv(override=True) 
project_endpoint = os.environ["AZURE_PROJECT_ENDPOINT"] 
project_api_key = os.environ["AZURE_PROJECT_API_KEY"] 
agent_model = os.getenv("AZURE_AGENT_MODEL", "gpt-4o-mini") 
agent_name = os.getenv("AZURE_FN_AGENT_NAME", "fn-agent-in-a-team")
api_version = "2025-05-01-Preview" 
agent_max_output_tokens=10000 
object_uri = os.getenv("AZURE_RED_CAR_2_SAS_URL").strip('"')
scene_uri = os.getenv("AZURE_QUERY_SAS_URI").strip('"') 
from azure.ai.projects import AIProjectClient 
project_client = AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential()) 
agents_client = AgentsClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(),
)

def agentic_retrieval(pattern_uri: Optional[str] = None, content_uri: Optional[str] = None) -> str:
    import dbscan
    if not pattern_uri:
        print(f"No pattern uri for object to be detected found.")
        pattern_uri = object_uri
    if not content_uri:
        print(f"No content uri for scene to detect objects found.")
        content_uri = scene_uri
    count = dbscan.count_multiple_matches(scene_uri, object_uri)
    return f"{count}"

image_user_functions: Set[Callable[..., Any]] = {
    agentic_retrieval
}

# Initialize function tool with user functions
functions = FunctionTool(functions=image_user_functions)
# instructions = "You are a helpful agent."
# query_text = "Hello, what is the weather in New York?"
instructions = "You are an assistant that answers the question how many objects were found in an image when both are given by their image URI. You evaluate a function to do this by passing their uri to the function and respond with the count."
query_text = f"How many objects given by its image URI {object_uri} are found in the image given by its image URI {scene_uri}?"
with agents_client:
    # Create an agent and run user's request with function calls
    # agent = agents_client.get_agent(agent_id="asst_qyMFcz1BnU0BS0QUmhxAAyFk")
    # """
    agent = agents_client.create_agent(
        model=agent_model,
        name=agent_name,
        instructions=instructions,
        tools=functions.definitions,
        tool_resources=functions.resources,
        top_p=1
    )
    # """
    print(f"Created agent, ID: {agent.id}")

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
                if isinstance(tool_call, RequiredFunctionToolCall):
                    print("Is an instance of RequiredFunctionToolCall")
                    try:
                        print(f"Executing tool call: {tool_call}")
                        output = functions.execute(tool_call)
                        print(output)
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
    agents_client.delete_agent(agent.id)
    print("Deleted agent")

    # Fetch and log all messages
    messages = agents_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
    for msg in messages:
        if msg.text_messages:
            last_text = msg.text_messages[-1]
            print(f"{msg.role}: {last_text.text.value}")
            
# Output:
"""
Created agent, ID: asst_qyMFcz1BnU0BS0QUmhxAAyFk
Created thread, ID: thread_qlS6b0Lo2zxEfk5wvdbmipOg
Created message, ID: msg_Ye6koVrBF7O7RkrPGYmFokSK
Created run, ID: run_AOY65vGkbyeswPvvhURRQ7c3
Is an instance of RequiredFunctionToolCall
Executing tool call: {'id': 'call_h3UG84BhilrrMfsTRDfyNoPK', 'type': 'function', 'function': {'name': 'fetch_weather', 'arguments': '{"location":"New York"}'}}
{"weather": "Sunny, 25\u00b0C"}
Tool outputs: [{'tool_call_id': 'call_h3UG84BhilrrMfsTRDfyNoPK', 'output': '{"weather": "Sunny, 25\\u00b0C"}'}]
Current run status: RunStatus.COMPLETED
MessageRole.USER: Hello, what is the weather in New York?
MessageRole.AGENT: The weather in New York is sunny with a temperature of 25°C.\
"""

"""
Created agent, ID: asst_GUJsx765VDAhAvJ0oR0fTpyI
Created thread, ID: thread_xZ62v8T9LeIwQiHHr2IINi1G
Created message, ID: msg_UMUmOKxgNNpYn5SUuqqjw38F
Created run, ID: run_BXR4UlGDGCuA3dLtppTpgLoV
Current run status: RunStatus.IN_PROGRESS
Current run status: RunStatus.IN_PROGRESS
Current run status: RunStatus.IN_PROGRESS
Current run status: RunStatus.IN_PROGRESS
Is an instance of RequiredFunctionToolCall
Executing tool call: {'id': 'call_hfneTrfgMLgUA5Ue0zO043dI', 'type': 'function', 'function': {'name': 'agentic_retrieval', 'arguments': '{"pattern_uri": "https://saravinoteblogs.blob.core.windows.net/playground/vision/query/RedCar2.jpg?sp=racwdme&st=2025-07-13T23:03:29Z&se=2025-07-14T07:18:29Z&spr=https&sv=2024-11-04&sr=b&sig=g26%2F04hten1Bte14mMepIMWV9YdzOChCiWIkkpy73Zw%3D", "content_uri": "https://saravinoteblogs.blob.core.windows.net/playground/vision/query/RedCar4.jpg?sp=racwdyme&st=2025-07-13T23:05:31Z&se=2025-07-13T07:20:31Z&spr=https&sv=2024-11-04&sr=b&sig=%2BYhCfPeq8nOgIPzQk5%2BVkxI2mIE07Z2Hnkd%2BU85uuJo%3D"}'}}
len of labels=24 and labels=[ 1 -1 -1  1 -1  1  0 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0 -1  1 -1]
Estimated object instances: 5
len of labels=24 and labels=[ 1 -1 -1  1 -1  1  0 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0 -1  1 -1]
5
Is an instance of RequiredFunctionToolCall
Executing tool call: {'id': 'call_u719pByAgb6Gi2c9z87yVicY', 'type': 'function', 'function': {'name': 'agentic_retrieval', 'arguments': '{"pattern_uri": "https://saravinoteblogs.blob.core.windows.net/playground/vision/query/RedCar4.jpg?sp=racwdyme&st=2025-07-13T23:05:31Z&se=2025-07-13T07:20:31Z&spr=https&sv=2024-11-04&sr=b&sig=%2BYhCfPeq8nOgIPzQk5%2BVkxI2mIE07Z2Hnkd%2BU85uuJo%3D", "content_uri": "https://saravinoteblogs.blob.core.windows.net/playground/vision/query/RedCar2.jpg?sp=racwdme&st=2025-07-13T23:03:29Z&se=2025-07-13T07:18:29Z&spr=https&sv=2024-11-04&sr=b&sig=g26%2F04hten1Bte14mMepIMWV9YdzOChCiWIkkpy73Zw%3D"}'}}
len of labels=24 and labels=[ 1 -1 -1  1 -1  1  0 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0 -1  1 -1]
5
Tool outputs: [{'tool_call_id': 'call_hfneTrfgMLgUA5Ue0zO043dI', 'output': '5'}, {'tool_call_id': 'call_u719pByAgb6Gi2c9z87yVicY', 'output': '5'}]
Current run status: RunStatus.COMPLETED
Run completed with status: RunStatus.COMPLETED and details {'id': 'run_BXR4UlGDGCuA3dLtppTpgLoV', 'object': 'thread.run', 'created_at': 1752476935, 'assistant_id': 'asst_GUJsx765VDAhAvJ0oR0fTpyI', 'thread_id': 'thread_xZ62v8T9LeIwQiHHr2IINi1G', 'status': 'completed', 'started_at': 1752476964, 'expires_at': None, 'cancelled_at': None, 'failed_at': None, 'completed_at': 1752476965, 'required_action': None, 'last_error': None, 'model': 'gpt-4o-mini', 'instructions': 'You are an assistant that answers the question how many objects were found in an image when both are given by their image URI. You evaluate a function to do this by passing their uri to the function and respond with the count.', 'tools': [{'type': 'function', 'function': {'name': 'agentic_retrieval', 'description': 'No description', 'parameters': {'type': 'object', 'properties': {'pattern_uri': {'type': ['string', 'null'], 'description': 'No description'}, 'content_uri': {'type': ['string', 'null'], 'description': 'No description'}}, 'required': []}, 'strict': False}}], 'tool_resources': {}, 'metadata': {}, 'temperature': 1.0, 'top_p': 1.0, 'max_completion_tokens': None, 'max_prompt_tokens': None, 'truncation_strategy': {'type': 'auto', 'last_messages': None}, 'incomplete_details': None, 'usage': {'prompt_tokens': 1734, 'completion_tokens': 524, 'total_tokens': 2258, 'prompt_token_details': {'cached_tokens': 0}}, 'response_format': 'auto', 'tool_choice': 'auto', 'parallel_tool_calls': True}
MessageRole.USER: How many objects given by its image URI https://saravinoteblogs.blob.core.windows.net/playground/vision/query/RedCar2.jpg?sp=racwdme&st=2025-07-13T23:03:29Z&se=2025-07-13T07:18:29Z&spr=https&sv=2024-11-04&sr=b&sig=g26%2F04hten1Bte14mMepIMWV9YdzOChCiWIkkpy73Zw%3D are found in the image given by its image URI https://saravinoteblogs.blob.core.windows.net/playground/vision/query/RedCar4.jpg?sp=racwdyme&st=2025-07-13T23:05:31Z&se=2025-07-13T07:20:31Z&spr=https&sv=2024-11-04&sr=b&sig=%2BYhCfPeq8nOgIPzQk5%2BVkxI2mIE07Z2Hnkd%2BU85uuJo%3D?
MessageRole.AGENT: The count of objects found in the images is 5 for both URIs provided.
"""