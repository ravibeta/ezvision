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
import torch
from urllib.parse import urlparse
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
agent_name = os.getenv("AZURE_VEHICLE_COUNT_AGENT_NAME", "vehicle-agent-in-a-team")
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

def read_image_from_blob(sas_url):
    """Reads an image from Azure Blob Storage using its SAS URL."""
    response = requests.get(sas_url)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    else:
        # raise Exception(f"Failed to fetch image. Status code: {response.status_code}")
        return None
        
def detect_vehicles(frame):
    results = model(frame)
    # Keep only 'car', 'truck', 'bus', 'motorcycle' detections
    vehicle_labels = ['car', 'truck', 'bus', 'motorcycle']
    detections = results.pandas().xyxy[0]
    vehicles = detections[detections['name'].isin(vehicle_labels)]
    return vehicles

def get_image_output_url(scene_uri):
    # Parse the original video URL to get account, container, and path
    parsed = urlparse(scene_uri)
    path_parts = parsed.path.split('/')
    container = path_parts[1]
    blob_name = path_parts[-1].split('.')[0]
    blob_path = '/'.join(path_parts[2:])
    # Remove the file name from the blob path
    blob_dir = '/'.join(blob_path.split('/')[:-1])
    if blob_dir == "" or blob_dir == None:
        blob_dir = "output"
    # Create image path
    image_path = f"{blob_dir}/analyzed/{blob_name}withvehicles.jpg"
    # Rebuild the base URL (without SAS token)
    base_url = f"{parsed.scheme}://{parsed.netloc}/{container}/{image_path}"
    # Add the SAS token if present
    sas_token = parsed.query
    if sas_token:
        image_url = f"{base_url}?{sas_token}"
    else:
        image_url = base_url
    return image_url
    
def detect_vehicles_from_uri(scene_uri: Optional[str] = None) -> str:
    if not scene_uri:
        return None
    frame = read_image_from_blob(scene_uri)
    if not frame:
        return None
    vehicles = detect_vehicles(frame)
    print(vehicles)
    for _, v in vehicles.iterrows():
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    image_uri = get_image_output_url(scene_uri)
    image_blob_client = BlobClient.from_blob_url(image_url)
    image_blob_client.upload_blob(image_bytes, overwrite=True) 
    return image_uri
    
image_user_functions: Set[Callable[..., Any]] = {
    detect_vehicles_from_uri
}

# Initialize function tool with user functions
functions = FunctionTool(functions=image_user_functions)
instructions = "You are an assistant that answers the question how many vehicles were found in an image when the image is given by an image URI. You evaluate a function to do this by passing their uri to the function and respond with the count."
query_text = f"How many vehicles are found in the image given by its image URI {scene_uri}?"
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
