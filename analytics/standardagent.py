from azure.ai.agents import AgentsClient, Agent, MessageRole
from azure.identity import DefaultAzureCredential
import os

# Set up the Foundry endpoint and model deployment name
PROJECT_ENDPOINT = os.environ["PROJECT_ENDPOINT"]  # e.g., "https://<resource>.services.ai.azure.com/api/projects/<project>"
MODEL_DEPLOYMENT_NAME = os.environ["MODEL_DEPLOYMENT_NAME"]

# Initialize the Agents client
client = AgentsClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential())

# 1. Create a standard agent that tracks the last five user messages
instructions = (
    "You are a helpful assistant. Always keep track of the last five messages from the user. "
    "Summarize or reference them as needed in your responses."
)
standard_agent = client.create_agent(
    model=MODEL_DEPLOYMENT_NAME,
    name="standard-agent",
    instructions=instructions
)

# 2. Create two specialized agents
how_many_agent = client.create_agent(
    model=MODEL_DEPLOYMENT_NAME,
    name="how-many-agent",
    instructions="You answer only questions that begin with 'How many'."
)

how_far_agent = client.create_agent(
    model=MODEL_DEPLOYMENT_NAME,
    name="how-far-agent",
    instructions="You answer only questions that begin with 'How Far'."
)

# 3. Routing logic (example function)
def route_query(query):
    q = query.strip().lower()
    if q.startswith("how many"):
        return how_many_agent.id
    elif q.startswith("how far"):
        return how_far_agent.id
    else:
        return standard_agent.id

# 4. Example: handle a user query
def handle_user_query(user_query, thread_id=None):
    agent_id = route_query(user_query)
    # Create a new thread if not provided
    if not thread_id:
        thread = client.create_thread()
        thread_id = thread.id
    # Add user message
    client.create_message(thread_id, MessageRole.User, user_query)
    # Run the agent
    run = client.create_run(thread_id, agent_id)
    # Poll for completion (simplified)
    while run.status in ("queued", "in_progress", "requires_action"):
        run = client.get_run(thread_id, run.id)
    # Get messages
    messages = client.get_messages(thread_id)
    return [m.content for m in messages if m.role == MessageRole.Assistant][-1]

# Example usage:
response = handle_user_query("How many stars are in the galaxy?")
print(response)
response = handle_user_query("How Far is the moon?")
print(response)
response = handle_user_query("Tell me a joke.")
print(response)
