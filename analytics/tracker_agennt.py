import os
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectedAgentTool, MessageRole
from azure.ai.agents.models import CodeInterpreterTool
from azure.identity import DefaultAzureCredential

# Initialize Azure AI Project client
project_client = AIProjectClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
    api_version="latest"
)

def create_connected_agent(name, instructions, tools):
    return project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name=name,
        instructions=instructions,
        tools=tools
    )

# 1. Bing Search Agent
bing_search_tool = {"type": "bing_search"}  # Replace with actual BingSearchTool if available
bing_agent = create_connected_agent(
    name="bing_search_agent",
    instructions="Use Bing Search to ground answers with current web information.",
    tools=[bing_search_tool]
)

# 2. File Search Agent
file_search_tool = {"type": "file_search"}  # Replace with actual FileSearchTool if available
file_search_agent = create_connected_agent(
    name="file_search_agent",
    instructions="Search files and documents to find relevant information.",
    tools=[file_search_tool]
)

# 3. AI Search Agent
ai_search_tool = {"type": "ai_search"}  # Replace with actual AI Search tool if available
ai_search_agent = create_connected_agent(
    name="ai_search_agent",
    instructions="Perform AI-powered semantic search over data sources.",
    tools=[ai_search_tool]
)

# 4. Function Calling Agent
function_calling_tool = {"type": "function_calling"}  # Replace with actual FunctionCallingTool if available
function_calling_agent = create_connected_agent(
    name="function_calling_agent",
    instructions="Call external functions and APIs to fulfill user requests.",
    tools=[function_calling_tool]
)

# 5. Code Interpreter Agent
code_interpreter_tool = CodeInterpreterTool()
code_interpreter_agent = create_connected_agent(
    name="code_interpreter_agent",
    instructions="Interpret and execute code to solve problems and analyze data.",
    tools=code_interpreter_tool.definitions
)

# Register connected agents with the main agent
connected_agents_tools = []
for agent in [bing_agent, file_search_agent, ai_search_agent, function_calling_agent, code_interpreter_agent]:
    connected_agents_tools.append(
        ConnectedAgentTool(
            id=agent.id,
            name=agent.name.replace(" ", "_"),
            description=f"Handles queries for {agent.name.replace('_', ' ')}"
        )
    )

# Main agent that delegates to sub-agents and tracks conversation history
main_agent = project_client.agents.create_agent(
    model=os.environ["MODEL_DEPLOYMENT_NAME"],
    name="main_delegator_agent",
    instructions=(
        "You are the main agent. Keep track of all conversation history with the user. "
        "Delegate each query to the appropriate connected agent based on its capability."
    ),
    tools=connected_agents_tools
)

print(f"Main agent created: {main_agent.id}")
