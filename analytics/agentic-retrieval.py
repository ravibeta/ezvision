#!/usr/bin/python
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
agent_name = os.getenv("AZURE_SEARCH_AGENT_NAME", "agentic-retrieval-drone-images")

from azure.search.documents.indexes.models import KnowledgeAgent, KnowledgeAgentAzureOpenAIModel, KnowledgeAgentTargetIndex, KnowledgeAgentRequestLimits, AzureOpenAIVectorizerParameters
from azure.search.documents.indexes import SearchIndexClient

agent = KnowledgeAgent(
    name=agent_name,
    models=[
        KnowledgeAgentAzureOpenAIModel(
            azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                resource_url=azure_openai_endpoint,
                deployment_name=azure_openai_gpt_deployment,
                model_name=azure_openai_gpt_model
            )
        )
    ],
    target_indexes=[
        KnowledgeAgentTargetIndex(
            index_name=index_name,
            default_reranker_threshold=2.5
        )
    ],
    request_limits=KnowledgeAgentRequestLimits(
        max_output_size=10000
    )
)

index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
index_client.create_or_update_agent(agent)
print(f"Knowledge agent '{agent_name}' created or updated successfully")

from azure.ai.projects import AIProjectClient

project_client = AIProjectClient(endpoint=project_endpoint, credential=credential)

list(project_client.agents.list_agents())
instructions = """
A Q&A agent that can answer questions about the drone images stored in Azure AI Search.
Sources have a JSON description and vector format with a ref_id that must be cited in the answer.
If you do not have the answer, respond with "I don't know".
"""
agent = project_client.agents.create_agent(
    model=agent_model,
    name=agent_name,
    instructions=instructions
)

print(f"AI agent '{agent_name}' created or updated successfully")
from azure.ai.agents.models import FunctionTool, ToolSet, ListSortOrder

from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import KnowledgeAgentRetrievalRequest, KnowledgeAgentMessage, KnowledgeAgentMessageTextContent, KnowledgeAgentIndexParams

agent_client = KnowledgeAgentRetrievalClient(endpoint=search_endpoint, agent_name=agent_name, credential=credential)

thread = project_client.agents.threads.create()
retrieval_results = {}

def agentic_retrieval() -> str:
    """
        Searches drone images about objects detected and their facts.
        The returned string is in a JSON format that contains the reference id.
        Be sure to use the same format in your agent's response
        You must refer to references by id number
    """
    # Take the last 5 messages in the conversation
    messages = project_client.agents.messages.list(thread.id, limit=5, order=ListSortOrder.DESCENDING)
    # Reverse the order so the most recent message is last
    messages = list(messages)
    messages.reverse()
    retrieval_result = agent_client.retrieve(
        retrieval_request=KnowledgeAgentRetrievalRequest(
            messages=[KnowledgeAgentMessage(role=msg["role"], content=[KnowledgeAgentMessageTextContent(text=msg.content[0].text)]) for msg in messages if msg["role"] != "system"],
            target_index_params=[KnowledgeAgentIndexParams(index_name=index_name, reranker_threshold=2.5)]
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
project_client.agents.enable_auto_function_calls(toolset)

from azure.ai.agents.models import AgentsNamedToolChoice, AgentsNamedToolChoiceType, FunctionName

message = project_client.agents.messages.create(
    thread_id=thread.id,
    role="user",
    content="""
        How many parking lots are empty when compared to all the parking lots?
        How many red cars could be found as parked?
    """
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