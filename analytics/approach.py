#! /usr/bin/python3
## agentic retrieval
from azure.search.documents.agent.aio import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import (
    KnowledgeAgentAzureSearchDocReference,
    KnowledgeAgentIndexParams,
    KnowledgeAgentMessage,
    KnowledgeAgentMessageTextContent,
    KnowledgeAgentRetrievalRequest,
    KnowledgeAgentRetrievalResponse,
    KnowledgeAgentSearchActivityRecord,
)
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import (
    QueryCaptionResult,
    QueryType,
    VectorizedQuery,
    VectorQuery,
)
from openai import AsyncOpenAI, AsyncStream
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionReasoningEffort,
    ChatCompletionToolParam,
)
    async def run_agentic_retrieval(
        messages: list[ChatCompletionMessageParam],
        agent_client: KnowledgeAgentRetrievalClient,
        search_index_name: str,
        top: Optional[int] = None,
        filter_add_on: Optional[str] = None,
        minimum_reranker_score: Optional[float] = None,
        max_docs_for_reranker: Optional[int] = None,
        results_merge_strategy: Optional[str] = None,
    ) -> tuple[KnowledgeAgentRetrievalResponse, list[Document]]:
        # STEP 1: Invoke agentic retrieval
        response = await agent_client.retrieve(
            retrieval_request=KnowledgeAgentRetrievalRequest(
                messages=[
                    KnowledgeAgentMessage(
                        role=str(msg["role"]), content=[KnowledgeAgentMessageTextContent(text=str(msg["content"]))]
                    )
                    for msg in messages
                    if msg["role"] != "system"
                ],
                target_index_params=[
                    KnowledgeAgentIndexParams(
                        index_name=search_index_name,
                        reranker_threshold=minimum_reranker_score,
                        max_docs_for_reranker=max_docs_for_reranker,
                        filter_add_on=filter_add_on,
                        include_reference_source_data=True,
                    )
                ],
            )
        )

        # STEP 2: Generate a contextual and content specific answer using the search results and chat history
        activities = response.activity
        activity_mapping = (
            {
                activity.id: activity.query.search if activity.query else ""
                for activity in activities
                if isinstance(activity, KnowledgeAgentSearchActivityRecord)
            }
            if activities
            else {}
        )

        results = []
        if response and response.references:
            if results_merge_strategy == "interleaved":
                # Use interleaved reference order
                references = sorted(response.references, key=lambda reference: int(reference.id))
            else:
                # Default to descending strategy
                references = response.references
            for reference in references:
                if isinstance(reference, KnowledgeAgentAzureSearchDocReference) and reference.source_data:
                    results.append(
                        Document(
                            id=reference.doc_key,
                            content=reference.source_data["content"],
                            sourcepage=reference.source_data["sourcepage"],
                            search_agent_query=activity_mapping[reference.activity_source],
                        )
                    )
                if top and len(results) == top:
                    break

        return response, results
