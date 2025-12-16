import json

from typing import Dict, Any
from fastapi import HTTPException, APIRouter
from fastapi.responses import StreamingResponse
from .state_types import ChatRequest
from .rag_graph import agent_tool_node
from ...globals import global_resources, stream_text_naturally
from .chat_constant import STREAMING_TOKEN_TAG_PREFIX
from .chat_enum import RAGDecision, LangGraphEvent, RAGNode

api_router = APIRouter(
    prefix="/chat"
)

@api_router.post("/threads/{thread_id}/runs/stream")
async def handle_chat_stream(thread_id: str, request: ChatRequest):
    """Handles the streaming chat request by executing the RAG graph."""

    # Use singleton thread ID for simplicity as defined in globals
    thread_id_to_use = global_resources.get("SINGLETON_THREAD_ID")

    # Extract the latest human query
    all_messages = request.input.messages
    latest_human_message = next((msg for msg in reversed(all_messages) if msg.type == "human"), None)
    query_text = next((item.text for item in latest_human_message.content if item.type == "text"), None)

    if not latest_human_message or not query_text:
        raise HTTPException(status_code=400, detail="No valid human query found in messages.")

    # Initialize history store if needed
    history_store = global_resources.get("history_store", {})
    if thread_id_to_use not in history_store:
        history_store[thread_id_to_use] = []

    # Prepare initial state for LangGraph
    initial_state = {
        "query": query_text,
        "enhanced_queries": [],
        "documents": [],
        "answer": "",
        "decision": RAGDecision.GENERATE.value
    }

    # Check RAG prerequisites
    rag_app = global_resources.get("rag_app")
    chroma_ready = global_resources.get("document_collection") is not None

    if not rag_app or not chroma_ready:
        status = "LangGraph compiled but ChromaDB is down." if rag_app else "RAG system not initialized (Compilation failed)."
        raise HTTPException(status_code=503, detail=f"RAG system not initialized. Reason: {status}")

    # Start the async generator function
    response_generator = stream_rag_agent_response(thread_id_to_use, query_text, initial_state)

    # Return the StreamingResponse with the SSE media type
    return StreamingResponse(response_generator, media_type="text/event-stream")

async def stream_rag_agent_response(thread_id: str, query: str, initial_state: Dict[str, Any]):
    """
    Executes the RAG/Agent system using LangGraph's astream_events,
    using Pre-Generation Routing (critique_context_node) to decide the flow.
    """

    rag_app = global_resources.get("rag_app")
    final_answer = ""
    current_rag_answer_buffer = ""
    source_info = "system"
    decision = initial_state['decision']

    yield f"data: {json.dumps({'event': 'start', 'thread_id': thread_id})}\n\n"

    try:
        if decision == RAGDecision.GENERATE.value:
            if not rag_app:
                raise Exception("RAG system not initialized (Compilation failed).")

            is_tool_agent_running = False
            initial_state['answer'] = ""

            async for event in rag_app.astream_events(initial_state, version="v1"):

                # A. Catch LLM TOKEN from 'generate' node
                if event["event"] == LangGraphEvent.ON_CHAIN_STREAM.value and event["name"] == RAGNode.GENERATE.value:
                    # Check duplicated token: Filter for true streaming tokens using the 'STREAMING_TOKEN_TAG_PREFIX' tag prefix
                    is_streaming_token = any(tag.startswith(STREAMING_TOKEN_TAG_PREFIX) for tag in event.get("tags", []))
                    if not is_streaming_token:
                        # This typically catches the redundant 'graph:step:X' event at the end.
                        print(f"Skipping redundant or non-streaming event. Tags: {event.get('tags')}")
                        continue

                    chunk = event["data"]["chunk"]
                    if isinstance(chunk, dict) and "answer" in chunk:
                        token_data = chunk["answer"]
                        if token_data:
                            current_rag_answer_buffer += token_data
                            # Yield token chunk formatted as SSE
                            yield f"data: {json.dumps({'event': 'stream', 'data': {'chunk': token_data}})}\n\n"

                # A'. Process Document Retrieval Metadata
                elif event["event"] == LangGraphEvent.ON_CHAIN_END.value and event["name"] == RAGNode.RETRIEVE.value:
                    doc_count = len(event["data"]["output"].get("documents", []))
                    source_info = f"RAG Documents ({doc_count} chunks)"
                    yield f"data: {json.dumps({'event': 'metadata', 'step': f'Retrieved {doc_count} documents', 'status': 'Critiquing Context'})}\n\n"

                # B. Pre-Generation Router Logic (critique_context_node)
                elif event["event"] == LangGraphEvent.ON_DECIDE.value and event["name"] == RAGNode.CRITIQUE_CONTEXT.value:
                    route_to = event["data"].get("output")

                    if route_to == RAGDecision.TOOL.value:
                        print("Router confirmed RAG failure (Context Irrelevant), running tool.")
                        current_rag_answer_buffer = ""
                        yield f"data: {json.dumps({'event': 'metadata', 'step': 'Context Irrelevant, Switching to Web Search', 'status': 'Running'})}\n\n"

                    elif route_to == RAGDecision.GENERATE.value:
                        print("Router confirmed RAG success (Context Relevant), generating answer.")
                        yield f"data: {json.dumps({'event': 'metadata', 'step': 'Context Relevant, Generating RAG Answer', 'status': 'Running'})}\n\n"


                # C. Start Tool Agent (RAG Fallback)
                elif event["event"] == LangGraphEvent.ON_CHAIN_START.value and event["name"] == RAGNode.TOOL_AGENT.value:
                    is_tool_agent_running = True
                    yield f"data: {json.dumps({'event': 'metadata', 'step': 'Executing Tool Search', 'status': 'Running'})}\n\n"

                # D. Process Tool Agent Result (on completion)
                elif event["event"] == LangGraphEvent.ON_CHAIN_END.value and event["name"] == RAGNode.TOOL_AGENT.value:
                    output = event["data"].get("output")

                    if isinstance(output, dict) and 'answer' in output:
                        tool_result = output['answer']

                        if tool_result:
                            final_answer = tool_result
                            source_info = "Tool Agent (search_web) fallback"

                            async for char in stream_text_naturally(final_answer):
                                yield f"data: {json.dumps({'event': 'stream', 'data': {'chunk': char}})}\n\n"

                    is_tool_agent_running = False

                elif event["event"] == LangGraphEvent.ON_GRAPH_END.value:
                    pass


        # --- Tool Agent Workflow (Direct decision == "tool") ---
        elif decision == RAGDecision.TOOL.value:
            yield f"data: {json.dumps({'event': 'metadata', 'step': 'Tool Calling', 'status': 'Executing search_web'})}\n\n"

            tool_result = agent_tool_node(initial_state)
            final_answer = tool_result['answer']
            source_info = "Tool Agent (search_web)"

            if final_answer:
                async for char in stream_text_naturally(final_answer):
                    yield f"data: {json.dumps({'event': 'stream', 'data': {'chunk': char}})}\n\n"

        if decision == RAGDecision.GENERATE.value and not is_tool_agent_running:
            final_answer = current_rag_answer_buffer

        final_state_update = {
            "answer": final_answer,
            "query": query,
            "enhanced_queries": [],
            "decision": initial_state.get("decision", RAGDecision.GENERATE.value)
        }

        yield f"data: {json.dumps({'event': 'end', 'result': 'success', 'source': source_info, 'state': final_state_update})}\n\n"

    except Exception as e:
        error_msg = f"Execution failed: {str(e)}"
        print(f"Streaming run error: {error_msg}")
        yield f"data: {json.dumps({'event': 'error', 'message': error_msg})}\n\n"
        yield f"data: {json.dumps({'event': 'end', 'result': 'failure'})}\n\n"