import re
import json
import time
import logging

from langgraph.graph import StateGraph, END
from typing import Dict, Any, Literal, Optional, AsyncGenerator, List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

# Import logic from other modules
from .state_types import RAGState
from .llm_tools import search_web
from ...globals import global_resources, get_db_collection
from ...config.app_config import settings
from .chat_constant import DEFAULT_RESPONSE
from .chat_enum import RAGDecision, RAGNode, SearchStrategy
from .document_searcher import DocumentSearcher

logger = logging.getLogger(__name__)

# --- LangGraph Nodes ---

def enhance_query_node(state: RAGState, config: Optional[RunnableConfig]) -> RAGState:
    """Uses LLM to enhance the original query (Node 1 in RAG workflow)."""
    print("--- RAG: ENHANCING QUERY ---")
    llm = global_resources.get("gemini_llm")

    query = state["query"]
    enhanced_queries = enhance_query(llm, query)
    return {"enhanced_queries": enhanced_queries}


def retrieve_node(state: RAGState, config: Optional[RunnableConfig]) -> RAGState:
    """Performs RAG-Fusion retrieval based on all queries (Node 2 in RAG workflow)."""
    print("--- RAG: RETRIEVING DOCUMENTS ---")

    query = state["query"]
    enhanced_queries = state["enhanced_queries"]
    all_queries = [query] + enhanced_queries

    # Call the enhanced search_documents function
    docs, metadatas, scores, search_stats = search_documents(queries=all_queries)

    return {
        "documents": docs,
        "document_metadatas": metadatas,
        "retrieval_scores": scores,
        "retrieval_stats": search_stats,
        "query": query,
        "enhanced_queries": all_queries
    }


async def critique_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    [ASYNC NODE] Uses a fast LLM to decide if the retrieved documents are relevant
    enough to answer the query, or if a web search is needed.
    """
    print("--- ROUTER: CRITIQUING CONTEXT RELEVANCE ---")

    llm = global_resources.get("gemini_llm")

    query = state["query"]
    docs = state["documents"]
    context_text = "\n\n".join(docs)

    if not docs:
        print(f"Decision: No documents found. Routing to {RAGDecision.TOOL.name}.")
        return {"decision": RAGDecision.TOOL.value}

    critique_prompt = f"""
    Bạn là một hệ thống đánh giá. 
    Dưới đây là câu hỏi của người dùng và các đoạn văn bản trích xuất từ tài liệu nội bộ.

    --- CÂU HỎI ---
    {query}

    --- NGỮ CẢNH TRÍCH XUẤT ---
    {context_text}

    Bạn có thể trả lời câu hỏi của người dùng MỘT CÁCH ĐẦY ĐỦ chỉ dựa trên ngữ cảnh này không?
    Chỉ trả lời "YES" nếu ngữ cảnh cung cấp thông tin trực tiếp để trả lời.
    Chỉ trả lời "NO" nếu ngữ cảnh không liên quan hoặc thiếu thông tin quan trọng.
    """

    try:
        response = await llm.ainvoke(critique_prompt)
        decision_text = response.content.strip().upper()

        if "YES" in decision_text:
            print(f"Decision: Context is RELEVANT. Routing to {RAGDecision.GENERATE.name}.")
            return {"decision": RAGDecision.GENERATE.value}
        else:
            print(f"Decision: Context is IRRELEVANT/INSUFFICIENT. Routing to {RAGDecision.TOOL.name}.")
            return {"decision": RAGDecision.TOOL.value}

    except Exception as e:
        print(f"Critique failed: {e}. Defaulting to GENERATE.")
        return {"decision": RAGDecision.GENERATE.value}


async def generate_node(state: Dict[str, Any], config: Optional[RunnableConfig]) -> AsyncGenerator[
    Dict[str, Any], None]:
    """
    [ASYNC GENERATOR] Generates and streams the RAG answer (assuming context is relevant).
    (Node 4 - Generation)
    """
    print("--- RAG: STREAMING ANSWER (Context Pre-Checked) ---")
    llm = global_resources.get("gemini_llm")
    query = state["query"]
    docs = state["documents"]

    context_text = "\n\n".join(docs)

    prompt_content = f"""
    Bạn là một trợ lý AI hữu ích. Dưới đây là các đoạn văn bản được trích xuất từ tài liệu PDF mà người dùng đã tải lên.

    --- NGỮ CẢNH (CONTEXT) ---
    {context_text}
    --- END CONTEXT ---

    Câu hỏi của người dùng: {query}

    Vui lòng trả lời CHỈ dựa trên nội dung trong ngữ cảnh ở trên. 
    Trả lời một cách tự nhiên và trực tiếp, KHÔNG được bắt đầu bằng các cụm từ như "Theo ngữ cảnh", "Dựa trên tài liệu", hay "Theo thông tin được cung cấp".
    Nếu cuối cùng không có kết quả, hãy trả lời: {DEFAULT_RESPONSE}
    """

    try:
        response_stream = llm.astream([HumanMessage(content=prompt_content)])

        async for token_chunk in response_stream:
            token = token_chunk.content
            if token:
                yield {"answer": token}
    except Exception as e:
        print(f"Error during LLM streaming: {e}")
        yield {"answer": "An error occurred during response generation."}


def agent_tool_node(state: RAGState) -> RAGState:
    print("--- TOOL: RUNNING AGENT FOR WEB SEARCH ---")
    query = state["query"]
    llm = global_resources.get("gemini_llm")
    existing_enhanced_queries = state.get("enhanced_queries", [])
    decision = state.get("decision", RAGDecision.TOOL.value)

    tools = [search_web]

    system_prompt = (
        "Bạn là một trợ lý chuyên dụng, chỉ sử dụng công cụ tìm kiếm trên web (search_web) để trả lời các câu hỏi của người dùng. "
        "QUY TẮC BẮT BUỘC: Bạn PHẢI chỉ dựa vào thông tin được cung cấp bởi công cụ tìm kiếm web. "
        "NẾU kết quả tìm kiếm trên web (Search Results) không cung cấp đủ thông tin để trả lời câu hỏi của người dùng, "
        "HOẶC kết quả tìm kiếm không liên quan trực tiếp, "
        f"hãy trả lời NGAY LẬP TỨC bằng câu sau: '{DEFAULT_RESPONSE}'"
        "Nếu có thông tin, hãy tóm tắt kết quả tìm kiếm một cách rõ ràng và trực tiếp. "
        "Khi trả lời, TUYỆT ĐỐI KHÔNG sử dụng bất kỳ ký tự định dạng Markdown nào, bao gồm dấu hoa thị kép (**) để in đậm, dấu gạch dưới (__) để in nghiêng, hoặc dấu gạch ngang (-) cho danh sách không có thứ tự. Chỉ sử dụng danh sách được đánh số hoặc văn bản thuần túy (plain text)"
        "Khi tạo danh sách được đánh số, hãy đảm bảo mỗi mục đều được đánh số liên tục và không có dấu xuống dòng thừa giữa số và nội dung."
        "KHÔNG ĐƯỢC sử dụng kiến thức nội bộ để bác bỏ kết quả tìm kiếm mới nhất."
    )

    agent_executor = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    messages = [HumanMessage(content=query)]

    try:
        agent_response = agent_executor.invoke({"messages": messages})
        answer = extract_final_answer(agent_response)

        if not answer:
            answer = DEFAULT_RESPONSE

        return {
            "query": query,
            "answer": answer,
            "documents": [],
            "enhanced_queries": existing_enhanced_queries,
            "decision": decision
        }

    except Exception as e:
        print(f"Tool Agent execution error during invoke: {e}")
        return {
            "query": query,
            "answer": "Đã xảy ra lỗi trong quá trình tìm kiếm thông tin trên web.",
            "documents": [],
            "enhanced_queries": existing_enhanced_queries,
            "decision": decision
        }

# --- LangGraph Routers ---
def route_context(state: Dict[str, Any]) -> Literal["generate", "tool"]:
    """
    Router: Reads the decision set by critique_context_node (tool or generate).
    """
    if not settings.GOOGLE_SEARCH.ENABLED:
        print("--- ROUTER (Context): Fallback DISABLED. Routing to GENERATE ---")
        return RAGDecision.GENERATE.value

    decision = state.get("decision", RAGDecision.GENERATE.value)

    valid_decisions = RAGDecision.to_literals()
    if decision not in valid_decisions:
        return RAGDecision.GENERATE.value

    print(f"--- ROUTER (Context): Routing to {decision.upper()} ---")
    return decision


# Helpers function
def enhance_query(llm: ChatGoogleGenerativeAI, original_query: str) -> List[str]:
    """Uses LLM to generate multiple different search queries (Query Expansion)."""
    enhancement_prompt = f"""
    Bạn là một công cụ mở rộng truy vấn. Nhiệm vụ của bạn là nhận một câu hỏi gốc từ người dùng
    và tạo ra 3 truy vấn tìm kiếm độc lập, chi tiết và giàu ngữ cảnh để tối ưu hóa khả năng truy xuất
    trong cơ sở dữ liệu vector.

    Xuất kết quả dưới dạng một mảng JSON (List[str]).

    Câu hỏi gốc: {original_query}

    Ví dụ:
    Câu hỏi gốc: 'Kỹ thuật RAG là gì?'
    Đầu ra mong muốn: ['Giải thích về kiến trúc RAG (Retrieval-Augmented Generation)', 'Các thành phần chính của hệ thống RAG', 'Lợi ích của RAG so với fine-tuning']

    Đầu ra JSON (3 truy vấn):
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            enhanced_q_response = llm.invoke(enhancement_prompt)
            raw_content = enhanced_q_response.content.strip()
            match = re.search(r'\[\s*".*"\s*\]', raw_content, re.DOTALL)
            json_string = raw_content
            if match:
                json_string = match.group(0).strip()
            try:
                queries = json.loads(json_string)
                if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                    return queries
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse clean JSON from: {raw_content}")
                pass
        except Exception as e:
            print(f"Error during LLM invocation or processing: {e}")
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
        else:
            print(f"Error enhancing query after {max_retries} attempts.")
            return [original_query]

    return [original_query]


def search_documents(
        queries: List[str],
        k: int = settings.RAG.RETRIEVAL_K,
        strategy: str = "fusion",
        filters: Optional[Dict[str, Any]] = None,
        enable_reranking: bool = True,
        score_threshold: float = 0.3
) -> Tuple[List[str], List[Dict[str, Any]], List[float], Dict[str, Any]]:
    try:
        collection = get_db_collection()

        searcher = DocumentSearcher(
            collection=collection,
            default_k=k,
            default_strategy=SearchStrategy(strategy),
            score_threshold=score_threshold,
            enable_reranking=enable_reranking
        )

        if filters:
            results = searcher.search_with_filters(
                query=queries[0] if queries else "",
                k=k,
                filters=filters
            )
        elif len(queries) > 1:
            results = searcher.search_multiple_queries_fusion(
                queries=queries,
                k=k
            )
        elif len(queries) == 1:
            results = searcher.language_aware_search(
                query=queries[0],
                k=k
            )
        else:
            results = []

        final_results = results[:settings.RAG.CONTEXT_TOP_N]

        documents = [r.document for r in final_results]
        metadatas = [r.metadata for r in final_results]
        scores = [r.score for r in final_results]

        search_stats = searcher.get_search_stats()
        search_stats.update({
            "query_count": len(queries),
            "child_k_requested": k,
            "parents_returned": len(documents),
            "max_context_limit": settings.RAG.CONTEXT_TOP_N,
            "mode": "hierarchical_parent_retrieval",
            "strategy_used": strategy
        })

        return documents, metadatas, scores, search_stats

    except Exception as e:
        logger.error(f"Error in search_documents: {e}")
        return [], [], [], {"error": str(e)}


def extract_final_answer(agent_response):
    messages = agent_response.get("messages", [])
    for msg in reversed(messages):
        msg_type = msg.__class__.__name__
        content = normalize_content(msg.content)

        if msg_type == "AIMessage":
            func_call = msg.additional_kwargs.get("function_call")
            if not func_call:
                cleaned_content = content.strip()

                if cleaned_content:
                    return cleaned_content
    return None


def normalize_content(content):
    if content is None:
        return ""

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("content") or ""
                parts.append(str(txt))
            else:
                parts.append(str(item))
        return "".join(parts)

    if isinstance(content, str):
        return content

    return str(content)


# --- LangGraph Compiler ---

def create_rag_app(gemini_llm: ChatGoogleGenerativeAI):
    """Compiles the LangGraph workflow."""
    workflow = StateGraph(RAGState)

    # Add RAG Nodes
    workflow.add_node(RAGNode.ENHANCE_QUERY.value, enhance_query_node)
    workflow.add_node(RAGNode.RETRIEVE.value, retrieve_node)
    workflow.add_node(RAGNode.CRITIQUE_CONTEXT.value, critique_context_node)
    workflow.add_node(RAGNode.GENERATE.value, generate_node, streamable=True)
    # Add Fallback Nodes
    workflow.add_node(RAGNode.TOOL_AGENT.value, lambda state: agent_tool_node(state))

    # Define RAG Edges
    workflow.set_entry_point(RAGNode.ENHANCE_QUERY.value)
    workflow.add_edge(RAGNode.ENHANCE_QUERY.value, RAGNode.RETRIEVE.value)
    workflow.add_edge(RAGNode.RETRIEVE.value, RAGNode.CRITIQUE_CONTEXT.value)
    workflow.add_conditional_edges(
        RAGNode.CRITIQUE_CONTEXT.value,
        route_context,
        {
            RAGDecision.GENERATE.value: RAGNode.GENERATE.value,
            RAGDecision.TOOL.value: RAGNode.TOOL_AGENT.value,
        },
    )

    workflow.add_edge(RAGNode.GENERATE.value, END)
    workflow.add_edge(RAGNode.TOOL_AGENT.value, END)

    # Compile the graph
    app = workflow.compile()

    # Bind external resources using config
    app = app.with_config({"configurable": {"llm": gemini_llm, "global_resources": global_resources}})

    return app