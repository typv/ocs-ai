from enum import Enum
from typing import List

class RAGDecision(Enum):
    GENERATE = "generate"
    TOOL = "tool"

    @classmethod
    def to_literals(cls) -> List[str]:
        return [member.value for member in cls]


class RAGNode(Enum):
    ENHANCE_QUERY = "enhance_query"
    RETRIEVE = "retrieve"
    CRITIQUE_CONTEXT = "critique_context"
    GENERATE = "generate"
    TOOL_AGENT = "tool_agent"

class LangGraphEvent(Enum):
    """Event names emitted by LangGraph's astream_events."""
    ON_CHAIN_STREAM = "on_chain_stream"
    ON_CHAIN_END = "on_chain_end"
    ON_CHAIN_START = "on_chain_start"
    ON_DECIDE = "on_decide"
    ON_GRAPH_END = "on_graph_end"

class SearchStrategy(Enum):
    """Search strategies for document retrieval"""
    SIMPLE = "simple"  # Simple vector similarity
    RERANK = "rerank"  # Vector similarity with reranking
    HYBRID = "hybrid"  # Combine multiple search methods
    FUSION = "fusion"  # RAG-Fusion style