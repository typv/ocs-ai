from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field

# LangGraph State Definition
class RAGState(TypedDict):
    """State for the RAG/Agent process."""
    query: str
    enhanced_queries: List[str]
    documents: List[str]
    answer: str
    decision: str

# FastAPI/Messaging Models
class ContentText(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    id: str
    type: Literal["human", "ai", "system"]
    content: List[ContentText]

class InputMessages(BaseModel):
    messages: List[Message]

class ChatRequest(BaseModel):
    input: InputMessages
    # stream_mode: List[str] = Field(default=["values"])
    # stream_subgraphs: bool = Field(default=False)
    # stream_resumable: bool = Field(default=False)
    # assistant_id: str = Field(default="")
    # on_disconnect: Literal["continue", "stop"] = Field(default="continue")