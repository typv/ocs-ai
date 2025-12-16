from fastapi import FastAPI, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from contextlib import asynccontextmanager
from pathlib import Path
import os

# Import Project Modules
from .globals import global_resources, SINGLETON_THREAD_ID, API_PREFIX
from .database.chroma_db_connector import ChromaDBConnector
from .config.app_config import settings
from .modules.chat.rag_graph import create_rag_app
from .modules.chat.chat_controller import api_router as chat_router
from .modules.indexing.indexing_controller import api_router as embedding_router

# --- Global & Path Setup ---
CURRENT_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE_PATH.parent.parent
TEMP_UPLOAD_DIR = PROJECT_ROOT / "tmp/uploads"
FRONTEND_BUILD_DIR = os.path.join(PROJECT_ROOT, "chat-ui")

# Export resources to global context for modules to access
global_resources["TEMP_UPLOAD_DIR"] = TEMP_UPLOAD_DIR
global_resources["SINGLETON_THREAD_ID"] = SINGLETON_THREAD_ID

# LLM Initialization (before lifespan for agent creation)
gemini_llm = ChatGoogleGenerativeAI(model=settings.GEMINI.MODEL, api_key=settings.GEMINI.API_KEY)
global_resources["gemini_llm"] = gemini_llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ChromaDB connection setup
    connector = ChromaDBConnector(mode="remote", host=settings.CHROMA.HOST, port=settings.CHROMA.PORT)

    if connector.is_connected():
        print("--- INFO: Connected to ChromaDB successfully. ---")
        global_resources["chroma_client"] = connector.client
        global_resources["document_collection"] = connector.get_collection(settings.CHROMA.DEFAULT_COLLECTION_NAME)
    else:
        print("--- ERROR: FAILED to connect to ChromaDB. RAG will not work. ---")
        global_resources["chroma_client"] = None
        global_resources["document_collection"] = None

    # 2. Compile RAG LangGraph
    global_resources["rag_app"] = create_rag_app(gemini_llm)

    yield

# --- FastAPI App Initialization ---

app = FastAPI(title="Main App", lifespan=lifespan)

# CORS Middleware setup
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Mount the API router
app.include_router(chat_router, prefix=API_PREFIX)
app.include_router(embedding_router, prefix=API_PREFIX)

app.mount("/", StaticFiles(directory=FRONTEND_BUILD_DIR, html=True), name="static")