from langchain_core.tools import tool
from googleapiclient.discovery import build
from ...config.app_config import settings

@tool
def search_web(query: str) -> str:
    """
    Use Google Custom Search Engine (CSE) to find updated information or general knowledge.
    Input should be the specific query to search.
    """
    print(f"--- TOOL: EXECUTING GOOGLE CSE for: {query} ---")

    try:
        service = build(
            "customsearch", "v1", developerKey=settings.GOOGLE_SEARCH.API_KEY
        )
        res = service.cse().list(
            q=query,
            cx=settings.GOOGLE_SEARCH.CX_ID,
            num=5
        ).execute()

        if 'items' not in res or not res['items']:
            print(f"Warning: Google CSE returned no results for query: {query}")
            return "ERROR: GOOGLE_CSE_NO_RESULTS"

        formatted_context = "Web Search Results:\n"
        for i, item in enumerate(res['items']):
            title = item.get('title', 'N/A')
            snippet = item.get('snippet', 'N/A')
            formatted_context += f"Source {i + 1} ({title}):\n"
            formatted_context += f"{snippet}\n"

        return formatted_context

    except Exception as e:
        print(f"Web search failed (Google CSE): {e}")
        return f"Web search tool execution failed: {str(e)}"
