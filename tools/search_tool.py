import os
from langchain_community.tools.tavily_search import TavilySearchResults


def build_search_tool() -> TavilySearchResults:
    """
    Returns a configured TavilySearchResults tool instance.
    Requires TAVILY_API_KEY in environment.
    """

    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError("TAVILY_API_KEY not set in environment")

    return TavilySearchResults(
        max_results=int(os.getenv("SEARCH_MAX_RESULTS", "5")),
    )