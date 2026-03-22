from tools.search_tool import build_search_tool
 
TOOL_REGISTRY: dict = {
    "tavily_search": build_search_tool,
}
 
__all__ = ["TOOL_REGISTRY"]