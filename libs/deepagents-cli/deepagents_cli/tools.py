"""Custom tools for the CLI agent."""

import os
from typing import Any, Literal

import requests
from markdownify import markdownify

from deepagents_cli.config import settings

# Initialize search clients based on available API keys
_tavily_client = None
_serper_api_key = os.environ.get("SERPER_API_KEY")

if settings.has_tavily:
    try:
        from tavily import TavilyClient
        _tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    except ImportError:
        pass

# Try to import duckduckgo_search
_ddg_available = False
try:
    from duckduckgo_search import DDGS
    _ddg_available = True
except ImportError:
    pass


def is_search_available() -> bool:
    """Check if any search provider is available."""
    return _tavily_client is not None or _serper_api_key is not None or _ddg_available


def get_search_provider() -> str | None:
    """Get the name of the active search provider."""
    if _tavily_client is not None:
        return "Tavily"
    if _serper_api_key is not None:
        return "Serper"
    if _ddg_available:
        return "DuckDuckGo"
    return None


def http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
    params: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Make HTTP requests to APIs and web services.

    Args:
        url: Target URL
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        headers: HTTP headers to include
        data: Request body data as JSON dict
        params: URL query parameters
        timeout: Request timeout in seconds

    Returns:
        Dictionary with response data including status, headers, and content
    """
    try:
        kwargs = {"url": url, "method": method.upper(), "timeout": timeout}

        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        if data:
            kwargs["json"] = data

        response = requests.request(**kwargs)

        try:
            content = response.json()
        except:
            content = response.text

        return {
            "success": response.status_code < 400,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": content,
            "url": response.url,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request timed out after {timeout} seconds",
            "url": url,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request error: {e!s}",
            "url": url,
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Error making request: {e!s}",
            "url": url,
        }


def _search_duckduckgo(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search using DuckDuckGo (free, no API key needed)."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return {
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "content": r.get("body", ""),
                        "score": 1.0,  # DDG doesn't provide scores
                    }
                    for r in results
                ],
                "query": query,
                "provider": "duckduckgo",
            }
    except Exception as e:
        return {"error": f"DuckDuckGo search error: {e!s}", "query": query}


def _search_serper(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search using Serper API (Google results)."""
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": _serper_api_key,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": max_results},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for r in data.get("organic", [])[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "content": r.get("snippet", ""),
                "score": 1.0,
            })

        return {
            "results": results,
            "query": query,
            "provider": "serper",
        }
    except Exception as e:
        return {"error": f"Serper search error: {e!s}", "query": query}


def _search_tavily(
    query: str,
    max_results: int = 5,
    topic: str = "general",
    include_raw_content: bool = False,
) -> dict[str, Any]:
    """Search using Tavily API."""
    try:
        result = _tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        result["provider"] = "tavily"
        return result
    except Exception as e:
        return {"error": f"Tavily search error: {e!s}", "query": query}


def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Search the web for current information and documentation.

    Uses multiple providers with fallback: Tavily → Serper → DuckDuckGo (free).

    This tool searches the web and returns relevant results. After receiving results,
    you MUST synthesize the information into a natural, helpful response for the user.

    Args:
        query: The search query (be specific and detailed)
        max_results: Number of results to return (default: 5)
        topic: Search topic type - "general" for most queries, "news" for current events
        include_raw_content: Include full page content (warning: uses more tokens)

    Returns:
        Dictionary containing:
        - results: List of search results, each with:
            - title: Page title
            - url: Page URL
            - content: Relevant excerpt from the page
            - score: Relevance score (0-1)
        - query: The original search query
        - provider: Which search provider was used

    IMPORTANT: After using this tool:
    1. Read through the 'content' field of each result
    2. Extract relevant information that answers the user's question
    3. Synthesize this into a clear, natural language response
    4. Cite sources by mentioning the page titles or URLs
    5. NEVER show the raw JSON to the user - always provide a formatted response
    """
    # Try Tavily first (best quality)
    if _tavily_client:
        result = _search_tavily(query, max_results, topic, include_raw_content)
        if "error" not in result:
            return result

    # Try Serper as backup (Google results)
    if _serper_api_key:
        result = _search_serper(query, max_results)
        if "error" not in result:
            return result

    # Fallback to DuckDuckGo (free, no API key)
    if _ddg_available:
        result = _search_duckduckgo(query, max_results)
        if "error" not in result:
            return result

    # No providers available
    return {
        "error": "No search provider available. Set TAVILY_API_KEY or SERPER_API_KEY, or install duckduckgo-search (pip install duckduckgo-search).",
        "query": query,
    }


def fetch_url(url: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch content from a URL and convert HTML to markdown format.

    This tool fetches web page content and converts it to clean markdown text,
    making it easy to read and process HTML content. After receiving the markdown,
    you MUST synthesize the information into a natural, helpful response for the user.

    Args:
        url: The URL to fetch (must be a valid HTTP/HTTPS URL)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dictionary containing:
        - success: Whether the request succeeded
        - url: The final URL after redirects
        - markdown_content: The page content converted to markdown
        - status_code: HTTP status code
        - content_length: Length of the markdown content in characters

    IMPORTANT: After using this tool:
    1. Read through the markdown content
    2. Extract relevant information that answers the user's question
    3. Synthesize this into a clear, natural language response
    4. NEVER show the raw markdown to the user unless specifically requested
    """
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepAgents/1.0)"},
        )
        response.raise_for_status()

        # Convert HTML content to markdown
        markdown_content = markdownify(response.text)

        return {
            "url": str(response.url),
            "markdown_content": markdown_content,
            "status_code": response.status_code,
            "content_length": len(markdown_content),
        }
    except Exception as e:
        return {"error": f"Fetch URL error: {e!s}", "url": url}
