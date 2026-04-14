from __future__ import annotations

import time
from typing import Any, Dict, List

from ddgs import DDGS


def duckduckgo_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search the web using DuckDuckGo and return titles, snippets, and URLs.

    Returns a dict with key "results" containing a list of dicts, each with:
      - title: page title
      - snippet: short excerpt from the page
      - url: full URL

    On failure returns {"error": "<reason>", "results": []}.
    """
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            hits = list(DDGS().text(query, max_results=max_results))
            if hits:
                return {
                    "results": [
                        {
                            "title": h.get("title", ""),
                            "snippet": h.get("body", ""),
                            "url": h.get("href", ""),
                        }
                        for h in hits
                    ]
                }
            # Empty but no exception — DDG may be rate-limiting; back off
            time.sleep(2 * (attempt + 1))
        except Exception as exc:
            last_exc = exc
            time.sleep(2 * (attempt + 1))

    reason = str(last_exc) if last_exc else "DuckDuckGo returned no results after 3 attempts"
    return {"error": reason, "results": []}
