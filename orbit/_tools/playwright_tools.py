import logging
from typing import Dict, Any

from .browser import global_browser

log = logging.getLogger("orbit.playwright_tools")

async def dom_navigate(url: str) -> Dict[str, Any]:
    """Navigate the browser to a specific URL using the inside-DOM Playwright engine."""
    if not global_browser.active_page:
        return {"status": "error", "message": "Browser is not active."}
    try:
        await global_browser.active_page.goto(url, wait_until="domcontentloaded")
        return {"status": "success", "message": f"Navigated to {url}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def dom_click(selector: str) -> Dict[str, Any]:
    """Click an element matching the given CSS selector using the DOM."""
    if not global_browser.active_page:
        return {"status": "error", "message": "Browser is not active."}
    try:
        await global_browser.active_page.click(selector, timeout=5000)
        return {"status": "success", "message": f"Clicked {selector}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def dom_fill(selector: str, value: str) -> Dict[str, Any]:
    """Fill an input field matching the CSS selector with a value using the DOM."""
    if not global_browser.active_page:
        return {"status": "error", "message": "Browser is not active."}
    try:
        await global_browser.active_page.fill(selector, value, timeout=5000)
        return {"status": "success", "message": f"Filled {selector} with {value}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def dom_extract(selector: str = "body") -> Dict[str, Any]:
    """Extract internal text content from elements matching the given CSS selector using the DOM."""
    if not global_browser.active_page:
        return {"status": "error", "message": "Browser is not active."}
    try:
        elements = await global_browser.active_page.query_selector_all(selector)
        texts = [await el.inner_text() for el in elements]
        return {"status": "success", "data": texts}
    except Exception as e:
        return {"status": "error", "message": str(e)}