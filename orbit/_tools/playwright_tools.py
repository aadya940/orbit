import logging
from typing import Dict, Any
from urllib.parse import urlparse

from .browser import global_browser

log = logging.getLogger("orbit.playwright_tools")

async def dom_navigate(url: str) -> Dict[str, Any]:
    """Navigate to a specific URL using the inside-DOM Playwright engine. 
    This automatically launches the built-in browser if it is not open. 
    Use this tool first whenever asked to visit a webpage, rather than relying on OS navigation.
    
    IMPORTANT STATE RULE: Before we interact visually with a browser, we must ensure
    the bounding box is completely synchronized. We will maximize the window here
    because we assume that if a navigation was called, the browser is our current target.
    """
    await global_browser.ensure_active_page()
    if not global_browser.active_page:
        return {"status": "error", "message": "Browser is not active."}
    try:
        # Before navigating, let's force the browser process window to maximize via shell
        # so that if the LLM swaps to "visual tools" (click_first), the coordinates perfectly map.
        import subprocess
        subprocess.run(["xdotool", "search", "--onlyvisible", "--class", "chromium", "windowactivate", "windowsize", "100%", "100%"], capture_output=True)        # Reset any active frame before navigating
        global_browser.active_frame = None        # Proceed with normal DOM navigation
        await global_browser.active_frame_or_page.goto(url, wait_until="domcontentloaded")
        return {"status": "success", "message": f"Navigated to {url}, forced UI window maximize"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def dom_click(selector: str) -> Dict[str, Any]:
    """Click an element matching the given CSS selector using the DOM."""
    await global_browser.ensure_active_page()
    if not global_browser.active_page:
        return {"status": "error", "message": "Browser is not active."}
    try:
        await global_browser.active_frame_or_page.click(selector, timeout=5000)
        return {"status": "success", "message": f"Clicked {selector}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def dom_fill(selector: str, value: str) -> Dict[str, Any]:
    """Fill an input field matching the CSS selector with a value using the DOM."""
    await global_browser.ensure_active_page()
    if not global_browser.active_page:
        return {"status": "error", "message": "Browser is not active."}
    try:
        await global_browser.active_frame_or_page.fill(selector, value, timeout=5000)
        return {"status": "success", "message": f"Filled {selector} with {value}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def dom_extract(selector: str = "body") -> Dict[str, Any]:
    """Extract internal text content from elements matching the given CSS selector using the DOM."""
    await global_browser.ensure_active_page()
    if not global_browser.active_page:
        return {"status": "error", "message": "Browser is not active."}
    try:
        elements = await global_browser.active_frame_or_page.query_selector_all(selector)
        texts = [await el.inner_text() for el in elements]
        return {"status": "success", "data": texts}
    except Exception as e:
        return {"status": "error", "message": str(e)}
async def dom_list_frames() -> Dict[str, Any]:
    """List all frames/iframes on the current page."""
    await global_browser.ensure_active_page()
    page = global_browser.active_page
    if not page:
        return {"status": "error", "message": "Browser is not active."}
    
    frames = []
    for i, frame in enumerate(page.frames):
        frames.append({
            "index": i,
            "name": frame.name,
            "url": frame.url,
            "selector": None
        })
    
    for iframe in await page.query_selector_all("iframe"):
        src = await iframe.get_attribute("src") or ""
        for f in frames:
            if src and urlparse(src).path == urlparse(f["url"]).path:
                f["selector"] = f"iframe[src*='{src.split('?')[0]}']"
    return {"status": "success", "frames": frames}

async def dom_switch_frame(selector_or_index=None) -> Dict[str, Any]:
    """Switch the active frame using an exact index or CSS selector."""
    await global_browser.ensure_active_page()
    page = global_browser.active_page
    if not page:
        return {"status": "error", "message": "Browser is not active."}
    
    if selector_or_index is None:
        global_browser.active_frame = None
        return {"status": "success", "frame": "main"}
    
    if isinstance(selector_or_index, int):
        global_browser.active_frame = page.frames[selector_or_index]
    else:
        element = await page.query_selector(selector_or_index)
        if hasattr(element, 'content_frame'):
            global_browser.active_frame = await element.content_frame()
        else:
            return {"status": "error", "message": "Selector did not match an iframe."}
    
    return {"status": "success", "url": getattr(global_browser.active_frame, 'url', None)}

async def dom_switch_frame_default() -> Dict[str, Any]:
    """Switch back to the main document frame."""
    global_browser.active_frame = None
    return {"status": "success", "frame": "main"}

async def dom_click_text(text: str) -> Dict[str, Any]:
    """Click an element by its text content using the DOM."""
    await global_browser.ensure_active_page()
    if not global_browser.active_page:
        return {"status": "error", "message": "Browser is not active."}
    
    frame = global_browser.active_frame_or_page
    try:
        try:
            await frame.get_by_role("button", name=text, exact=False).first.click(timeout=3000)
        except Exception:
            await frame.get_by_text(text, exact=False).first.click(timeout=3000)
        return {"status": "success", "message": f"Clicked element with text '{text}'"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
