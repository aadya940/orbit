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
        texts = []
        for el in elements:
            try:
                texts.append(await el.inner_text())
            except Exception:
                continue
        return {"status": "success", "data": texts}
    except Exception as e:
        msg = str(e)
        if "execution context was destroyed" in msg.lower() or "navigation" in msg.lower():
            return {"status": "error", "message": "Page navigated during extraction — call dom_extract again once the page has settled."}
        return {"status": "error", "message": msg}

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
            "selector": None,
        })

    # query_selector_all can throw if the page navigates while we're querying.
    # Catch the destroyed-context error gracefully and return frames without selectors.
    try:
        for iframe in await page.query_selector_all("iframe"):
            try:
                src = await iframe.get_attribute("src") or ""
            except Exception:
                continue
            for f in frames:
                if src and urlparse(src).path == urlparse(f["url"]).path:
                    f["selector"] = f"iframe[src*='{src.split('?')[0]}']"
    except Exception as e:
        if "execution context was destroyed" in str(e).lower() or "navigation" in str(e).lower():
            return {"status": "success", "frames": frames, "note": "Page navigated during frame scan; selectors unavailable."}
        return {"status": "error", "message": str(e)}

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

async def dom_get_interactive_elements() -> Dict[str, Any]:
    """
    Returns all visible, interactive elements on the current page — buttons, links,
    inputs, selects, textareas — each with its type, display text, and a CSS selector
    you can pass directly to dom_click or dom_fill.

    Call this before dom_click or dom_fill when you don't know the exact selector.
    It scans the live DOM so the selectors are guaranteed to exist on the current page.
    """
    await global_browser.ensure_active_page()
    if not global_browser.active_page:
        return {"status": "error", "message": "Browser is not active."}
    try:
        page = global_browser.active_frame_or_page
        elements = await page.evaluate("""() => {
            const results = [];
            const seen = new Set();
            const queries = [
                'button:not([disabled])',
                'a[href]',
                'input:not([type="hidden"]):not([disabled])',
                'select:not([disabled])',
                'textarea:not([disabled])',
                '[role="button"]:not([disabled])',
                '[role="link"]',
                '[role="menuitem"]',
                '[role="tab"]',
                '[role="checkbox"]',
                '[role="radio"]',
                '[contenteditable="true"]',
            ];
            for (const q of queries) {
                for (const el of document.querySelectorAll(q)) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) continue;
                    const tag = el.tagName.toLowerCase();
                    const inputType = el.type || '';
                    const text = (
                        el.innerText ||
                        el.value ||
                        el.placeholder ||
                        el.getAttribute('aria-label') ||
                        el.getAttribute('title') ||
                        el.getAttribute('alt') ||
                        ''
                    ).trim().replace(/\\s+/g, ' ').slice(0, 100);

                    const dedupeKey = `${tag}|${inputType}|${text}`;
                    if (seen.has(dedupeKey)) continue;
                    seen.add(dedupeKey);

                    let selector = '';
                    if (el.id) {
                        selector = '#' + CSS.escape(el.id);
                    } else if (el.getAttribute('data-testid')) {
                        selector = `[data-testid="${el.getAttribute('data-testid')}"]`;
                    } else if (el.getAttribute('name')) {
                        selector = `${tag}[name="${el.getAttribute('name')}"]`;
                    } else if (el.getAttribute('aria-label')) {
                        selector = `[aria-label="${el.getAttribute('aria-label')}"]`;
                    } else if (el.getAttribute('placeholder')) {
                        selector = `${tag}[placeholder="${el.getAttribute('placeholder')}"]`;
                    } else {
                        selector = q;
                    }

                    const entry = {
                        type: tag === 'a' ? 'link' : (inputType || tag),
                        text: text || null,
                        selector: selector,
                    };
                    if (el.placeholder) entry.placeholder = el.placeholder;
                    if (el.href) entry.href = el.href;
                    results.push(entry);
                    if (results.length >= 60) return results;
                }
            }
            return results;
        }""")
        return {
            "status": "success",
            "url": global_browser.active_page.url,
            "count": len(elements),
            "elements": elements,
        }
    except Exception as e:
        msg = str(e)
        if "execution context was destroyed" in msg.lower() or "navigation" in msg.lower():
            return {"status": "error", "message": "Page navigated during element scan — call dom_get_interactive_elements again once the page has settled."}
        return {"status": "error", "message": msg}


async def dom_upload_file(selector: str, path: str) -> Dict[str, Any]:
    """Upload a file to a file input element using the DOM.

    Works with hidden <input type="file"> elements — no file dialog is opened.
    Use this instead of upload_file when the browser's file upload button does
    not open a native file dialog (e.g. LinkedIn, Greenhouse, Lever).

    selector: CSS selector for the <input type="file"> element, e.g.
              'input[type="file"]' or '.jobs-easy-apply-modal input[type="file"]'
    path:     Absolute path to the file on the server, e.g. '/workspace/RESUME.pdf'
    """
    await global_browser.ensure_active_page()
    if not global_browser.active_page:
        return {"status": "error", "message": "Browser is not active."}
    try:
        frame = global_browser.active_frame_or_page
        # Make the input visible/interactable if hidden, then set files
        await frame.evaluate("""
            (s) => {
                const el = document.querySelector(s);
                if (el) {
                    el.style.display = 'block';
                    el.style.visibility = 'visible';
                    el.style.opacity = '1';
                }
            }
        """, selector)
        await frame.set_input_files(selector, path)
        return {"status": "success", "message": f"File '{path}' set on '{selector}'"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def dom_select_option(selector: str, label: str) -> Dict[str, Any]:
    """Select an option in a <select> element or ARIA combobox/listbox dropdown.

    Handles both native HTML <select> (via Playwright's select_option) and
    custom ARIA dropdowns (role="combobox", role="listbox") by clicking the
    trigger first, waiting for the option list, then clicking the target item.

    selector: CSS selector for the <select> or combobox trigger element,
              e.g. 'select[name="country"]' or '[role="combobox"]'
    label:    The visible option text to select (case-insensitive match).
    """
    await global_browser.ensure_active_page()
    if not global_browser.active_page:
        return {"status": "error", "message": "Browser is not active."}
    try:
        frame = global_browser.active_frame_or_page
        # Detect element type — pass selector as arg to avoid injection issues
        tag = await frame.evaluate(
            "(s) => document.querySelector(s)?.tagName?.toLowerCase()", selector
        )
        if tag == "select":
            await frame.select_option(selector, label=label, timeout=5000)
            return {"status": "success", "message": f"Selected '{label}' in native <select> {selector}"}
        # Custom ARIA combobox / listbox — click to open, then click the option
        await frame.click(selector, timeout=5000)
        await frame.wait_for_selector(
            '[role="option"], [role="listbox"] li, [role="menu"] [role="menuitem"]',
            timeout=3000,
        )
        clicked = await frame.evaluate("""
            (label) => {
                const candidates = [
                    ...document.querySelectorAll('[role="option"]'),
                    ...document.querySelectorAll('[role="listbox"] li'),
                    ...document.querySelectorAll('[role="menuitem"]'),
                ];
                const target = candidates.find(el =>
                    el.textContent.trim().toLowerCase() === label.toLowerCase()
                );
                if (target) { target.click(); return true; }
                return false;
            }
        """, label)
        if clicked:
            return {"status": "success", "message": f"Selected '{label}' via ARIA combobox {selector}"}
        return {"status": "error", "message": f"Option '{label}' not found in dropdown {selector}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def dom_fill_by_label(label_text: str, value: str) -> Dict[str, Any]:
    """Fill an input by finding its associated <label> text.

    Searches for a <label> whose visible text includes *label_text*
    (case-insensitive), then fills the associated <input>, <select>, or
    <textarea> via the label's `for` attribute or DOM proximity.

    Useful for modals (e.g. LinkedIn Easy Apply) where input selectors are
    not predictable but label text is stable.

    label_text: Visible label text, e.g. "First name" or "Phone number"
    value:      Text to fill into the associated input.
    """
    await global_browser.ensure_active_page()
    if not global_browser.active_page:
        return {"status": "error", "message": "Browser is not active."}
    try:
        frame = global_browser.active_frame_or_page
        selector = await frame.evaluate("""
            (labelText) => {
                const labels = [...document.querySelectorAll('label')];
                const label = labels.find(l =>
                    l.textContent.trim().toLowerCase().includes(labelText.toLowerCase())
                );
                if (!label) return null;
                if (label.htmlFor) {
                    const el = document.getElementById(label.htmlFor);
                    if (el) return '#' + CSS.escape(label.htmlFor);
                }
                // Implicit label — find first input/select/textarea child
                let el = label.querySelector('input,select,textarea');
                // If not inside the label, check the next sibling (may be a wrapper div)
                if (!el) {
                    const sib = label.nextElementSibling;
                    if (sib) {
                        const tag = sib.tagName.toLowerCase();
                        el = (tag === 'input' || tag === 'select' || tag === 'textarea')
                            ? sib
                            : sib.querySelector('input,select,textarea');
                    }
                }
                if (!el) return null;
                if (el.id) return '#' + CSS.escape(el.id);
                if (el.name) return el.tagName.toLowerCase() + '[name="' + el.name + '"]';
                return null;
            }
        """, label_text)
        if not selector:
            return {"status": "error", "message": f"No input found for label '{label_text}'"}
        await frame.fill(selector, value, timeout=5000)
        return {"status": "success", "message": f"Filled '{label_text}' ({selector}) with '{value}'"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def dom_open_browser(name: str = "main") -> Dict[str, Any]:
    """Launch a second Chrome instance registered under *name*.

    Use this only when a second browser is needed alongside one already open.
    The new browser immediately becomes the active target for all dom_* tools.
    For the first/only browser you do NOT need this — dom_navigate auto-launches.

    name: Registry key, e.g. "secondary", "chrome2"
    """
    try:
        from .browser import open_browser
        await open_browser(name)
        return {"status": "success", "message": f"Browser '{name}' launched and active"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def dom_connect_cdp(port: int, name: str) -> Dict[str, Any]:
    """Attach dom_* tools to an externally-running Chromium app via CDP.

    The target app must have been launched with --remote-debugging-port=<port>.
    Common debug ports: 9222 (Chrome default), 9229 (many Electron apps).
    After this call, the connected app becomes the active target for dom_* tools.

    port: CDP debug port the app is listening on
    name: Registry key, e.g. "electron_app", "vscode", "slack"
    """
    try:
        from .browser import connect_browser_cdp
        await connect_browser_cdp(port, name)
        return {"status": "success", "message": f"Connected to CDP port {port}, registered as '{name}'"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def dom_switch_browser(name: str) -> Dict[str, Any]:
    """Switch which registered browser all subsequent dom_* calls target.

    name: Registry key previously used in dom_open_browser or dom_connect_cdp
          (the first browser is always "main")
    """
    try:
        from .browser import switch_active_browser
        switch_active_browser(name)
        return {"status": "success", "message": f"Active browser switched to '{name}'"}
    except KeyError as e:
        return {"status": "error", "message": str(e)}


async def dom_run(code: str) -> Dict[str, Any]:
    """Execute arbitrary async Playwright Python code against the active browser page.

    Pre-bound variables available in the code:
      page    — the active Playwright Page
      context — the active BrowserContext
      frame   — the active frame (or page if no frame is set)

    The return value of the last expression (if any) is captured and returned
    in the result field. stdout is also captured and returned.

    Use this for any browser interaction not covered by other dom_* tools:
      • Mouse move/hover : await page.mouse.move(x, y)
      • Keyboard         : await page.keyboard.press('Escape')
      • Scroll           : await page.evaluate('window.scrollBy(0, 500)')
      • Raw JS           : result = await page.evaluate('() => document.title')
      • Hover trigger    : await page.hover('selector')
      • Drag             : await page.drag_and_drop('src', 'dst')

    Destructive calls (page.close, context.close, browser.close) are blocked.
    Any Python exception is caught and returned as status "error" — Chrome will
    not crash from a bad code string.
    """
    import ast
    import io
    import contextlib

    await global_browser.ensure_active_page()
    page = global_browser.active_page
    if not page:
        return {"status": "error", "message": "Browser is not active."}

    # Block calls that would close the browser or page
    BLOCKED = ("page.close", "context.close", "browser.close", "playwright.stop")
    for b in BLOCKED:
        if b in code:
            return {"status": "error", "message": f"Blocked: '{b}' is not allowed in dom_run."}

    context = global_browser._browser_context
    frame = global_browser.active_frame_or_page

    # Wrap the code in an async function so `await` works at top level
    # Indent every line of user code by 4 spaces
    indented = "\n".join("    " + line for line in code.splitlines())
    fn_src = f"async def _dom_run_fn(page, context, frame):\n{indented}\n"

    # Validate syntax before exec to give a clean error
    try:
        ast.parse(fn_src)
    except SyntaxError as e:
        return {"status": "error", "message": f"SyntaxError: {e}"}

    globs: dict = {}
    try:
        exec(fn_src, globs)  # noqa: S102
    except Exception as e:
        return {"status": "error", "message": f"Compile error: {e}"}

    stdout_buf = io.StringIO()
    result = None
    try:
        with contextlib.redirect_stdout(stdout_buf):
            result = await globs["_dom_run_fn"](page, context, frame)
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "stdout": stdout_buf.getvalue(),
        }

    return {
        "status": "success",
        "result": result,
        "stdout": stdout_buf.getvalue(),
    }


async def dom_solve_turnstile() -> Dict[str, Any]:
    """Solve a Cloudflare Turnstile challenge by performing human-like mouse
    movements across the page, then clicking the Turnstile checkbox.

    Call this whenever a Turnstile 'Verify you are human' widget appears.
    The tool:
      1. Jiggles the mouse across the viewport in randomised bezier curves to
         build up mouse-movement entropy (Turnstile requires this before the
         checkbox becomes interactive).
      2. Locates the Turnstile iframe via common selectors.
      3. Moves to the checkbox inside the iframe and clicks it.
      4. Waits up to 8 s for the challenge to clear.

    Returns status "success" when the Turnstile widget disappears or the
    checkbox is checked, "error" otherwise.
    """
    import asyncio
    import math
    import random

    await global_browser.ensure_active_page()
    page = global_browser.active_page
    if not page:
        return {"status": "error", "message": "Browser is not active."}

    try:
        # ── 1. Human-like mouse jiggle across the viewport ──────────────────
        vw, vh = 1280, 768  # matches _CHROME_FLAGS window size

        def _bezier(p0, p1, p2, p3, t):
            """Cubic bezier point at parameter t."""
            mt = 1 - t
            return (
                mt**3 * p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0],
                mt**3 * p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1],
            )

        async def _human_move(x0, y0, x1, y1, steps=None):
            if steps is None:
                dist = math.hypot(x1 - x0, y1 - y0)
                steps = max(10, int(dist / 8))
            cx1 = random.uniform(min(x0, x1), max(x0, x1))
            cy1 = random.uniform(min(y0, y1) - 60, max(y0, y1) + 60)
            cx2 = random.uniform(min(x0, x1), max(x0, x1))
            cy2 = random.uniform(min(y0, y1) - 60, max(y0, y1) + 60)
            for i in range(steps + 1):
                t = i / steps
                px, py = _bezier((x0, y0), (cx1, cy1), (cx2, cy2), (x1, y1), t)
                await page.mouse.move(px, py)
                await asyncio.sleep(random.uniform(0.005, 0.018))

        # Start from center-ish, jiggle around 4–6 random waypoints
        cur_x, cur_y = vw // 2, vh // 2
        await page.mouse.move(cur_x, cur_y)
        for _ in range(random.randint(4, 6)):
            nx = random.uniform(vw * 0.1, vw * 0.9)
            ny = random.uniform(vh * 0.1, vh * 0.7)
            await _human_move(cur_x, cur_y, nx, ny)
            cur_x, cur_y = nx, ny
            await asyncio.sleep(random.uniform(0.05, 0.15))

        # ── 2. Find the Turnstile iframe ─────────────────────────────────────
        IFRAME_SELECTORS = [
            "iframe[src*='challenges.cloudflare.com']",
            "iframe[src*='turnstile']",
            "iframe[title*='Widget']",
            "iframe[title*='challenge']",
        ]
        iframe_el = None
        for sel in IFRAME_SELECTORS:
            try:
                iframe_el = await page.query_selector(sel)
                if iframe_el:
                    break
            except Exception:
                pass

        if not iframe_el:
            return {
                "status": "error",
                "message": "Turnstile iframe not found — widget may not be present.",
            }

        # ── 3. Get iframe bounding box and click the checkbox ────────────────
        bbox = await iframe_el.bounding_box()
        if not bbox:
            return {"status": "error", "message": "Turnstile iframe has no bounding box."}

        # The checkbox sits in the left portion of the iframe (~24 px from left, vertically centred)
        cb_x = bbox["x"] + 24
        cb_y = bbox["y"] + bbox["height"] / 2

        # Move to checkbox with human curve, then click
        await _human_move(cur_x, cur_y, cb_x, cb_y)
        await asyncio.sleep(random.uniform(0.08, 0.2))
        await page.mouse.click(cb_x, cb_y)

        # ── 4. Wait for Turnstile to clear (widget disappears or gets checked) ─
        for _ in range(16):
            await asyncio.sleep(0.5)
            still_visible = await page.query_selector(IFRAME_SELECTORS[0])
            if not still_visible:
                return {"status": "success", "message": "Turnstile solved — widget cleared."}
            # Check if checkbox inside iframe is now checked
            try:
                frame = await iframe_el.content_frame()
                if frame:
                    checked = await frame.evaluate(
                        "() => !!document.querySelector('input[type=\"checkbox\"]')?.checked"
                    )
                    if checked:
                        return {"status": "success", "message": "Turnstile solved — checkbox checked."}
            except Exception:
                pass

        return {"status": "error", "message": "Turnstile did not clear within 8 s — may need manual intervention."}

    except Exception as e:
        return {"status": "error", "message": str(e)}


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
