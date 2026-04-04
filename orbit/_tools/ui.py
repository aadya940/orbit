import asyncio
import subprocess
import platform
import os
import time
from dataclasses import dataclass

import base64
from io import BytesIO

from google.adk.tools.tool_context import ToolContext
from google.genai import types

from typing import Optional, Any, Dict, List
from .._oculus_client import OculOS

oculos_client = OculOS()

try:
    import pyautogui

    _PYAUTOGUI_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:  # pragma: no cover - environment-dependent (e.g. headless CI)
    pyautogui = None
    _PYAUTOGUI_IMPORT_ERROR = e


def _require_pyautogui() -> None:
    if pyautogui is None:
        raise RuntimeError(
            "pyautogui is unavailable in this environment "
            f"(import error: {_PYAUTOGUI_IMPORT_ERROR!r})"
        )

#
# Speed: tiny TTL cache for discovery calls
# -----------------------------------------
# OculOS discovery calls are synchronous HTTP round-trips. We keep a short-lived cache
# to avoid repeating identical calls back-to-back within the same UI state.


_DISCOVERY_CACHE_TTL_SEC = 0.75

# Maps oculos_id → {pid, query, element_type} so interact_with_element can
# re-find an element automatically when its ID goes stale between calls.
_element_meta: dict[str, dict] = {}


@dataclass
class _CacheEntry:
    t: float
    value: Any


_discovery_cache: dict[tuple, _CacheEntry] = {}


def _cache_get(key: tuple) -> Any:
    ent = _discovery_cache.get(key)
    if not ent:
        return None
    if (time.monotonic() - float(ent.t)) > _DISCOVERY_CACHE_TTL_SEC:
        _discovery_cache.pop(key, None)
        return None
    return ent.value


def _cache_set(key: tuple, value: Any) -> Any:
    _discovery_cache[key] = _CacheEntry(t=time.monotonic(), value=value)
    return value


def _invalidate_discovery_cache() -> None:
    # Keep it simple: any interaction likely changes the accessibility tree.
    _discovery_cache.clear()


def _norm_query(q: Optional[str]) -> Optional[str]:
    if q is None:
        return None
    q = str(q)
    q = " ".join(q.split())
    return q if q else None


def _cached_list_windows() -> list[dict]:
    key = ("list_windows",)
    hit = _cache_get(key)
    if hit is not None:
        return hit
    return _cache_set(key, oculos_client.list_windows())


def _cached_find_elements(
    pid: int,
    *,
    query: Optional[str] = None,
    element_type: Optional[str] = None,
    interactive: Optional[bool] = None,
) -> list[dict]:
    key = (
        "find_elements",
        int(pid),
        _norm_query(query),
        str(element_type) if element_type is not None else None,
        bool(interactive) if interactive is not None else None,
    )
    hit = _cache_get(key)
    if hit is not None:
        return hit
    return _cache_set(
        key,
        oculos_client.find_elements(
            int(pid),
            query=_norm_query(query),
            element_type=element_type,
            interactive=interactive,
        ),
    )


def _cached_find_elements_hwnd(
    hwnd: int,
    *,
    query: Optional[str] = None,
    element_type: Optional[str] = None,
    interactive: Optional[bool] = None,
) -> list[dict]:
    key = (
        "find_elements_hwnd",
        int(hwnd),
        _norm_query(query),
        str(element_type) if element_type is not None else None,
        bool(interactive) if interactive is not None else None,
    )
    hit = _cache_get(key)
    if hit is not None:
        return hit
    return _cache_set(
        key,
        oculos_client.find_elements_hwnd(
            int(hwnd),
            query=_norm_query(query),
            element_type=element_type,
            interactive=interactive,
        ),
    )


def _slim_element(el: dict) -> dict:
    """Strip an element dict to the fields the LLM needs."""
    slim = {"oculos_id": el.get("oculos_id")}
    if el.get("element_type"):
        slim["element_type"] = el["element_type"]
    for key in ("name", "title", "label"):
        val = el.get(key)
        if val:
            slim[key] = val
    for key in ("value", "text_content"):
        val = el.get(key)
        if val:
            slim[key] = val
    for key in ("checked", "toggle_state", "is_selected"):
        val = el.get(key)
        if val is not None:
            slim[key] = val
    if el.get("is_enabled") is False:
        slim["is_enabled"] = False
    return slim


def _slim_elements(elements: list[dict]) -> list[dict]:
    return [_slim_element(el) for el in elements]


def list_active_windows() -> Dict[str, Any]:
    """
    Retrieves a list of all currently visible desktop windows.
    Use this to find the Process ID (pid) of the application you want to control.

    Returns:
        dict: A list of dictionaries containing 'pid' and 'title' for each window.
    """
    try:
        windows = _cached_list_windows()
        return {"status": "success", "windows": windows}
    except Exception as e:
        return {"status": "error", "message": f"Failed to list windows: {str(e)}"}


async def wait_for_element(
    pid: int,
    query: str,
    timeout: int = 3,
    interval: float = 0.5,
    max_polls: Optional[int] = None,
    element_type: Optional[str] = None,
    interactive: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Polls until a UI element appears in the window or timeout is reached.
    Use after launching an app, clicking a button, or navigating — any time the UI needs a moment to load.

    Robustness (website-agnostic):
    - Tries exact query first, then query.lower() in the same poll if no match (handles case differences).
    - Default timeout is 5s to limit cost when the element never appears; pass timeout=10 for slow loads.
    Each poll is one or two find_ui_elements calls (OculOS round-trip, often ~1s each).

    Args:
        pid (int): The Process ID of the window to search inside.
        query (str): The text or name of the element to wait for.
        timeout (int): Maximum seconds to wait. Default 5; use 10 for slow pages.
        interval (float): Unused; kept for API compatibility.
        element_type (str, optional): Semantic role e.g. 'Button', 'Edit'.
        interactive (bool, optional): If True, only match interactive elements.
    """
    start = time.perf_counter()
    polls_done = 0
    last_result: Optional[Dict[str, Any]] = None
    query_lower = query.lower() if query else ""

    while time.perf_counter() - start < timeout and (
        max_polls is None or polls_done < max_polls
    ):
        result = find_ui_elements(
            pid,
            query=query,
            element_type=element_type,
            interactive=interactive,
        )
        polls_done += 1
        last_result = result

        if result["status"] == "success" and result.get("elements"):
            elapsed = time.perf_counter() - start
            return {
                "status": "success",
                "message": f"Element '{query}' found after {round(elapsed, 2)}s ({polls_done} polls).",
                "elements": result["elements"],
                "elapsed_sec": round(elapsed, 3),
                "polls_done": polls_done,
            }

        if (
            query_lower
            and query != query_lower
            and (max_polls is None or polls_done < max_polls)
        ):
            result = find_ui_elements(
                pid,
                query=query_lower,
                element_type=element_type,
                interactive=interactive,
            )
            polls_done += 1
            last_result = result
            if result["status"] == "success" and result.get("elements"):
                elapsed = time.perf_counter() - start
                return {
                    "status": "success",
                    "message": f"Element '{query}' found (via lowercase) after {round(elapsed, 2)}s ({polls_done} polls).",
                    "elements": result["elements"],
                    "elapsed_sec": round(elapsed, 3),
                    "polls_done": polls_done,
                }

        # Yield to the event loop; spacing polls avoids hot-looping when an element is absent.
        await asyncio.sleep(interval if interval and interval > 0 else 0)

    elapsed = time.perf_counter() - start
    timeout_message = (
        f"Element '{query}' not found after {round(elapsed, 2)}s ({polls_done} polls). "
        f"Try the same query in lowercase, a shorter substring, or fallback_vision_agent. Use timeout=10 for slow pages."
    )
    return {
        "status": "timeout",
        "message": timeout_message,
        "query": query,
        "pid": pid,
        "elapsed_sec": round(elapsed, 3),
        "polls_done": polls_done,
        "timeout_sec": timeout,
        "last_poll_status": last_result.get("status") if last_result else None,
        "last_poll_message": (
            last_result.get("message", "")[:200] if last_result else None
        ),
    }


def manage_window(
    action: str, pid: Optional[int] = None, app_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Manages the state of a specific window or launches a new application.

    Args:
        action (str):
            The action to perform. Must be 'focus', 'close', or 'launch'.
        pid (int, optional): The Process ID of the window. Required for 'focus' and 'close'.
        app_name (str, optional): The executable name (e.g., 'chrome.exe', 'notepad.exe'). Required ONLY for 'launch'.
    """
    try:
        if action == "launch" and app_name:
            system = platform.system()
            if system == "Windows":
                try:
                    os.startfile(app_name)
                except (FileNotFoundError, OSError):
                    subprocess.Popen(f"start {app_name}", shell=True)
            elif system == "Darwin":
                subprocess.Popen(["open", "-a", app_name])
            else:  # Linux
                subprocess.Popen(
                    app_name, shell=True
                )  # shell=True handles args in app_name string

            return {
                "status": "success",
                "message": f"Successfully launched {app_name}. You can now run list_active_windows to find its PID.",
            }
        elif action == "focus" and pid is not None:
            oculos_client.focus_window(pid)
            return {"status": "success", "message": f"Window {pid} focused."}
        elif action == "close" and pid is not None:
            oculos_client.close_window(pid)
            return {"status": "success", "message": f"Window {pid} closed."}
        else:
            return {
                "status": "error",
                "message": "Invalid action or missing required parameters (pid for focus/close, app_name for launch).",
            }
    except Exception as e:
        return {"status": "error", "message": f"Failed to manage window: {str(e)}"}


def find_ui_elements(
    pid: int,
    query: Optional[str] = None,
    element_type: Optional[str] = None,
    interactive: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Searches the accessibility tree of a specific window for UI elements.
    Returns a list of matching elements. You MUST use the 'oculos_id' from these results to interact with them.

    Args:
        pid (int):
            The Process ID of the window to search inside.
        query (str, optional):
            The text, name, or title of the element to search for (e.g., 'Submit', 'File').
        element_type (str, optional):
            The semantic role of the element (e.g., 'Button', 'Document', 'Edit').
        interactive (bool, optional):
            If True, only returns elements that can be clicked or typed into.
    """
    try:
        elements = _cached_find_elements(
            pid, query=query, element_type=element_type, interactive=interactive
        )
        if not elements:
            return {
                "status": "success",
                "message": "No elements found matching the criteria.",
                "elements": [],
            }
        for el in elements:
            eid = el.get("oculos_id")
            if eid:
                _element_meta[eid] = {
                    "pid": pid,
                    "query": query,
                    "element_type": element_type,
                }
        return {"status": "success", "elements": _slim_elements(elements)}
    except Exception as e:
        return {"status": "error", "message": f"Failed to find elements: {str(e)}"}


def fill_form_fields(
    pid: int,
    field_labels: List[str],
    field_values: List[str],
) -> Dict[str, Any]:
    """
    Fill multiple form fields in a single tool call.

    Instead of calling find_ui_elements + interact_with_element for each field,
    pass all fields at once as parallel lists.

    Args:
        pid (int): Process ID of the window containing the form.
        field_labels (list[str]): Field labels/queries in order —
            e.g. ["First name", "Last name", "Phone number"]
        field_values (list[str]): Values to type, matching the labels by index —
            e.g. ["Jane", "Doe", "555-0100"]

    Returns:
        dict with "filled" (succeeded), "errors" (failed), and "status".
    """
    if len(field_labels) != len(field_values):
        return {
            "status": "error",
            "message": "field_labels and field_values must be the same length.",
        }
    filled: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    for label, value in zip(field_labels, field_values):
        try:
            elements = oculos_client.find_elements(
                pid, query=str(label), interactive=True
            )
            if not elements:
                errors[label] = "element not found"
                continue
            eid = elements[0]["oculos_id"]
            _element_meta[eid] = {"pid": pid, "query": str(label), "element_type": None}
            oculos_client.set_text(eid, str(value))
            filled[label] = value
        except Exception as e:
            errors[label] = str(e)
    _invalidate_discovery_cache()
    status = "success" if not errors else ("partial" if filled else "error")
    return {"status": status, "filled": filled, "errors": errors}


def find_ui_elements_hwnd(
    hwnd: int,
    query: Optional[str] = None,
    element_type: Optional[str] = None,
    interactive: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Searches the accessibility tree of a specific window by hwnd for UI elements.
    Use this for transient windows like context menus (often hosted in PopupHost).

    Args:
        hwnd (int): The window handle to search inside.
        query (str, optional): Text/name/title to search for.
        element_type (str, optional): Semantic role (e.g. 'MenuItem', 'Button').
        interactive (bool, optional): If True, only returns interactive elements.
    """
    try:
        elements = _cached_find_elements_hwnd(
            hwnd, query=query, element_type=element_type, interactive=interactive
        )
        if not elements:
            return {
                "status": "success",
                "message": "No elements found matching the criteria.",
                "elements": [],
            }
        return {"status": "success", "elements": _slim_elements(elements)}
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to find elements by hwnd: {str(e)}",
        }


def _prune_accessibility_tree(node: dict) -> dict:
    """Recursively removes layout data and empty containers to save LLM context window tokens."""
    pruned_node = {
        "id": node.get("oculos_id"),
        "role": node.get("element_type"),
        "name": node.get("title") or node.get("name", ""),
    }

    if not pruned_node["name"]:
        del pruned_node["name"]

    if "children" in node and node["children"]:
        valid_children = []
        for child in node["children"]:
            pruned_child = _prune_accessibility_tree(child)
            # Keep the child if it has a name, a specific role (not just a Pane), or has valid children of its own
            if (
                pruned_child.get("name")
                or pruned_child.get("role") != "Pane"
                or "children" in pruned_child
            ):
                valid_children.append(pruned_child)

        if valid_children:
            pruned_node["children"] = valid_children

    return pruned_node


def get_window_tree(pid: int) -> Dict[str, Any]:
    """
    Retrieves the full UI element tree for a given window.
    Use this ONLY if find_ui_elements fails and you need to inspect the raw structural layout of the app.

    Args:
        pid (int): The Process ID of the window.
    """
    try:
        raw_tree = oculos_client.get_tree(pid)
        # Prune the tree before sending it back to the LLM
        lean_tree = _prune_accessibility_tree(raw_tree)
        return {"status": "success", "tree": lean_tree}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get tree: {str(e)}"}


def get_window_tree_hwnd(hwnd: int) -> Dict[str, Any]:
    """
    Retrieves the full UI element tree for a given window handle (hwnd).
    Useful for transient windows such as context menus hosted in PopupHost.

    Args:
        hwnd (int): The window handle.
    """
    try:
        raw_tree = oculos_client.get_tree_hwnd(hwnd)
        lean_tree = _prune_accessibility_tree(raw_tree)
        return {"status": "success", "tree": lean_tree}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get tree by hwnd: {str(e)}"}


def get_popuphost_menu_window(pid: int) -> Dict[str, Any]:
    """
    Heuristic helper: pick the most likely PopupHost window for an open context menu.

    On Windows, desktop and shell context menus are often hosted in transient windows titled
    'PopupHost' (explorer.exe). These may not be discoverable via the parent window's pid
    accessibility tree (e.g. Program Manager), so you need the specific menu hwnd.

    This tool scans list_active_windows() output and returns the largest visible PopupHost
    window for the given pid, which tends to correspond to the currently open menu surface.

    Args:
        pid (int): The explorer.exe pid (e.g. Program Manager pid) that owns the PopupHost windows.
    """
    try:
        windows = oculos_client.list_windows()
        candidates: list[dict[str, Any]] = []
        for w in windows:
            if not w.get("visible"):
                continue
            if w.get("pid") != pid:
                continue
            if (w.get("title") or "") != "PopupHost":
                continue
            rect = w.get("rect") or {}
            area = int(rect.get("width") or 0) * int(rect.get("height") or 0)
            candidates.append({**w, "_area": area})

        if not candidates:
            return {
                "status": "error",
                "message": f"No visible PopupHost windows found for pid={pid}.",
            }

        # Prefer the largest PopupHost window; this typically matches the open menu panel.
        best = max(candidates, key=lambda ww: int(ww.get("_area") or 0))
        best.pop("_area", None)
        return {
            "status": "success",
            "hwnd": best.get("hwnd"),
            "pid": best.get("pid"),
            "rect": best.get("rect"),
            "window": best,
            "message": "Selected most likely PopupHost menu window (largest area).",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to find PopupHost menu window: {str(e)}",
        }


async def interact_with_element(
    element_id: str,
    action: str,
    text_input: Optional[str] = None,
    scroll_direction: Optional[str] = None,
    range_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Performs a physical interaction with a specific UI element using its oculos_id.

    Args:
        element_id (str): The 'oculos_id' of the target element.
        action (str): The interaction type. Must be one of: 'click', 'set_text', 'send_keys', 'focus',
                      'toggle', 'expand', 'collapse', 'select', 'set_range', 'scroll', 'scroll_into_view', 'highlight'.
        text_input (str, optional): Required ONLY for 'set_text' and 'send_keys'.
        scroll_direction (str, optional): Required ONLY for 'scroll'. E.g., 'up', 'down', 'left', 'right'.
        range_value (float, optional): Required ONLY for 'set_range'.
    """

    def _do(eid: str) -> None:
        if action == "click":
            oculos_client.click(eid)
        elif action == "set_text" and text_input is not None:
            oculos_client.set_text(eid, text_input)
        elif action == "send_keys" and text_input is not None:
            oculos_client.send_keys(eid, text_input)
        elif action == "focus":
            oculos_client.focus(eid)
        elif action == "toggle":
            oculos_client.toggle(eid)
        elif action == "expand":
            oculos_client.expand(eid)
        elif action == "collapse":
            oculos_client.collapse(eid)
        elif action == "select":
            oculos_client.select(eid)
        elif action == "set_range" and range_value is not None:
            oculos_client.set_range(eid, range_value)
        elif action == "scroll" and scroll_direction is not None:
            oculos_client.scroll(eid, scroll_direction)
        elif action == "scroll_into_view":
            oculos_client.scroll_into_view(eid)
        elif action == "highlight":
            oculos_client.highlight(eid)
        else:
            raise ValueError(
                f"Invalid action '{action}' or missing required parameters."
            )

    def _post_state_str(eid: str) -> str:
        """Return element state as a plain string so it stays in the message field."""
        meta = _element_meta.get(eid) or _element_meta.get(element_id)
        if not meta or not meta.get("query"):
            return ""
        try:
            fresh = oculos_client.find_elements(
                meta["pid"],
                query=meta["query"],
                element_type=meta.get("element_type"),
            )
            if fresh:
                el = fresh[0]
                parts = [
                    f"{k}={el[k]}"
                    for k in (
                        "toggle_state",
                        "checked",
                        "value",
                        "is_selected",
                        "is_enabled",
                    )
                    if el.get(k) is not None
                ]
                return (" | " + ", ".join(parts)) if parts else ""
        except Exception:
            pass
        return ""

    # Attempt 1 — direct
    try:
        _do(element_id)
        _invalidate_discovery_cache()
        return {
            "status": "success",
            "message": f"Performed '{action}' on {element_id}.{_post_state_str(element_id)}",
        }
    except Exception as e:
        msg = str(e)

    # Attempt 2 — transient COM error, retry same ID
    com_transient = any(code in msg for code in ("0x80004005", "0x80040201"))
    if com_transient:
        try:
            _do(element_id)
            _invalidate_discovery_cache()
            return {
                "status": "success",
                "message": f"Performed '{action}' on {element_id} after COM retry.{_post_state_str(element_id)}",
            }
        except Exception as e2:
            msg = str(e2)

    # Attempt 3 — stale element: re-find by cached query and retry
    meta = _element_meta.get(element_id)
    if meta and meta.get("query"):
        try:
            fresh = oculos_client.find_elements(
                meta["pid"],
                query=meta["query"],
                element_type=meta.get("element_type"),
                interactive=True,
            )
            if fresh:
                fresh_id = fresh[0]["oculos_id"]
                _element_meta[fresh_id] = meta
                _do(fresh_id)
                _invalidate_discovery_cache()
                return {
                    "status": "success",
                    "message": f"Performed '{action}' after re-finding stale element.{_post_state_str(fresh_id)}",
                }
        except Exception as e3:
            msg = str(e3)

    return {"status": "error", "message": f"Interaction failed: {msg}"}


def click_first(
    pid: int,
    query: str,
    element_type: Optional[str] = "Button",
    interactive: bool = True,
    anchor_probe_query: Optional[str] = None,
    allow_browser_chrome: bool = False,
) -> Dict[str, Any]:
    """
    High-leverage composed action: optional short wait, then find once, then click once.

    This is the most efficient “default” pattern for advancing flows:
      (optional wait) -> find_ui_elements -> interact_with_element(click)

    Args:
        pid: window PID
        query: label/text to find
        element_type: defaults to Button to reduce search space
        interactive: require clickability
        anchor_probe_query: if provided, do a single cheap anchor probe before searching/clicking
        allow_browser_chrome: if True, permit clicks on browser chrome (bookmarks/tabs/address bar).
    """

    def _is_browser_chrome_element(el: Dict[str, Any]) -> bool:
        text = " ".join(
            str(el.get(k) or "")
            for k in ("name", "title", "label", "value", "text_content", "element_type")
        ).lower()
        chrome_markers = (
            "address and search bar",
            "bookmark",
            "bookmarks",
            "tab search",
            "new tab",
            "tab",
            "extensions",
            "profile",
            "chrome toolbar",
            "omnibox",
        )
        return any(marker in text for marker in chrome_markers)

    try:
        if anchor_probe_query:
            # Single cheap probe; not a real "wait".
            _cached_find_elements(pid, query=anchor_probe_query, interactive=True)

        found = _cached_find_elements(
            pid,
            query=query,
            element_type=element_type,
            interactive=interactive,
        )
        if not found:
            return {
                "status": "error",
                "message": f"No element found to click for query={query!r} type={element_type!r}.",
            }
        candidates = found
        if not allow_browser_chrome:
            filtered = [el for el in candidates if not _is_browser_chrome_element(el)]
            if filtered:
                candidates = filtered
            elif any(_is_browser_chrome_element(el) for el in found):
                return {
                    "status": "error",
                    "message": (
                        "Refusing to click browser chrome (bookmarks/tabs/address bar). "
                        "Use page-specific query or set allow_browser_chrome=True only when intentional."
                    ),
                    "matches_sample": [
                        (el.get("name") or el.get("title") or el.get("label") or "")
                        for el in found[:5]
                    ],
                }

        element_id = candidates[0].get("oculos_id")
        if not element_id:
            return {
                "status": "error",
                "message": "Matched element missing oculos_id.",
                "element": candidates[0],
            }
        oculos_client.click(str(element_id))
        _invalidate_discovery_cache()
        return {
            "status": "success",
            "message": f"Clicked first match for {query!r}.",
            "element_id": str(element_id),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def type_into(
    pid: int,
    field_query: str,
    text: str,
    verify: bool = False,
    element_type: str = "Edit",
    interactive: bool = True,
) -> Dict[str, Any]:
    """
    High-leverage composed action: find a text field once, set text, optionally verify.
    """
    try:
        found = _cached_find_elements(
            pid,
            query=field_query,
            element_type=element_type,
            interactive=interactive,
        )
        if not found:
            return {
                "status": "error",
                "message": f"Field not found for query={field_query!r} type={element_type!r}.",
            }
        element_id = found[0].get("oculos_id")
        if not element_id:
            return {
                "status": "error",
                "message": "Matched field missing oculos_id.",
                "element": found[0],
            }

        oculos_client.set_text(str(element_id), str(text))
        _invalidate_discovery_cache()

        if not verify:
            return {
                "status": "success",
                "message": f"Set text for {field_query!r}.",
                "element_id": str(element_id),
            }

        # Re-find the same element id and check value/text_content best-effort.
        refreshed = _cached_find_elements(
            pid, element_type=element_type, interactive=interactive
        )
        for el in refreshed or []:
            if str(el.get("oculos_id")) != str(element_id):
                continue
            v = el.get("value") or el.get("text_content") or ""
            ok = str(text).strip() in str(v)
            return {
                "status": "success" if ok else "warning",
                "message": (
                    "Typed and verified." if ok else "Typed but could not verify value."
                ),
                "element_id": str(element_id),
                "observed_value": str(v)[:200],
            }

        return {
            "status": "warning",
            "message": "Typed, but could not re-locate the same field to verify.",
            "element_id": str(element_id),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _nav_url_norm(u: str) -> str:
    u = u.strip().lower()
    for prefix in ("https://", "http://"):
        if u.startswith(prefix):
            u = u[len(prefix) :]
    return u.rstrip("/")


def navigate_to_url(pid: int, url: str) -> Dict[str, Any]:
    try:
        oculos_client.focus_window(pid)
        elements = oculos_client.find_elements(
            pid, query="Address and search bar", interactive=True
        )
        if not elements:
            return {"status": "error", "message": "Address bar not found."}

        address_bar_id = elements[0]["oculos_id"]
        oculos_client.click(address_bar_id)
        oculos_client.set_text(address_bar_id, url)
        oculos_client.send_keys(address_bar_id, "{ENTER}")

        # Wait for the address bar to reflect the target URL (navigation started),
        # then hold for a short render window. This collapses the common
        # navigate → wait_for_element pattern into a single tool call.
        target_norm = _nav_url_norm(url)[:50]
        deadline = time.time() + 8.0
        navigated = False
        while time.time() < deadline:
            time.sleep(0.25)
            try:
                bars = oculos_client.find_elements(
                    pid, query="Address and search bar", interactive=True
                )
                if bars:
                    bar_val = bars[0].get("value") or bars[0].get("text_content") or ""
                    if _nav_url_norm(bar_val)[:50].startswith(target_norm[:30]):
                        navigated = True
                        break
            except Exception:
                pass

        # Short render wait after URL appears (SPA hydration, dynamic content)
        time.sleep(0.8)
        _invalidate_discovery_cache()

        status = "navigated" if navigated else "navigation_sent"
        return {
            "status": "success",
            "message": f"Navigated to {url} ({status}) — page is ready.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def launch_and_get_pid(app_name: str) -> Dict[str, Any]:
    try:
        manage_window(action="launch", app_name=app_name)
        deadline = time.time() + 10.0
        while time.time() < deadline:
            result = list_active_windows()
            if result["status"] == "success" and result["windows"]:
                return {"status": "success", "windows": result["windows"]}
            await asyncio.sleep(0)
        return {
            "status": "error",
            "message": f"App {app_name} did not appear after launch.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def take_screenshot(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Takes a screenshot of the current screen for visual analysis.
    Use this when accessibility tools fail to find an element.
    After calling this, analyze the image and then continue using standard UI tools
    (find_ui_elements / get_window_tree / interact_with_element) to act.
    """
    try:
        _require_pyautogui()
        screenshot = pyautogui.screenshot()
        screenshot = screenshot.resize((768, 768))
        buffer = BytesIO()
        screenshot.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        if len(image_bytes) < 100:
            return {"status": "error", "message": "Screenshot encoding failed — empty or corrupt JPEG"}

        artifact = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        await tool_context.save_artifact(filename="screenshot.jpg", artifact=artifact)
        return {
            "status": "success",
            "message": "Screenshot saved.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def scroll_page(direction: str, amount: int = 3) -> Dict[str, Any]:
    """
    Scrolls the current browser page or any focused window.
    Use this when content is not visible and needs scrolling to find it.

    Args:
        direction (str): 'up' or 'down'
        amount (int): Number of scroll steps. Default 3.
    """
    try:
        _require_pyautogui()
        if direction == "down":
            pyautogui.scroll(-amount * 100)
        elif direction == "up":
            pyautogui.scroll(amount * 100)
        else:
            return {"status": "error", "message": "Direction must be 'up' or 'down'"}
        return {
            "status": "success",
            "message": f"Scrolled {direction} by {amount} steps.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_form_fields(pid: int) -> Dict[str, Any]:
    try:
        text_fields = oculos_client.find_elements(
            pid, interactive=True, element_type="Edit"
        )
        dropdowns = oculos_client.find_elements(
            pid, interactive=True, element_type="ComboBox"
        )
        checkboxes = oculos_client.find_elements(
            pid, interactive=True, element_type="CheckBox"
        )
        buttons = oculos_client.find_elements(
            pid, interactive=True, element_type="Button"
        )
        number_inputs = oculos_client.find_elements(
            pid, interactive=True, element_type="Spinner"
        )
        labels = oculos_client.find_elements(
            pid, interactive=False, element_type="Text"
        )
        radio_buttons = oculos_client.find_elements(
            pid, interactive=True, element_type="RadioButton"
        )

        return {
            "status": "success",
            "text_fields": text_fields,
            "dropdowns": dropdowns,
            "checkboxes": checkboxes,
            "buttons": buttons,
            "number_inputs": number_inputs,
            "labels": labels,  # so agent can read question text
            "radio_buttons": radio_buttons,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def select_dropdown_option(
    pid: int, dropdown_query: str, option: str
) -> Dict[str, Any]:
    """
    Select an option from a dropdown/ComboBox.

    This tool is hardened for web UIs (e.g., LinkedIn) where:
    - The dropdown label text may not match exactly (minor typos/punctuation).
    - Options may render in an overlay that is not part of the same PID tree.
    - A click can succeed without actually changing the dropdown value.
    """
    try:
        import re

        def _norm(s: str) -> str:
            s = (s or "").lower()
            s = re.sub(r"\s+", " ", s)
            s = re.sub(r"[^a-z0-9 ]+", "", s)
            return s.strip()

        def _token_score(a: str, b: str) -> float:
            a_toks = set(_norm(a).split())
            b_toks = set(_norm(b).split())
            if not a_toks or not b_toks:
                return 0.0
            return len(a_toks & b_toks) / max(1, len(b_toks))

        # 1) Find candidate dropdowns in this PID
        candidates = (
            oculos_client.find_elements(pid, interactive=True, element_type="ComboBox")
            or []
        )

        # Try exact-ish query first (keeps behavior when labels match well)
        direct = oculos_client.find_elements(
            pid, query=dropdown_query, interactive=True, element_type="ComboBox"
        )
        if direct:
            dropdown = direct[0]
        else:
            # 2) Fuzzy match by label/title token overlap
            best = None
            best_score = 0.0
            for c in candidates:
                label = c.get("label") or c.get("title") or c.get("name") or ""
                score = _token_score(label, dropdown_query)
                if score > best_score:
                    best_score = score
                    best = c

            if not best or best_score < 0.35:
                return {
                    "status": "error",
                    "message": f"Dropdown '{dropdown_query}' not found (best_match_score={round(best_score, 2)}).",
                    "available_dropdown_labels": [
                        (c.get("label") or c.get("title") or c.get("name") or "")
                        for c in candidates
                    ][:20],
                }
            dropdown = best

        dropdown_id = dropdown["oculos_id"]

        # Helper: check if dropdown value reflects `option`
        def _value_is_set() -> bool:
            refreshed = (
                oculos_client.find_elements(
                    pid, interactive=True, element_type="ComboBox"
                )
                or []
            )
            for c in refreshed:
                if c.get("oculos_id") == dropdown_id:
                    val = c.get("value") or c.get("text_content") or ""
                    return _norm(option) in _norm(str(val))
            return False

        def _find_opts():
            opts = oculos_client.find_elements(
                pid, query=option, interactive=True, element_type="ListItem"
            )
            return opts or oculos_client.find_elements(
                pid, query=option, interactive=True
            )

        for _attempt in range(3):
            oculos_client.click(dropdown_id)
            await wait_for_element(
                pid=pid,
                query=option,
                timeout=2,
                interval=0.2,
                max_polls=6,
                interactive=True,
            )

            opts = _find_opts()
            if opts:
                oculos_client.click(opts[0]["oculos_id"])
                if _value_is_set():
                    return {
                        "status": "success",
                        "message": f"Selected '{option}' from '{dropdown.get('label') or dropdown_query}'.",
                    }

            opts = oculos_client.find_elements(
                pid, query=option, interactive=True, element_type="ListItem"
            )
            if not opts:
                opts = oculos_client.find_elements(pid, query=option, interactive=True)
            if opts:
                oculos_client.click(opts[0]["oculos_id"])
                if _value_is_set():
                    return {
                        "status": "success",
                        "message": f"Selected '{option}' from '{dropdown.get('label') or dropdown_query}'.",
                    }

        # 4) If we couldn't verify selection, return diagnostics
        available_options = []
        items = (
            oculos_client.find_elements(pid, interactive=True, element_type="ListItem")
            or []
        )
        for i in items:
            name = i.get("name") or i.get("label") or i.get("title") or ""
            if name:
                available_options.append(name)

        return {
            "status": "error",
            "message": f"Could not select '{option}' for dropdown '{dropdown_query}' (value did not update).",
            "dropdown_label": dropdown.get("label")
            or dropdown.get("title")
            or dropdown.get("name"),
            "available_options_sample": available_options[:30],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def select_option_by_label(pid: int, label_text: str) -> Dict[str, Any]:
    """
    Selects an option that behaves like a radio/choice based on its visible label text.
    This works even when the control is implemented as a Button/ListItem instead of
    a true RadioButton, which is common on sites like LinkedIn.
    """
    try:
        elements = oculos_client.find_elements(
            pid,
            query=label_text,
            interactive=True,
        )
        if not elements:
            return {
                "status": "error",
                "message": f"No interactive element found with label '{label_text}'.",
            }

        target_id = elements[0]["oculos_id"]
        oculos_client.click(target_id)

        return {
            "status": "success",
            "message": f"Selected option with label '{label_text}'.",
            "element_id": target_id,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _collect_text(node: dict) -> list[str]:
    """Recursively collect all text content from an accessibility tree node."""
    texts = []
    for key in ("name", "title", "text_content", "value"):
        val = node.get(key)
        if val and isinstance(val, str) and val.strip():
            texts.append(val.strip())
    for child in node.get("children", []):
        texts.extend(_collect_text(child))
    return texts


def get_page_text(pid: int) -> Dict[str, Any]:
    """
    Extract all visible text from a window as a single string.

    This is much faster and cheaper than get_window_tree when you only need
    the text content (not element IDs or structure). Useful for reading page
    content, verifying text is present, or extracting data.

    Args:
        pid (int): The Process ID of the window.
    """
    try:
        raw_tree = oculos_client.get_tree(pid)
        all_texts = _collect_text(raw_tree)
        # Deduplicate adjacent repeated strings (common in a11y trees)
        deduped = []
        for t in all_texts:
            if not deduped or t != deduped[-1]:
                deduped.append(t)
        text = "\n".join(deduped)
        # Truncate to avoid blowing up context
        if len(text) > 15000:
            text = text[:15000] + "\n...[TRUNCATED]"
        return {"status": "success", "text": text, "length": len(text)}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get page text: {str(e)}"}


async def wait_for_text(pid: int, text: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Wait until specific text appears on screen in the given window.

    Polls the accessibility tree every ~1 second until the text is found
    or the timeout is reached. More efficient than the agent taking repeated
    screenshots to check for text.

    Args:
        pid (int): The Process ID of the window to watch.
        text (str): The text to wait for (case-insensitive substring match).
        timeout (int): Maximum seconds to wait (default: 10).
    """
    try:
        text_lower = text.lower()
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw_tree = oculos_client.get_tree(pid)
            all_text = "\n".join(_collect_text(raw_tree)).lower()
            if text_lower in all_text:
                return {
                    "status": "success",
                    "found": True,
                    "message": f"Text '{text}' appeared on screen.",
                }
            await asyncio.sleep(1.0)
        return {
            "status": "success",
            "found": False,
            "message": f"Text '{text}' did not appear within {timeout}s.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
