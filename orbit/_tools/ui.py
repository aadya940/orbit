import asyncio
import subprocess
import platform
import os
import time

import pyautogui
import base64
from io import BytesIO

from google.adk.tools.tool_context import ToolContext
from google.genai import types

from typing import Optional, Any, Dict, List
from orbit._oculus_client import OculOS

oculos_client = OculOS()

def list_active_windows() -> Dict[str, Any]:
    """
    Retrieves a list of all currently visible desktop windows.
    Use this to find the Process ID (pid) of the application you want to control.

    Returns:
        dict: A list of dictionaries containing 'pid' and 'title' for each window.
    """
    try:
        windows = oculos_client.list_windows()
        return {"status": "success", "windows": windows}
    except Exception as e:
        return {"status": "error", "message": f"Failed to list windows: {str(e)}"}


async def wait_for_element(
    pid: int,
    query: str,
    timeout: int = 3,
    interval: float = 0.5,
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

    while time.perf_counter() - start < timeout:
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

        if query_lower and query != query_lower:
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

        await asyncio.sleep(0)

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
        "last_poll_message": last_result.get("message", "")[:200] if last_result else None,
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
        elements = oculos_client.find_elements(
            pid, query=query, element_type=element_type, interactive=interactive
        )
        if not elements:
            return {
                "status": "success",
                "message": "No elements found matching the criteria.",
                "elements": [],
            }
        return {"status": "success", "elements": elements}
    except Exception as e:
        return {"status": "error", "message": f"Failed to find elements: {str(e)}"}


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
    def _do() -> None:
        if action == "click":
            oculos_client.click(element_id)
        elif action == "set_text" and text_input is not None:
            oculos_client.set_text(element_id, text_input)
        elif action == "send_keys" and text_input is not None:
            oculos_client.send_keys(element_id, text_input)
        elif action == "focus":
            oculos_client.focus(element_id)
        elif action == "toggle":
            oculos_client.toggle(element_id)
        elif action == "expand":
            oculos_client.expand(element_id)
        elif action == "collapse":
            oculos_client.collapse(element_id)
        elif action == "select":
            oculos_client.select(element_id)
        elif action == "set_range" and range_value is not None:
            oculos_client.set_range(element_id, range_value)
        elif action == "scroll" and scroll_direction is not None:
            oculos_client.scroll(element_id, scroll_direction)
        elif action == "scroll_into_view":
            oculos_client.scroll_into_view(element_id)
        elif action == "highlight":
            oculos_client.highlight(element_id)
        else:
            raise ValueError(f"Invalid action '{action}' or missing required parameters.")

    try:
        _do()
        return {
            "status": "success",
            "message": f"Successfully performed '{action}' on element {element_id}.",
        }
    except Exception as e:
        # UIA/COM transient failures can happen if the page rerenders between
        # element discovery and interaction (common on LinkedIn).
        msg = str(e)
        transient = any(code in msg for code in ("0x80004005", "0x80040201"))
        if transient:
            try:
                await asyncio.sleep(0)
                _do()
                return {
                    "status": "success",
                    "message": f"Successfully performed '{action}' on element {element_id}' after retry.",
                }
            except Exception as e2:
                return {"status": "error", "message": f"Interaction failed: {str(e2)}"}
        return {"status": "error", "message": f"Interaction failed: {msg}"}

def navigate_to_url(pid: int, url: str) -> Dict[str, Any]:
    try:
        oculos_client.focus_window(pid)
        elements = oculos_client.find_elements(
            pid,
            query="Address and search bar",
            interactive=True
        )
        if not elements:
            return {"status": "error", "message": "Address bar not found."}
        
        address_bar_id = elements[0]["oculos_id"]
        
        # Click to focus, select all existing content, replace with new URL
        oculos_client.click(address_bar_id)
        oculos_client.set_text(address_bar_id, url)  # set_text replaces entirely
        oculos_client.send_keys(address_bar_id, "{ENTER}")
        
        return {
            "status": "success",
            "message": f"Navigated to {url}. Call wait_for_element to confirm page loaded. Do NOT call press_hotkey or interact_with_element after this."
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
        return {"status": "error", "message": f"App {app_name} did not appear after launch."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def take_screenshot(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Takes a screenshot of the current screen for visual analysis.
    Use this when accessibility tools fail to find an element.
    After calling this, analyze the image and use mouse_click(x, y) to interact.
    """
    try:
        screenshot = pyautogui.screenshot()
        screenshot = screenshot.resize((768, 768))
        buffer = BytesIO()
        screenshot.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        artifact = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        )
        await tool_context.save_artifact(
            filename="screenshot.jpg",
            artifact=artifact
        )
        return {
            "status": "success",
            "message": "Screenshot saved. Analyze it and use mouse_click(x, y) to interact."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def mouse_click(x: int, y: int) -> Dict[str, Any]:
    """
    Clicks at specific screen coordinates.
    Only use after take_screenshot to know exact coordinates.

    Args:
        x (int): X coordinate in pixels.
        y (int): Y coordinate in pixels.
    """
    try:
        pyautogui.click(x, y)
        return {"status": "success", "message": f"Clicked at ({x}, {y})."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def mouse_type(text: str) -> Dict[str, Any]:
    """
    Types text at the current cursor position.
    Always call mouse_click first to focus the right element.

    Args:
        text (str): Text to type.
    """
    try:
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        return {"status": "success", "message": f"Typed text successfully."}
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
        if direction == "down":
            pyautogui.scroll(-amount * 100)
        elif direction == "up":
            pyautogui.scroll(amount * 100)
        else:
            return {"status": "error", "message": "Direction must be 'up' or 'down'"}
        return {
            "status": "success",
            "message": f"Scrolled {direction} by {amount} steps."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_form_fields(pid: int) -> Dict[str, Any]:
    try:
        text_fields = oculos_client.find_elements(pid, interactive=True, element_type="Edit")
        dropdowns = oculos_client.find_elements(pid, interactive=True, element_type="ComboBox")
        checkboxes = oculos_client.find_elements(pid, interactive=True, element_type="CheckBox")
        buttons = oculos_client.find_elements(pid, interactive=True, element_type="Button")
        number_inputs = oculos_client.find_elements(pid, interactive=True, element_type="Spinner")
        labels = oculos_client.find_elements(pid, interactive=False, element_type="Text")
        radio_buttons = oculos_client.find_elements(pid, interactive=True, element_type="RadioButton")

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


async def select_dropdown_option(pid: int, dropdown_query: str, option: str) -> Dict[str, Any]:
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
        candidates = oculos_client.find_elements(pid, interactive=True, element_type="ComboBox") or []

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
            refreshed = oculos_client.find_elements(pid, interactive=True, element_type="ComboBox") or []
            for c in refreshed:
                if c.get("oculos_id") == dropdown_id:
                    val = (c.get("value") or c.get("text_content") or "")
                    return _norm(option) in _norm(str(val))
            return False

        # 3) Try selecting with verification; poll until conditions (no fixed sleeps)
        async def _poll_until(check, timeout: float = 2.0) -> bool:
            deadline = time.time() + timeout
            while time.time() < deadline:
                if check():
                    return True
                await asyncio.sleep(0)
            return False

        def _find_opts():
            opts = oculos_client.find_elements(
                pid, query=option, interactive=True, element_type="ListItem"
            )
            return opts or oculos_client.find_elements(pid, query=option, interactive=True)

        for _attempt in range(3):
            oculos_client.click(dropdown_id)
            await _poll_until(_find_opts, timeout=2.0)

            opts = _find_opts()
            if opts:
                oculos_client.click(opts[0]["oculos_id"])
                if await _poll_until(_value_is_set, timeout=2.0):
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
                if await _poll_until(_value_is_set, timeout=2.0):
                    return {
                        "status": "success",
                        "message": f"Selected '{option}' from '{dropdown.get('label') or dropdown_query}'.",
                    }

        # 4) If we couldn't verify selection, return diagnostics
        available_options = []
        items = oculos_client.find_elements(pid, interactive=True, element_type="ListItem") or []
        for i in items:
            name = i.get("name") or i.get("label") or i.get("title") or ""
            if name:
                available_options.append(name)

        return {
            "status": "error",
            "message": f"Could not select '{option}' for dropdown '{dropdown_query}' (value did not update).",
            "dropdown_label": dropdown.get("label") or dropdown.get("title") or dropdown.get("name"),
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