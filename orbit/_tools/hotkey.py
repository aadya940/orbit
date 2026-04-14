import time
import threading
from typing import Dict, Any

try:
    import pyautogui
    _PYAUTOGUI_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - environment-dependent (e.g. headless CI)
    pyautogui = None
    _PYAUTOGUI_IMPORT_ERROR = e

KEYBOARD_LOCK = threading.Lock()

def press_hotkey(keys: str) -> Dict[str, Any]:
    """
    Presses a keyboard shortcut or key combination.
    Use this for system-level shortcuts that don't have a UI element.

    Args:
        keys (str): Key combination e.g. 'ctrl+c', 'alt+tab',
                    'ctrl+shift+t', 'win+d', 'alt+f4'
    """
    try:
        if pyautogui is None:
            raise RuntimeError(
                "pyautogui is unavailable in this environment "
                f"(import error: {_PYAUTOGUI_IMPORT_ERROR!r})"
            )
        
        parts = keys.lower().split("+")
        
        # Wrap execution in the lock to prevent LLM parallel-call collisions
        with KEYBOARD_LOCK:
            pyautogui.hotkey(*parts)
            time.sleep(0.05)
            
        return {"status": "success", "message": f"Pressed {keys}."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to press hotkey: {str(e)}"}

def type_text(text: str) -> Dict[str, Any]:
    """
    Types a string of text sequentially. 
    Use this tool whenever you need to type words, numbers, or sentences into an input field.
    Do NOT use press_hotkey for typing text.
    
    Args:
        text (str): The exact string of text to type.
    """
    try:
        if pyautogui is None:
            raise RuntimeError(
                "pyautogui is unavailable in this environment "
                f"(import error: {_PYAUTOGUI_IMPORT_ERROR!r})"
            )
            
        with KEYBOARD_LOCK:
            # pyautogui.typewrite does not handle \n — split on newlines
            # and press Enter between segments so newlines are typed correctly.
            segments = text.split('\n')
            for i, segment in enumerate(segments):
                if segment:
                    pyautogui.typewrite(segment, interval=0.05)
                if i < len(segments) - 1:
                    pyautogui.press('enter')
            time.sleep(0.05)
            
        return {"status": "success", "message": f"Typed: {text}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to type text: {str(e)}"}