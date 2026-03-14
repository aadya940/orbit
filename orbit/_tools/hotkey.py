import pyautogui
from typing import Dict, Any


def press_hotkey(keys: str) -> Dict[str, Any]:
    """
    Presses a keyboard shortcut or key combination.
    Use this for system-level shortcuts that don't have a UI element.

    Args:
        keys (str): Key combination e.g. 'ctrl+c', 'alt+tab',
                    'ctrl+shift+t', 'win+d', 'alt+f4'
    """
    try:
        parts = keys.lower().split("+")
        pyautogui.hotkey(*parts)
        return {"status": "success", "message": f"Pressed {keys}."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to press hotkey: {str(e)}"}
