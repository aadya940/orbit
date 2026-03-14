import pyperclip
from typing import Dict, Any


def clipboard_get() -> Dict[str, Any]:
    """
    Reads the current content of the system clipboard.
    Use this to retrieve text that was copied by the user or by a previous action.
    """
    try:
        content = pyperclip.paste()
        return {"status": "success", "content": content}
    except Exception as e:
        return {"status": "error", "message": f"Failed to read clipboard: {str(e)}"}


def clipboard_set(text: str) -> Dict[str, Any]:
    """
    Writes text to the system clipboard.
    Use this to prepare text for pasting into any application.

    Args:
        text (str): The text to copy to the clipboard.
    """
    try:
        pyperclip.copy(text)
        return {"status": "success", "message": f"Clipboard set successfully."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to set clipboard: {str(e)}"}
