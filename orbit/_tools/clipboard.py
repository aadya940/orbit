import shutil
import subprocess
from typing import Dict, Any

import pyperclip


def _clip_exe_available() -> bool:
    return shutil.which("clip.exe") is not None


def _xclip_available() -> bool:
    return shutil.which("xclip") is not None


def _use_xclip() -> bool:
    """
    Returns True when we should use xclip instead of pyperclip.

    pyperclip detects Docker-on-Windows containers as WSL (because /proc/version
    contains "microsoft") and tries clip.exe — which doesn't exist in the container.
    We only use xclip when clip.exe is genuinely unreachable and xclip is installed.
    On real WSL, Windows, or macOS this returns False and pyperclip handles it as normal.
    """
    return not _clip_exe_available() and _xclip_available()


def _xclip_copy(text: str) -> None:
    result = subprocess.run(
        ["xclip", "-selection", "clipboard"],
        input=text.encode("utf-8"),
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"xclip copy failed: {result.stderr.decode('utf-8', errors='replace')}"
        )


def _xclip_paste() -> str:
    result = subprocess.run(
        ["xclip", "-selection", "clipboard", "-o"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"xclip paste failed: {result.stderr.decode('utf-8', errors='replace')}"
        )
    return result.stdout.decode("utf-8")


def clipboard_get() -> Dict[str, Any]:
    """
    Reads the current content of the system clipboard.
    Use this to retrieve text that was copied by the user or by a previous action.
    """
    try:
        content = _xclip_paste() if _use_xclip() else pyperclip.paste()
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
        if _use_xclip():
            _xclip_copy(text)
        else:
            pyperclip.copy(text)
        return {"status": "success", "message": "Clipboard set successfully."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to set clipboard: {str(e)}"}
