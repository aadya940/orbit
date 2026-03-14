import asyncio
import os
import shutil
import glob
import csv
import mimetypes
from pathlib import Path
from typing import Any, Dict, List
import pdfplumber
import platform
import logging
import time
import send2trash

from .ui import oculos_client

logger = logging.getLogger(__name__)

def get_system_info() -> Dict[str, Any]:
    """
    Returns system information including the current user's Desktop path,
    username, and home directory. Call this once at the start of any task
    that involves saving files or navigating the file system.
    """
    try:
        username = os.getlogin()
        home = str(Path.home())
        desktop = str(Path.home() / "Desktop")
        return {
            "status": "success",
            "username": username,
            "home": home,
            "desktop": desktop,
            "os": platform.system(),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def read_file(path: str) -> Dict[str, Any]:
    """Reads the content of a text file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"status": "success", "content": f.read()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def write_file(path: str, content: str) -> Dict[str, Any]:
    """Writes content to a file. Creates the file if it doesn't exist."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"status": "success", "message": f"File written to {path}."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def append_to_file(path: str, content: str) -> Dict[str, Any]:
    """
    Appends content to an existing file without overwriting it.
    Creates the file if it doesn't exist.
    """
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        return {"status": "success", "message": f"Content appended to {path}."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def read_pdf(path: str) -> Dict[str, Any]:
    """Reads and extracts text from a PDF file. Always use this for .pdf files."""
    try:
        with pdfplumber.open(path) as pdf:
            text = "\n".join(
                page.extract_text() or ""
                for page in pdf.pages
            )
        return {"status": "success", "content": text}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def read_csv(path: str) -> Dict[str, Any]:
    """
    Reads a CSV file and returns its content as a list of dictionaries.
    Each dictionary represents a row with column headers as keys.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return {"status": "success", "rows": rows, "count": len(rows)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def write_csv(path: str, headers: List[str], rows: List[List[str]]) -> Dict[str, Any]:
    """
    Writes data to a CSV file.

    Args:
        path (str): Output file path.
        headers (list): Column header names e.g. ['name', 'email', 'role']
        rows (list): List of rows, each row is a list of values e.g. [['John', 'j@x.com', 'Engineer']]
    """
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        return {"status": "success", "message": f"CSV written to {path}."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def list_directory(path: str) -> Dict[str, Any]:
    """Lists all files and folders in a directory."""
    try:
        entries = os.listdir(path)
        files = [f for f in entries if os.path.isfile(os.path.join(path, f))]
        folders = [f for f in entries if os.path.isdir(os.path.join(path, f))]
        return {"status": "success", "files": files, "folders": folders}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def search_files(directory: str, pattern: str) -> Dict[str, Any]:
    """
    Searches for files matching a pattern in a directory and subdirectories.

    Args:
        directory (str): Root directory to search in.
        pattern (str): Glob pattern e.g. '*.pdf', '*.txt', 'resume*'
    """
    try:
        matches = glob.glob(
            os.path.join(directory, "**", pattern),
            recursive=True
        )
        return {"status": "success", "matches": matches, "count": len(matches)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def file_exists(path: str) -> Dict[str, Any]:
    """Checks whether a file or directory exists at the given path."""
    exists = os.path.exists(path)
    return {"status": "success", "exists": exists, "path": path}


def get_file_info(path: str) -> Dict[str, Any]:
    """
    Returns metadata about a file including size, type, and timestamps.
    """
    try:
        stat = os.stat(path)
        mime_type, _ = mimetypes.guess_type(path)
        return {
            "status": "success",
            "path": path,
            "size_bytes": stat.st_size,
            "mime_type": mime_type,
            "extension": Path(path).suffix,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def copy_file(src: str, dst: str) -> Dict[str, Any]:
    """
    Copies a file from src to dst.
    If dst is a directory, the file is copied into it with the same name.
    """
    try:
        shutil.copy2(src, dst)
        return {"status": "success", "message": f"Copied {src} to {dst}."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def move_file(src: str, dst: str) -> Dict[str, Any]:
    """Moves or renames a file or folder."""
    try:
        shutil.move(src, dst)
        return {"status": "success", "message": f"Moved {src} to {dst}."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def delete_file(path: str) -> Dict[str, Any]:
    """
    Moves a file to the system trash (recoverable). Uses send2trash so the file
    is not permanently removed — user can restore from Trash/Recycle Bin.

    Args:
        path (str): Path to the file to delete (send to trash).
    """
    try:
        send2trash.send2trash(path)
        return {"status": "success", "message": f"Moved {path} to trash (recoverable)."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_directory(path: str) -> Dict[str, Any]:
    """
    Creates a directory and all necessary parent directories.
    Does not fail if the directory already exists.
    """
    try:
        os.makedirs(path, exist_ok=True)
        return {"status": "success", "message": f"Directory created at {path}."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def find_in_file(path: str, query: str) -> Dict[str, Any]:
    """
    Searches for a string inside a text file and returns matching lines.

    Args:
        path (str): Path to the file.
        query (str): Text to search for.
    """
    try:
        matches = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if query.lower() in line.lower():
                    matches.append({"line": i, "content": line.strip()})
        return {"status": "success", "matches": matches, "count": len(matches)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def wait_for_file_dialog(timeout=10):
    """Wait for Windows file dialog and return its PID. Polls until condition (no fixed sleep)."""
    start = time.time()
    while time.time() - start < timeout:
        windows = oculos_client.list_windows()
        for w in windows:
            title = (w.get("title") or "").lower()
            if title == "open":
                return w["pid"]
        await asyncio.sleep(0)
    return None


def _is_file_dialog_window(win: Dict[str, Any]) -> bool:
    """
    Heuristic to detect a file-open dialog window.
    Loosens the previous strict 'title == open' check so we also
    catch variants like 'Open File', 'Select file to upload', etc.
    """
    title = (win.get("title") or "").lower().strip()
    if not title:
        return False

    keywords = [
        "open",
        "open file",
        "select file",
        "choose file",
        "file upload",
    ]

    return any(kw in title for kw in keywords)


async def _wait_for_dialog_open(timeout: float = 10.0) -> int | None:
    """Wait for a file dialog window to appear and return its PID. Polls until condition (no fixed sleep)."""
    start = time.time()
    while time.time() - start < timeout:
        for w in oculos_client.list_windows():
            if _is_file_dialog_window(w):
                return w["pid"]
        await asyncio.sleep(0)
    return None


async def _wait_for_dialog_close(timeout: float = 10.0) -> bool:
    """Wait until all file dialog windows are gone. Polls until condition (no fixed sleep)."""
    start = time.time()
    while time.time() - start < timeout:
        any_dialog = False
        for w in oculos_client.list_windows():
            if _is_file_dialog_window(w):
                any_dialog = True
                break
        if not any_dialog:
            return True
        await asyncio.sleep(0)
    return False


def _find_file_name_field(dialog_pid: int) -> str | None:
    """
    Robust search for the 'File name' input in the dialog.
    Falls back to the last interactive Edit/ComboBox if label-based search fails.
    """
    # First, try label + control type combinations
    for query, etype in [
        ("File name:", "ComboBox"),
        ("File name:", "Edit"),
        ("File name", "Edit"),
    ]:
        inputs = oculos_client.find_elements(
            dialog_pid,
            query=query,
            element_type=etype,
            interactive=True,
        )
        if inputs:
            return inputs[0]["oculos_id"]

    # Fallback: last interactive Edit/ComboBox in this dialog
    for etype in ("Edit", "ComboBox"):
        inputs = oculos_client.find_elements(
            dialog_pid,
            element_type=etype,
            interactive=True,
        )
        if inputs:
            return inputs[-1]["oculos_id"]

    return None


def _find_open_button(dialog_pid: int) -> str | None:
    """
    Robust search for the 'Open' button in the file dialog.
    Falls back to the first interactive Button if label-based search fails.
    """
    buttons = oculos_client.find_elements(
        dialog_pid,
        query="Open",
        element_type="Button",
        interactive=True,
    )
    if buttons:
        return buttons[0]["oculos_id"]

    # Fallback: any interactive button (often the default 'Open' action)
    buttons = oculos_client.find_elements(
        dialog_pid,
        element_type="Button",
        interactive=True,
    )
    if buttons:
        return buttons[0]["oculos_id"]

    return None


async def upload_file(element_id: str, path: str) -> Dict[str, Any]:
    """
    Clicks an upload button, interacts with the Windows file dialog to select `path`,
    and waits until the dialog is fully closed before returning.
    """
    try:
        if not os.path.exists(path):
            return {"status": "error", "message": f"File not found: {path}"}

        path = os.path.abspath(path)
        filename = os.path.basename(path)

        # 1. Click the upload button on the web page / app
        oculos_client.click(element_id)

        # 2. Wait for the file dialog to appear (poll until condition)
        dialog_pid = await _wait_for_dialog_open(timeout=10.0)
        if not dialog_pid:
            return {"status": "error", "message": "File dialog not detected"}

        # 3. Locate the 'File name' input field
        field_id = _find_file_name_field(dialog_pid)
        if not field_id:
            return {"status": "error", "message": "File name field not found"}

        # 4. Focus the field and type the file path (no fixed sleeps; one yield before close check)
        oculos_client.focus(field_id)
        await asyncio.sleep(0)
        oculos_client.send_keys(field_id, "^a")
        oculos_client.send_keys(field_id, "{BACKSPACE}")
        await asyncio.sleep(0)
        oculos_client.send_keys(field_id, path)
        await asyncio.sleep(0)

        # 5. Click the 'Open' button or press ENTER as a fallback
        open_button_id = _find_open_button(dialog_pid)
        if open_button_id:
            oculos_client.click(open_button_id)
        else:
            oculos_client.send_keys(field_id, "{ENTER}")

        # 6. Wait until the dialog is fully closed (poll until condition)
        if not await _wait_for_dialog_close(timeout=10.0):
            return {
                "status": "error",
                "message": "File dialog did not close after submitting path",
            }

        return {
            "status": "success",
            "message": f"{filename} submitted via file dialog",
        }

    except Exception as e:
        logger.exception("[upload_file]")
        return {"status": "error", "message": str(e)}
