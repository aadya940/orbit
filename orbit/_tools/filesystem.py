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
from .clipboard import clipboard_set
from .hotkey import press_hotkey

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
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
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
        matches = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
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


def move_files(operations: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Moves multiple files or folders in one call. Each operation has "src" and "dst".
    Use this instead of many move_file calls when moving several files (e.g. into a folder).
    """
    results: List[Dict[str, Any]] = []
    for op in operations:
        src = op.get("src") or ""
        dst = op.get("dst") or ""
        try:
            shutil.move(src, dst)
            results.append({"src": src, "dst": dst, "status": "success"})
        except Exception as e:
            results.append({"src": src, "dst": dst, "status": "error", "message": str(e)})
    failed = [r for r in results if r.get("status") == "error"]
    return {
        "status": "success" if not failed else "partial",
        "count": len(results),
        "results": results,
        "message": f"Moved {len(results) - len(failed)} of {len(results)}." + (
            f" {len(failed)} failed." if failed else ""
        ),
    }


def create_directory_and_move(directory: str, src_paths: List[str]) -> Dict[str, Any]:
    """
    Creates the directory (and parents) then moves all given files/folders into it.
    Equivalent to create_directory(directory) then move_files([{src, dst} for each path]).
    """
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        return {"status": "error", "message": str(e)}
    operations = [
        {"src": p, "dst": str(Path(directory) / Path(p).name)} for p in src_paths
    ]
    return move_files(operations)


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


def _is_likely_address_bar(element: Dict[str, Any]) -> bool:
    """True if this looks like the path/address bar (typing here navigates). Must not paste path here."""
    name = (element.get("name") or element.get("title") or "").strip()
    name_lower = name.lower()
    # Explicit labels
    if any(s in name_lower for s in ("address", "location", "path", "search", "go to")):
        return True
    # Path-like: current path or breadcrumb (e.g. "<< projects freebsd-contrib src" or "C:\Users\...")
    if "\\" in name or "/" in name or name.startswith("<<"):
        return True
    # Multiple path-segment-like tokens (e.g. "projects freebsd-contrib src") with no "file name"
    if "file name" not in name_lower and len(name.split()) >= 2:
        return True
    return False


def _find_file_name_field(dialog_pid: int) -> str | None:
    """
    Find the bottom "File name:" text input in the Windows file dialog. Never return the address bar.
    Prefer Edit over ComboBox: focusing the ComboBox often focuses the dropdown arrow first, so
    Ctrl+V/Enter then act on the dropdown and close the dialog without pasting the path.
    """
    # Prefer Edit (the actual text box); avoid ComboBox (has dropdown arrow, focus goes there first)
    for query, etype in [
        ("File name:", "Edit"),
        ("File name", "Edit"),
        ("File name:", "ComboBox"),
        ("File name", "ComboBox"),
    ]:
        inputs = oculos_client.find_elements(
            dialog_pid, query=query, element_type=etype, interactive=True
        )
        for el in reversed(inputs):
            if not _is_likely_address_bar(el):
                return el["oculos_id"]

    # Fallback: any Edit or ComboBox with "file name" in name (and not address bar)
    for etype in ("Edit", "ComboBox"):
        for el in oculos_client.find_elements(
            dialog_pid, element_type=etype, interactive=True
        ):
            name = (el.get("name") or el.get("title") or "").lower()
            if "file name" in name and not _is_likely_address_bar(el):
                return el["oculos_id"]

    return None


def _find_open_button(dialog_pid: int) -> str | None:
    """
    Find the main 'Open' button (the one that submits the file), not the dropdown arrow.
    Skip buttons with automation_id 'DropDown' or very small rect (dropdown arrows).
    """
    buttons = oculos_client.find_elements(
        dialog_pid,
        query="Open",
        element_type="Button",
        interactive=True,
    )
    for b in buttons:
        aid = (b.get("automation_id") or "").strip()
        rect = b.get("rect") or {}
        w = rect.get("width") or 0
        if aid == "DropDown" or w < 30:
            continue
        return b["oculos_id"]
    if buttons:
        return buttons[-1]["oculos_id"]

    buttons = oculos_client.find_elements(
        dialog_pid,
        element_type="Button",
        interactive=True,
    )
    for b in buttons:
        if (b.get("automation_id") or "").strip() == "DropDown":
            continue
        rect = b.get("rect") or {}
        if (rect.get("width") or 0) >= 50 and "cancel" not in (
            b.get("name") or b.get("title") or ""
        ).lower():
            return b["oculos_id"]
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

        oculos_client.focus_window(dialog_pid)

        # 3. Locate the 'File name' input field (not the address bar)
        field_id = _find_file_name_field(dialog_pid)
        if not field_id:
            return {"status": "error", "message": "File name field not found"}

        # 4. Put path in the "File name" box: clipboard + system-level paste so the dialog
        #    actually receives it (element send_keys for ^v can fail in native dialogs).
        clipboard_set(path)
        oculos_client.focus(field_id)
        press_hotkey("ctrl+a")
        press_hotkey("ctrl+v")

        # 5. Submit with Enter (from the focused field)
        press_hotkey("enter")
        closed = await _wait_for_dialog_close(timeout=8.0)
        if not closed:
            open_button_id = _find_open_button(dialog_pid)
            if open_button_id:
                oculos_client.click(open_button_id)
            else:
                press_hotkey("enter")
            closed = await _wait_for_dialog_close(timeout=8.0)

        if not closed:
            return {
                "status": "error",
                "message": "File dialog did not close after submitting path (tried Enter and Open button)",
            }

        return {
            "status": "success",
            "message": f"{filename} submitted via file dialog",
        }

    except Exception as e:
        logger.exception("[upload_file]")
        return {"status": "error", "message": str(e)}
