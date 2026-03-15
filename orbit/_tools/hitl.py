"""Human-in-the-loop and Approval tools. Only disk I/O (write) tools require approval."""

from typing import Any, Callable, Dict, List

from . import filesystem as _fs


def _pending(tool: str, **kwargs: Any) -> Dict[str, Any]:
    return {"status": "pending", "tool": tool, **kwargs}


def write_file(path: str, content: str) -> Dict[str, Any]:
    """Writes content to a file. Creates the file if it doesn't exist. Requires approval."""
    return _pending("write_file", path=path, content=content)


def append_to_file(path: str, content: str) -> Dict[str, Any]:
    """Appends content to an existing file. Creates the file if it doesn't exist. Requires approval."""
    return _pending("append_to_file", path=path, content=content)


def write_csv(path: str, headers: List[str], rows: List[List[str]]) -> Dict[str, Any]:
    """Writes data to a CSV file. Requires approval."""
    return _pending("write_csv", path=path, headers=headers, rows=rows)


def copy_file(src: str, dst: str) -> Dict[str, Any]:
    """Copies a file from src to dst. Requires approval."""
    return _pending("copy_file", src=src, dst=dst)


def move_file(src: str, dst: str) -> Dict[str, Any]:
    """Moves or renames a file or folder. Requires approval."""
    return _pending("move_file", src=src, dst=dst)


def move_files(operations: List[Dict[str, str]]) -> Dict[str, Any]:
    """Moves multiple files/folders in one call (each op has "src" and "dst"). Requires approval."""
    return _pending("move_files", operations=operations)


def create_directory_and_move(directory: str, src_paths: List[str]) -> Dict[str, Any]:
    """Creates a directory then moves all given paths into it. Requires approval."""
    return _pending("create_directory_and_move", directory=directory, src_paths=src_paths)


def delete_file(path: str) -> Dict[str, Any]:
    """Moves a file to the system trash (recoverable). Requires approval."""
    return _pending("delete_file", path=path)


def create_directory(path: str) -> Dict[str, Any]:
    """Creates a directory and all necessary parent directories. Requires approval."""
    return _pending("create_directory", path=path)


def upload_file(element_id: str, path: str) -> Dict[str, Any]:
    """Clicks an upload button and selects the file at path via the file dialog. Requires approval."""
    return _pending("upload_file", element_id=element_id, path=path)


def request_human(
    description: str, context: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Ask a human to complete something the agent cannot do (e.g. CAPTCHA, login, or a blocked step).
    Use when automation has failed or the task requires human intervention.
    """
    return {
        "status": "pending",
        "tool": "request_human",
        "description": description,
        "context": context or {},
    }


# Registry of real tool implementations. Runner calls these when user approves.
# Only disk I/O (write) tools; read-only (ls, read_file, etc.) are direct in agents.
APPROVAL_TOOLS: Dict[str, Callable[..., Any]] = {
    "write_file": _fs.write_file,
    "append_to_file": _fs.append_to_file,
    "write_csv": _fs.write_csv,
    "copy_file": _fs.copy_file,
    "move_file": _fs.move_file,
    "move_files": _fs.move_files,
    "create_directory_and_move": _fs.create_directory_and_move,
    "delete_file": _fs.delete_file,
    "create_directory": _fs.create_directory,
    "upload_file": _fs.upload_file,
}
