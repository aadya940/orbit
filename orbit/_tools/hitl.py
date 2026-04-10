"""Human-in-the-loop and Approval tools. Only disk I/O (write) tools require approval."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List

from . import filesystem as _fs
from .._ui.toast import run_toast_ui

# Single-thread executor so tkinter toasts never overlap.
_toast_pool = ThreadPoolExecutor(max_workers=1)

# When False, all approval gates are bypassed and tools run autonomously.
# Set via set_hitl_enabled() before each agent run.
_hitl_enabled: bool = True


def set_hitl_enabled(enabled: bool) -> None:
    global _hitl_enabled
    _hitl_enabled = enabled


async def _confirm_and_run(
    tool: str,
    impl: Callable[..., Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Async HITL approval: runs the tkinter toast on a dedicated thread
    (safe on Windows), then executes impl (sync or async).
    When _hitl_enabled is False, skips approval and runs directly.
    """
    if not _hitl_enabled:
        try:
            if asyncio.iscoroutinefunction(impl):
                return await impl(**kwargs)
            return impl(**kwargs)
        except Exception as e:
            return {"status": "error", "message": str(e), "tool": tool}

    loop = asyncio.get_running_loop()
    decision = await loop.run_in_executor(
        _toast_pool,
        run_toast_ui,
        "approval",
        {"tool": tool, **kwargs},
    )
    if decision.get("status") != "approved":
        return {"status": "rejected", "message": "Rejected by user.", "tool": tool}
    try:
        if asyncio.iscoroutinefunction(impl):
            return await impl(**kwargs)
        return impl(**kwargs)
    except Exception as e:
        return {"status": "error", "message": str(e), "tool": tool}


async def write_file(path: str, content: str) -> Dict[str, Any]:
    """Writes content to a file. Creates the file if it doesn't exist. Requires approval."""
    return await _confirm_and_run(
        "write_file", _fs.write_file, path=path, content=content
    )


async def append_to_file(path: str, content: str) -> Dict[str, Any]:
    """Appends content to an existing file. Creates the file if it doesn't exist. Requires approval."""
    return await _confirm_and_run(
        "append_to_file", _fs.append_to_file, path=path, content=content
    )


async def write_csv(
    path: str, headers: List[str], rows: List[List[str]]
) -> Dict[str, Any]:
    """Writes data to a CSV file. Requires approval."""
    return await _confirm_and_run(
        "write_csv", _fs.write_csv, path=path, headers=headers, rows=rows
    )


async def copy_file(src: str, dst: str) -> Dict[str, Any]:
    """Copies a file from src to dst. Requires approval."""
    return await _confirm_and_run("copy_file", _fs.copy_file, src=src, dst=dst)


async def move_file(src: str, dst: str) -> Dict[str, Any]:
    """Moves or renames a file or folder. Requires approval."""
    return await _confirm_and_run("move_file", _fs.move_file, src=src, dst=dst)


async def move_files(operations: List[Dict[str, str]]) -> Dict[str, Any]:
    """Moves multiple files/folders in one call (each op has "src" and "dst"). Requires approval."""
    return await _confirm_and_run("move_files", _fs.move_files, operations=operations)


async def create_directory_and_move(
    directory: str, src_paths: List[str]
) -> Dict[str, Any]:
    """Creates a directory then moves all given paths into it. Requires approval."""
    return await _confirm_and_run(
        "create_directory_and_move",
        _fs.create_directory_and_move,
        directory=directory,
        src_paths=src_paths,
    )


async def delete_file(path: str) -> Dict[str, Any]:
    """Moves a file to the system trash (recoverable). Requires approval."""
    return await _confirm_and_run("delete_file", _fs.delete_file, path=path)


async def create_directory(path: str) -> Dict[str, Any]:
    """Creates a directory and all necessary parent directories. Requires approval."""
    return await _confirm_and_run("create_directory", _fs.create_directory, path=path)


async def upload_file(element_id: str, path: str) -> Dict[str, Any]:
    """Clicks an upload button and selects the file at path via the file dialog. Requires approval."""
    return await _confirm_and_run(
        "upload_file", _fs.upload_file, element_id=element_id, path=path
    )


async def request_human(
    description: str, context: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Ask a human to complete something the agent cannot do (e.g. CAPTCHA, login, or a blocked step).
    Use when automation has failed or the task requires human intervention.
    """
    if not _hitl_enabled:
        return {"status": "completed", "message": "Autonomous mode — proceeding without human input."}

    ctx = context or {}
    # Always show the toast when the agent asks for help. The description
    # alone is reason enough — no special context keys required.
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _toast_pool,
        run_toast_ui,
        "help",
        {"tool": "request_human", "description": description, "context": ctx},
    )
    if result.get("status") == "completed":
        return {"status": "completed", "message": result.get("message", "Done")}
    return {"status": "rejected", "message": result.get("message", "Cancelled")}


def _run_shell_impl(command: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute a shell command and return its output."""
    import subprocess
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if len(output) > 10000:
            output = output[:10000] + "\n...[TRUNCATED]"
        stderr = result.stderr
        if len(stderr) > 3000:
            stderr = stderr[:3000] + "\n...[TRUNCATED]"
        return {
            "status": "success" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "stdout": output,
            "stderr": stderr,
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": f"Command timed out after {timeout}s."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def run_shell(command: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute a shell command and return stdout/stderr. Requires human approval.

    Use this for tasks like: checking installed software, running scripts,
    installing packages, listing processes, or any command-line operation.

    Args:
        command (str): The shell command to execute (e.g. 'ls -la', 'pip install requests').
        timeout (int): Maximum seconds to wait for the command to complete (default: 30).
    """
    return await _confirm_and_run(
        "run_shell", _run_shell_impl, command=command, timeout=timeout
    )


# Registry kept for backwards compat / runner reference.
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
