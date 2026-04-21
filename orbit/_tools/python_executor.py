"""PythonExecutor — run arbitrary Python code in an isolated subprocess.

Intended to be passed as an extra tool to orbit agents (via ``extra_tools``),
giving the LLM the ability to execute Python code without needing a browser or
accessibility session.

Usage in generated workflows::

    from orbit import Do
    from orbit._tools.python_executor import run_python

    result = await Do("analyse the CSV", ..., extra_tools=[run_python]).run()

Or import the convenience alias::

    from orbit import PythonExecutor
    # PythonExecutor is the run_python function, ready to pass as an extra_tool.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


def run_python(code: str) -> dict:
    """Execute Python code in a subprocess and return its output.

    Use this tool when you need to run arbitrary Python code — for data
    processing, file manipulation, calculations, or anything that benefits
    from a real Python runtime rather than string manipulation.

    Args:
        code: Valid Python source code to execute. Multi-line strings are
              supported. The code runs in a fresh interpreter with no shared
              state from the current session. Use ``print()`` to produce
              output that will be returned in ``stdout``.

    Returns:
        A dict with keys:
          - ``stdout``   (str): captured standard output.
          - ``stderr``   (str): captured standard error.
          - ``returncode`` (int): process exit code (0 = success).
          - ``success``  (bool): True when returncode == 0.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "success": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Execution timed out after 120 seconds.",
            "returncode": -1,
            "success": False,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "stdout": "",
            "stderr": str(exc),
            "returncode": -1,
            "success": False,
        }


# Convenience alias — pass this directly to extra_tools=[]
PythonExecutor = run_python
