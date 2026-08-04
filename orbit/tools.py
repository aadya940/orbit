"""Tools the agent can call alongside screen actions.

Screen verbs (click/fill/observe) let the agent operate a UI; these let it
do the surrounding work — read and write files, run code, use the
clipboard, search the web — so a task like "extract these invoices and
save them as CSV" is one run instead of two.

Adding your own is a decorator and a function:

    from orbit.tools import tool

    @tool("send_slack", "Post a message to Slack", {"text": {"type": "string"}})
    async def send_slack(text: str) -> str:
        ...
        return "sent"

    async with orbit.session(tools=[send_slack]) as s: ...

Every tool returns a string (what the model sees). Raising is fine —
the error text goes back to the model, which can adapt.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_MAX_OUTPUT = 20_000  # keep a single tool result from swamping the context


@dataclass
class Tool:
    name: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    fn: Optional[Callable] = None

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.params,
                    "required": self.required,
                },
            },
        }

    async def call(self, **kwargs: Any) -> str:
        if self.fn is None:
            raise RuntimeError(f"tool {self.name!r} has no implementation")
        result = self.fn(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        if len(text) > _MAX_OUTPUT:
            text = text[:_MAX_OUTPUT] + f"\n[... truncated, {len(text)} chars total]"
        return text


def tool(name: str, description: str,
         params: Optional[Dict[str, Any]] = None,
         required: Optional[List[str]] = None) -> Callable:
    """Decorator turning a function into an agent-callable Tool."""
    def wrap(fn: Callable) -> Tool:
        return Tool(
            name=name,
            description=description,
            params=params or {},
            required=required if required is not None else list((params or {}).keys()),
            fn=fn,
        )
    return wrap


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

@tool("read_file", "Read a text file and return its contents.",
      {"path": {"type": "string", "description": "file path"},
       "max_chars": {"type": "integer", "description": "cap on returned characters"}},
      required=["path"])
def read_file(path: str, max_chars: int = _MAX_OUTPUT) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"no such file: {p}"
    if p.is_dir():
        return f"{p} is a directory — use list_dir"
    text = p.read_text(errors="replace")
    return text[:max_chars]


@tool("write_file", "Write text to a file, creating parent directories. Overwrites.",
      {"path": {"type": "string"}, "content": {"type": "string"}})
def write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {p}"


@tool("append_file", "Append text to a file, creating it if needed.",
      {"path": {"type": "string"}, "content": {"type": "string"}})
def append_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(content)
    return f"appended {len(content)} chars to {p}"


@tool("list_dir", "List files and folders in a directory.",
      {"path": {"type": "string"}}, required=[])
def list_dir(path: str = ".") -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"no such directory: {p}"
    entries = []
    for item in sorted(p.iterdir()):
        kind = "dir " if item.is_dir() else "file"
        size = "" if item.is_dir() else f"  {item.stat().st_size}b"
        entries.append(f"{kind} {item.name}{size}")
    return "\n".join(entries) or "(empty)"


@tool("find_files", "Find files matching a glob pattern, recursively.",
      {"pattern": {"type": "string", "description": "e.g. '**/*.csv'"},
       "path": {"type": "string"}},
      required=["pattern"])
def find_files(pattern: str, path: str = ".") -> str:
    root = Path(path).expanduser()
    hits = [str(p) for p in root.glob(pattern)][:500]
    return "\n".join(hits) or f"no files match {pattern}"


@tool("move_file", "Move or rename a file or directory.",
      {"src": {"type": "string"}, "dst": {"type": "string"}})
def move_file(src: str, dst: str) -> str:
    s, d = Path(src).expanduser(), Path(dst).expanduser()
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(d))
    return f"moved {s} -> {d}"


@tool("read_csv", "Read a CSV file as rows (list of dicts).",
      {"path": {"type": "string"}, "limit": {"type": "integer"}},
      required=["path"])
def read_csv(path: str, limit: int = 200) -> str:
    import csv

    p = Path(path).expanduser()
    if not p.exists():
        return f"no such file: {p}"
    with p.open(newline="") as fh:
        rows = list(csv.DictReader(fh))[:limit]
    return json.dumps(rows, default=str)


@tool("write_csv", "Write rows (list of dicts, as JSON) to a CSV file.",
      {"path": {"type": "string"},
       "rows": {"type": "string", "description": "JSON array of objects"}})
def write_csv(path: str, rows: str) -> str:
    import csv

    data = json.loads(rows) if isinstance(rows, str) else rows
    if not data:
        return "no rows to write"
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    return f"wrote {len(data)} rows to {p}"


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------

@tool("run_python", "Run Python code and return stdout. Use for data work, "
                    "parsing, math, and file processing.",
      {"code": {"type": "string"}, "timeout": {"type": "integer"}},
      required=["code"])
async def run_python(code: str, timeout: int = 60) -> str:
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", code,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return f"timed out after {timeout}s"
    text = out.decode(errors="replace").strip()
    return text or f"(no output, exit code {proc.returncode})"


@tool("run_command", "Run a shell command and return its output.",
      {"command": {"type": "string"}, "timeout": {"type": "integer"}},
      required=["command"])
async def run_command(command: str, timeout: int = 60) -> str:
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return f"timed out after {timeout}s"
    text = out.decode(errors="replace").strip()
    return text or f"(no output, exit code {proc.returncode})"


# ---------------------------------------------------------------------------
# Clipboard
# ---------------------------------------------------------------------------

def _clipboard():
    try:
        import pyperclip  # lazy
        return pyperclip
    except Exception as exc:
        raise RuntimeError(f"clipboard unavailable ({exc}); pip install pyperclip") from exc


@tool("clipboard_read", "Read the system clipboard.", {}, required=[])
def clipboard_read() -> str:
    return _clipboard().paste() or "(clipboard empty)"


@tool("clipboard_write", "Put text on the system clipboard.",
      {"text": {"type": "string"}})
def clipboard_write(text: str) -> str:
    _clipboard().copy(text)
    return f"copied {len(text)} chars to clipboard"


# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------

@tool("web_search", "Search the web and return result titles, urls and snippets. "
                    "Faster than browsing when you just need a fact or a link.",
      {"query": {"type": "string"}, "max_results": {"type": "integer"}},
      required=["query"])
def web_search(query: str, max_results: int = 8) -> str:
    try:
        try:
            from ddgs import DDGS  # newer package name
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore
    except ImportError:
        return "web search unavailable: pip install ddgs"
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=max_results))
    if not hits:
        return f"no results for {query!r}"
    return "\n\n".join(
        f"{h.get('title','')}\n{h.get('href','')}\n{h.get('body','')}" for h in hits
    )


@tool("fetch_url", "Fetch a URL and return its text content (no browser needed).",
      {"url": {"type": "string"}}, required=["url"])
async def fetch_url(url: str) -> str:
    import urllib.request

    def _get() -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode(errors="replace")
        import re
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    return await asyncio.to_thread(_get)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DEFAULT_TOOLS: List[Tool] = [
    read_file, write_file, append_file, list_dir, find_files, move_file,
    read_csv, write_csv,
    run_python, run_command,
    clipboard_read, clipboard_write,
    web_search, fetch_url,
]


def default_tools() -> List[Tool]:
    """The standard tool set given to every session."""
    return list(DEFAULT_TOOLS)


def build_registry(extra: Optional[List[Tool]] = None,
                   include_defaults: bool = True) -> Dict[str, Tool]:
    """Name -> Tool, with user tools overriding defaults of the same name."""
    registry: Dict[str, Tool] = {}
    if include_defaults:
        for t in default_tools():
            registry[t.name] = t
    for t in extra or []:
        registry[t.name] = t
    return registry
