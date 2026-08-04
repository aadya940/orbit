"""Tools the agent can call alongside screen actions.

Screen verbs (click/fill/observe) let the agent operate a UI; these let it
do the surrounding work: read and write files, run code, use the
clipboard, search the web. A task like "extract these invoices and
save them as CSV" is one run instead of two.

Adding your own is a decorator and a function:

    from orbit.tools import tool

    @tool("send_slack", "Post a message to Slack", {"text": {"type": "string"}})
    async def send_slack(text: str) -> str:
        ...
        return "sent"

    async with orbit.session(tools=[send_slack]) as s: ...

Every tool returns a string (what the model sees). Raising is fine:
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
    """An agent-callable tool: a name, a JSON schema and an implementation.

    Attributes
    ----------
    name : str
        Name the model uses to call the tool.
    description : str
        One-line explanation shown to the model.
    params : Dict[str, Any]
        JSON Schema properties for the tool's arguments.
    required : List[str]
        Names of arguments the model must supply.
    fn : Optional[Callable]
        Implementation, sync or async, returning a value the model
        reads. ``None`` means the tool is declared but not implemented.

    Examples
    --------
    >>> t = Tool(name="ping", description="Ping.", fn=lambda: "pong")
    >>> await t.call()
    'pong'

    Notes
    -----
    Results are truncated so a single tool call cannot swamp the model's
    context. Custom tools registered via :func:`tool` override defaults
    of the same name.
    """

    name: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    fn: Optional[Callable] = None

    def schema(self) -> dict:
        """Render the tool as an OpenAI-style function schema.

        Returns
        -------
        dict
            Function-calling schema with the tool's name, description
            and parameter object, ready to send to the model.

        Examples
        --------
        >>> Tool(name="ping", description="Ping.").schema()["function"]["name"]
        'ping'
        """
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
        """Invoke the tool and return its result as text for the model.

        Awaits the implementation if it is a coroutine, serialises
        non-string results as JSON, and truncates long output.

        Parameters
        ----------
        **kwargs : Any
            Arguments forwarded to the implementation, normally the
            arguments the model supplied in its tool call.

        Returns
        -------
        str
            The tool's output as text. If it exceeds the internal cap
            the text is cut and a note giving the full length is
            appended.

        Raises
        ------
        RuntimeError
            If the tool has no implementation attached.

        Examples
        --------
        >>> await Tool(name="add", description="Add.",
        ...            fn=lambda a, b: a + b).call(a=1, b=2)
        '3'

        Notes
        -----
        Results are truncated so a single tool call cannot swamp the
        model's context. Exceptions raised inside the implementation are
        not caught here: the caller returns them to the model as text so
        it can adapt rather than the run dying.
        """
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
    """Decorator turning a function into an agent-callable Tool.

    Parameters
    ----------
    name : str
        Name the model uses to call the tool.
    description : str
        One-line explanation shown to the model.
    params : Optional[Dict[str, Any]], optional
        JSON Schema properties for the arguments. Default is ``None``,
        meaning the tool takes no arguments.
    required : Optional[List[str]], optional
        Names of arguments the model must supply. Default is ``None``,
        which treats every declared parameter as required.

    Returns
    -------
    Callable
        Decorator that replaces the decorated function with a
        :class:`Tool`.

    Examples
    --------
    >>> @tool("send_slack", "Post a message.", {"text": {"type": "string"}})
    ... async def send_slack(text: str) -> str:
    ...     return "sent"

    Notes
    -----
    Custom tools registered via ``@tool`` override defaults of the same
    name. Their results are truncated so a single tool call cannot swamp
    the model's context, and exceptions they raise are returned to the
    model as text so it can adapt rather than the run dying.
    """
    def wrap(fn: Callable) -> Tool:
        """Build a :class:`Tool` from the decorated function.

        Parameters
        ----------
        fn : Callable
            Function to expose to the model, sync or async, returning a
            string.

        Returns
        -------
        Tool
            Tool carrying the decorator's name, description and schema,
            bound to ``fn``.

        Examples
        --------
        >>> @tool("ping", "Ping.", {})
        ... def ping() -> str:
        ...     return "pong"
        >>> ping.name
        'ping'

        Notes
        -----
        When no explicit ``required`` list is given, every declared
        parameter is treated as required. Tools built this way and
        passed to a session override defaults of the same name.
        """
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
    """Read a text file and hand its contents to the model.

    Parameters
    ----------
    path : str
        Path to the file. A leading ``~`` is expanded.
    max_chars : int, optional
        Cap on how many characters are returned. Default is the module
        output cap.

    Returns
    -------
    str
        The file text up to ``max_chars``, or an explanatory message the
        model can act on if the path is missing or is a directory.

    Examples
    --------
    >>> read_file.fn("notes.txt", max_chars=20)
    'first twenty chars..'

    Notes
    -----
    Results are truncated so a single tool call cannot swamp the model's
    context. If this raises, the error text is returned to the model so
    it can adapt rather than the run dying. A custom tool registered via
    ``@tool`` under this name overrides it.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return f"no such file: {p}"
    if p.is_dir():
        return f"{p} is a directory, use list_dir"
    text = p.read_text(errors="replace")
    return text[:max_chars]


@tool("write_file", "Write text to a file, creating parent directories. Overwrites.",
      {"path": {"type": "string"}, "content": {"type": "string"}})
def write_file(path: str, content: str) -> str:
    """Write text to a file, creating parent directories.

    Any existing file at the path is overwritten.

    Parameters
    ----------
    path : str
        Destination path. A leading ``~`` is expanded.
    content : str
        Text to write.

    Returns
    -------
    str
        Confirmation naming the number of characters written and the
        resolved path, which is what the model reads back.

    Examples
    --------
    >>> write_file.fn("out/report.txt", "hello")
    'wrote 5 chars to out/report.txt'

    Notes
    -----
    Results are truncated so a single tool call cannot swamp the model's
    context. If this raises, the error text is returned to the model so
    it can adapt rather than the run dying. A custom tool registered via
    ``@tool`` under this name overrides it.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {p}"


@tool("append_file", "Append text to a file, creating it if needed.",
      {"path": {"type": "string"}, "content": {"type": "string"}})
def append_file(path: str, content: str) -> str:
    """Append text to a file, creating it if needed.

    Parameters
    ----------
    path : str
        Destination path. A leading ``~`` is expanded.
    content : str
        Text to append.

    Returns
    -------
    str
        Confirmation naming the number of characters appended and the
        resolved path, which is what the model reads back.

    Examples
    --------
    >>> append_file.fn("out/log.txt", "line\n")
    'appended 5 chars to out/log.txt'

    Notes
    -----
    Results are truncated so a single tool call cannot swamp the model's
    context. If this raises, the error text is returned to the model so
    it can adapt rather than the run dying. A custom tool registered via
    ``@tool`` under this name overrides it.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(content)
    return f"appended {len(content)} chars to {p}"


@tool("list_dir", "List files and folders in a directory.",
      {"path": {"type": "string"}}, required=[])
def list_dir(path: str = ".") -> str:
    """List the files and folders in a directory.

    Parameters
    ----------
    path : str, optional
        Directory to list. Default is ``"."``.

    Returns
    -------
    str
        One line per entry, each tagged ``dir`` or ``file`` with a byte
        size for files, or ``(empty)`` for an empty directory, or a
        message if the path does not exist.

    Examples
    --------
    >>> list_dir.fn("out")
    'file report.txt  5b'

    Notes
    -----
    Results are truncated so a single tool call cannot swamp the model's
    context. If this raises, the error text is returned to the model so
    it can adapt rather than the run dying. A custom tool registered via
    ``@tool`` under this name overrides it.
    """
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
    """Find files matching a glob pattern, recursively.

    Parameters
    ----------
    pattern : str
        Glob pattern relative to ``path``, for example ``'**/*.csv'``.
    path : str, optional
        Root directory to search from. Default is ``"."``.

    Returns
    -------
    str
        Newline separated paths, capped at the first 500 matches, or a
        message saying nothing matched.

    Examples
    --------
    >>> find_files.fn("**/*.csv", "data")
    'data/2024/sales.csv'

    Notes
    -----
    Results are truncated so a single tool call cannot swamp the model's
    context. If this raises, the error text is returned to the model so
    it can adapt rather than the run dying. A custom tool registered via
    ``@tool`` under this name overrides it.
    """
    root = Path(path).expanduser()
    hits = [str(p) for p in root.glob(pattern)][:500]
    return "\n".join(hits) or f"no files match {pattern}"


@tool("move_file", "Move or rename a file or directory.",
      {"src": {"type": "string"}, "dst": {"type": "string"}})
def move_file(src: str, dst: str) -> str:
    """Move or rename a file or directory.

    Parent directories of the destination are created as needed.

    Parameters
    ----------
    src : str
        Existing path to move.
    dst : str
        Destination path.

    Returns
    -------
    str
        Confirmation naming the source and destination paths, which is
        what the model reads back.

    Examples
    --------
    >>> move_file.fn("a.txt", "archive/a.txt")
    'moved a.txt -> archive/a.txt'

    Notes
    -----
    Results are truncated so a single tool call cannot swamp the model's
    context. If this raises, the error text is returned to the model so
    it can adapt rather than the run dying. A custom tool registered via
    ``@tool`` under this name overrides it.
    """
    s, d = Path(src).expanduser(), Path(dst).expanduser()
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(d))
    return f"moved {s} -> {d}"


@tool("read_csv", "Read a CSV file as rows (list of dicts).",
      {"path": {"type": "string"}, "limit": {"type": "integer"}},
      required=["path"])
def read_csv(path: str, limit: int = 200) -> str:
    """Read a CSV file and return its rows as JSON.

    Parameters
    ----------
    path : str
        Path to the CSV file. A leading ``~`` is expanded.
    limit : int, optional
        Maximum number of rows to return. Default is 200.

    Returns
    -------
    str
        JSON array of row objects keyed by column name, or a message if
        the file does not exist.

    Examples
    --------
    >>> read_csv.fn("sales.csv", limit=1)
    '[{"item": "pen", "qty": "3"}]'

    Notes
    -----
    Results are truncated so a single tool call cannot swamp the model's
    context. If this raises, the error text is returned to the model so
    it can adapt rather than the run dying. A custom tool registered via
    ``@tool`` under this name overrides it.
    """
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
    """Write rows given as JSON to a CSV file.

    Column headers are taken from the keys of the first row.

    Parameters
    ----------
    path : str
        Destination path. Parent directories are created as needed.
    rows : str
        JSON array of objects. An already-decoded list is also accepted.

    Returns
    -------
    str
        Confirmation naming the row count and resolved path, or a note
        that there was nothing to write.

    Examples
    --------
    >>> write_csv.fn("out.csv", '[{"item": "pen", "qty": 3}]')
    'wrote 1 rows to out.csv'

    Notes
    -----
    Results are truncated so a single tool call cannot swamp the model's
    context. If this raises, the error text is returned to the model so
    it can adapt rather than the run dying. A custom tool registered via
    ``@tool`` under this name overrides it.
    """
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
    """Run Python code in a subprocess and return its output.

    Standard error is merged into standard output so the model sees
    tracebacks as well as results.

    Parameters
    ----------
    code : str
        Source passed to the interpreter with ``-c``.
    timeout : int, optional
        Seconds to wait before killing the process. Default is 60.

    Returns
    -------
    str
        Combined output, stripped. If nothing was printed, a note giving
        the exit code. If the timeout is hit, a note saying so.

    Examples
    --------
    >>> await run_python.fn("print(2 + 2)")
    '4'

    Notes
    -----
    Results are truncated so a single tool call cannot swamp the model's
    context. If this raises, the error text is returned to the model so
    it can adapt rather than the run dying. A custom tool registered via
    ``@tool`` under this name overrides it.
    """
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
    """Run a shell command and return its output.

    Standard error is merged into standard output so the model sees
    failures as well as results.

    Parameters
    ----------
    command : str
        Command line executed through the system shell.
    timeout : int, optional
        Seconds to wait before killing the process. Default is 60.

    Returns
    -------
    str
        Combined output, stripped. If nothing was printed, a note giving
        the exit code. If the timeout is hit, a note saying so.

    Examples
    --------
    >>> await run_command.fn("echo hi")
    'hi'

    Notes
    -----
    Results are truncated so a single tool call cannot swamp the model's
    context. If this raises, the error text is returned to the model so
    it can adapt rather than the run dying. A custom tool registered via
    ``@tool`` under this name overrides it.
    """
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
    """Import the clipboard backend lazily.

    Returns
    -------
    module
        The ``pyperclip`` module.

    Raises
    ------
    RuntimeError
        If ``pyperclip`` cannot be imported, carrying install advice.

    Examples
    --------
    >>> _clipboard().copy("hi")  # doctest: +SKIP

    Notes
    -----
    The import is deferred so sessions that never touch the clipboard do
    not need the dependency. The raised error reaches the model as text
    so it can adapt rather than the run dying.
    """
    try:
        import pyperclip  # lazy
        return pyperclip
    except Exception as exc:
        raise RuntimeError(f"clipboard unavailable ({exc}); pip install pyperclip") from exc


@tool("clipboard_read", "Read the system clipboard.", {}, required=[])
def clipboard_read() -> str:
    """Read the system clipboard.

    Returns
    -------
    str
        The clipboard text, or ``(clipboard empty)`` when there is
        nothing on it.

    Examples
    --------
    >>> clipboard_read.fn()
    'copied text'

    Notes
    -----
    Results are truncated so a single tool call cannot swamp the model's
    context. If this raises, the error text is returned to the model so
    it can adapt rather than the run dying. A custom tool registered via
    ``@tool`` under this name overrides it.
    """
    return _clipboard().paste() or "(clipboard empty)"


@tool("clipboard_write", "Put text on the system clipboard.",
      {"text": {"type": "string"}})
def clipboard_write(text: str) -> str:
    """Put text on the system clipboard.

    Parameters
    ----------
    text : str
        Text to copy.

    Returns
    -------
    str
        Confirmation naming the number of characters copied, which is
        what the model reads back.

    Examples
    --------
    >>> clipboard_write.fn("hello")
    'copied 5 chars to clipboard'

    Notes
    -----
    Results are truncated so a single tool call cannot swamp the model's
    context. If this raises, the error text is returned to the model so
    it can adapt rather than the run dying. A custom tool registered via
    ``@tool`` under this name overrides it.
    """
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
    """Search the web and return titles, urls and snippets.

    Faster than driving a browser when the agent just needs a fact or a
    link.

    Parameters
    ----------
    query : str
        Search query.
    max_results : int, optional
        Maximum number of results to return. Default is 8.

    Returns
    -------
    str
        Blank-line separated blocks of title, url and snippet, or a
        message saying there were no results or that the search
        dependency is missing.

    Examples
    --------
    >>> web_search.fn("orbit agent framework", max_results=1)
    'Orbit\nhttps://example.com\nAn agent framework.'

    Notes
    -----
    Results are truncated so a single tool call cannot swamp the model's
    context. If this raises, the error text is returned to the model so
    it can adapt rather than the run dying. A custom tool registered via
    ``@tool`` under this name overrides it.
    """
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
    """Fetch a URL and return its text content, without a browser.

    Scripts, styles and tags are stripped and whitespace collapsed, so
    the model gets readable prose rather than markup.

    Parameters
    ----------
    url : str
        Absolute URL to fetch.

    Returns
    -------
    str
        The page text with markup removed.

    Examples
    --------
    >>> await fetch_url.fn("https://example.com")
    'Example Domain This domain is for use in examples.'

    Notes
    -----
    Results are truncated so a single tool call cannot swamp the model's
    context. If this raises, the error text is returned to the model so
    it can adapt rather than the run dying. A custom tool registered via
    ``@tool`` under this name overrides it.
    """
    import urllib.request

    def _get() -> str:
        """Fetch and strip the page synchronously, off the event loop.

        Returns
        -------
        str
            The page text with scripts, styles and tags removed and
            whitespace collapsed.

        Examples
        --------
        >>> _get()  # doctest: +SKIP
        'Example Domain'

        Notes
        -----
        Kept separate so the blocking request can run in a worker
        thread. Any error it raises propagates out of the tool and is
        returned to the model as text so it can adapt rather than the
        run dying.
        """
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
    """Return the standard tool set given to every session.

    Returns
    -------
    List[Tool]
        A fresh list of the default tools, so callers can mutate it
        without disturbing the module-level set.

    Examples
    --------
    >>> sorted(t.name for t in default_tools())[:2]
    ['append_file', 'clipboard_read']

    Notes
    -----
    Every one of these returns a string the model reads, truncated so a
    single tool call cannot swamp the model's context.
    """
    return list(DEFAULT_TOOLS)


def build_registry(extra: Optional[List[Tool]] = None,
                   include_defaults: bool = True) -> Dict[str, Tool]:
    """Build the name to Tool registry a session hands to the agent.

    Parameters
    ----------
    extra : Optional[List[Tool]], optional
        User tools to register. Default is ``None``.
    include_defaults : bool, optional
        Whether to seed the registry with the standard tool set. Default
        is ``True``.

    Returns
    -------
    Dict[str, Tool]
        Mapping of tool name to tool.

    Examples
    --------
    >>> sorted(build_registry([], include_defaults=False))
    []

    Notes
    -----
    Extras are applied last, so custom tools registered via ``@tool``
    override defaults of the same name.
    """
    registry: Dict[str, Tool] = {}
    if include_defaults:
        for t in default_tools():
            registry[t.name] = t
    for t in extra or []:
        registry[t.name] = t
    return registry
