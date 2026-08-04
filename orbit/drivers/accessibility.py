"""AccessibilityDriver — OS accessibility tree via the OculOS daemon.

Transplanted from orbit v1:
- OculOSClient: thin REST wrapper (orbit/_oculus_client/client.py)
- OculOSDaemon: subprocess lifecycle + health-check retry loop
  (orbit/daemon.py), including the Linux toolkit-accessibility fix that
  makes Chrome expose its full AT-SPI tree
- element interaction with stale-element re-find retry, browser-chrome
  filtering, and multiline typing via keyboard (orbit/_tools/ui.py)
- app launch with per-app accessibility flags (ui.py manage_window)

No TTL cache: observe() always fetches fresh (the old 0.75s cache was
deliberately dropped; event-driven invalidation is future work).

All heavy imports (requests, pyautogui) are lazy.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..types import (
    Action,
    ActionKind,
    Bounds,
    Element,
    Observation,
    Source,
    SurfaceUnreadable,
    OrbitError,
    TargetNotFound,
    TargetObstructed,
)
from .base import is_web_target
from .matching import best_match, rank_matches, suggestions

log = logging.getLogger("orbit.drivers.accessibility")

_DEFAULT_BASE_URL = "http://127.0.0.1:7878"


def _default_binary_path() -> Path:
    """Locate the bundled OculOS binary.

    Package-relative first (how it ships), then an installed sibling
    package, then PATH — so a source checkout, a wheel install and a
    system install all work without configuration.
    """
    name = "oculos.exe" if os.name == "nt" else "oculos"
    package_bin = Path(__file__).resolve().parents[1] / "_bin" / name
    if package_bin.exists():
        return package_bin
    for parent in Path(__file__).resolve().parents[2:4]:
        candidate = parent / "orbit" / "_bin" / name
        if candidate.exists():
            return candidate
    found = shutil.which(name)
    return Path(found) if found else package_bin


def _kill_pid(pid: Optional[int]) -> None:
    """SIGTERM then SIGKILL a real process id (fallback when the daemon's
    window-close is unavailable, e.g. xdotool not installed)."""
    if not pid:
        return
    import signal
    try:
        os.kill(int(pid), signal.SIGTERM)
        time.sleep(0.3)
        os.kill(int(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, ValueError):
        pass


class OculOSError(OrbitError):
    """Raised when the OculOS API returns an error.

    Subclasses OrbitError so the fallback ladder treats daemon-side
    failures as a failed rung to escalate past, not a run-killing crash.
    """

    code = "oculos_error"


class OculOSClient:
    """Thin wrapper around the OculOS REST API (localhost daemon)."""

    def __init__(self, base_url: str = _DEFAULT_BASE_URL, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session: Any = None

    def _get_session(self) -> Any:
        if self._session is None:
            import requests  # lazy

            self._session = requests.Session()
        return self._session

    # -- discovery --

    def list_windows(self) -> List[dict]:
        return self._get("/windows")

    def get_tree(self, pid: int) -> dict:
        return self._get(f"/windows/{pid}/tree")

    def find_elements(
        self,
        pid: int,
        *,
        query: Optional[str] = None,
        element_type: Optional[str] = None,
        interactive: Optional[bool] = None,
    ) -> List[dict]:
        params: Dict[str, Any] = {}
        if query is not None:
            params["q"] = query
        if element_type is not None:
            params["type"] = element_type
        if interactive is not None:
            params["interactive"] = str(interactive).lower()
        return self._get(f"/windows/{pid}/find", params=params)

    # -- window ops --

    def focus_window(self, pid: int) -> None:
        self._post(f"/windows/{pid}/focus")

    def close_window(self, pid: int) -> None:
        self._post(f"/windows/{pid}/close")

    # -- element interactions --

    def click(self, element_id: str) -> dict:
        return self._post(f"/interact/{element_id}/click")

    def set_text(self, element_id: str, text: str) -> dict:
        return self._post(f"/interact/{element_id}/set-text", json={"text": text})

    def send_keys(self, element_id: str, keys: str) -> dict:
        return self._post(f"/interact/{element_id}/send-keys", json={"keys": keys})

    def focus(self, element_id: str) -> dict:
        return self._post(f"/interact/{element_id}/focus")

    def toggle(self, element_id: str) -> dict:
        return self._post(f"/interact/{element_id}/toggle")

    def expand(self, element_id: str) -> dict:
        return self._post(f"/interact/{element_id}/expand")

    def select(self, element_id: str) -> dict:
        return self._post(f"/interact/{element_id}/select")

    def scroll(self, element_id: str, direction: str) -> dict:
        return self._post(f"/interact/{element_id}/scroll", json={"direction": direction})

    def scroll_into_view(self, element_id: str) -> dict:
        return self._post(f"/interact/{element_id}/scroll-into-view")

    def health(self) -> dict:
        return self._get("/health")

    # -- internals --

    def _unwrap(self, r: Any) -> Any:
        try:
            body = r.json()
        except ValueError:
            r.raise_for_status()
            raise OculOSError(f"HTTP {r.status_code}: non-JSON response")
        if not body.get("success"):
            raise OculOSError(body.get("error", f"HTTP {r.status_code}"))
        return body.get("data")

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        s = self._get_session()
        return self._unwrap(
            s.get(f"{self.base_url}{path}", params=params, timeout=self._timeout)
        )

    def _post(self, path: str, json: Optional[dict] = None) -> Any:
        s = self._get_session()
        return self._unwrap(
            s.post(f"{self.base_url}{path}", json=json, timeout=self._timeout)
        )


class OculOSDaemon:
    """Manages the OculOS background daemon subprocess."""

    def __init__(
        self,
        binary_path: Optional[str] = None,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self.binary_path = Path(binary_path).resolve() if binary_path else _default_binary_path()
        self.base_url = base_url
        self.process: Optional[subprocess.Popen] = None
        atexit.register(self.stop)

    async def start(self) -> None:
        if not self.binary_path.exists():
            raise FileNotFoundError(f"OculOS binary not found at: {self.binary_path}")

        # On Linux, ensure toolkit-accessibility is enabled so apps like
        # Chrome expose their full AT-SPI tree (otherwise Chrome returns
        # only ~5 top-level elements).
        if os.name != "nt":
            try:
                subprocess.run(
                    ["gsettings", "set", "org.gnome.desktop.interface",
                     "toolkit-accessibility", "true"],
                    capture_output=True,
                    timeout=5,
                )
            except FileNotFoundError:
                log.debug("gsettings not found — skipping toolkit-accessibility check")
            except subprocess.TimeoutExpired:
                log.warning("gsettings timed out setting toolkit-accessibility")
            except Exception as exc:
                log.debug("could not set toolkit-accessibility: %s", exc)

        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0  # type: ignore[attr-defined]
        self.process = subprocess.Popen(
            [str(self.binary_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        await self._wait_for_health(timeout_seconds=5.0)

    async def _wait_for_health(self, timeout_seconds: float = 5.0) -> None:
        """Poll /health until the daemon is ready; yield between polls."""
        import requests  # lazy

        start = time.time()
        while time.time() - start < timeout_seconds:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=1)
                if r.status_code == 200:
                    log.info("OculOS daemon ready at %s", self.base_url)
                    return
            except requests.exceptions.ConnectionError:
                pass
            await asyncio.sleep(0.1)
        self.stop()
        raise TimeoutError("OculOS daemon failed to start within the timeout period")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
            log.info("OculOS daemon stopped")


# Elements that belong to browser chrome (address bar, tabs, bookmarks) —
# transplanted marker list from ui.py; we never target these by accident.
_BROWSER_CHROME_MARKERS = (
    "address and search bar", "search or enter url",
    "search google or type a url", "omnibox",
    "bookmark", "bookmarks", "tab search", "new tab", "tab",
    "extensions", "profile", "chrome toolbar",
)

# Per-app launch flags so launched apps expose accessibility trees.
_CHROMIUM_BASED = ("chrome", "chromium", "electron", "vscode", "code",
                   "slack", "spotify", "discord")


def _launch_flags(name: str) -> tuple:
    lname = name.lower()
    if any(n in lname for n in _CHROMIUM_BASED):
        return ("--force-renderer-accessibility --enable-accessibility "
                "--disable-gpu --no-sandbox", {})
    if "firefox" in lname:
        return ("", {"GTK_MODULES": "gail:atk-bridge", "GNOME_ACCESSIBILITY": "1"})
    return ("", {})


def _node_to_element(node: dict, pid: int, path: str) -> Element:
    # OculOS node schema (verified live): type, label, value, text_content,
    # rect, enabled, focused, is_keyboard_focusable, oculos_id, actions.
    rect = node.get("rect") or {}
    name = node.get("label") or node.get("title") or node.get("name") or ""
    # AT-SPI 'enabled' is unreliable on GTK (live GNOME Calculator reports
    # clickable buttons as enabled=false). Trust a False only when the node
    # is also not keyboard-focusable — i.e. two signals agree it's inert.
    enabled = True
    if node.get("enabled") is False and not node.get("is_keyboard_focusable", False):
        enabled = False
    return Element(
        role=str(node.get("type") or node.get("element_type") or "generic").lower(),
        name=str(name),
        bounds=Bounds(
            x=float(rect.get("x", 0)), y=float(rect.get("y", 0)),
            width=float(rect.get("width", 0)), height=float(rect.get("height", 0)),
        ) if rect else None,
        provenance=frozenset({Source.TREE}),
        ref={"node_id": node.get("oculos_id"), "pid": pid, "path": path},
        enabled=enabled,
        focused=bool(node.get("focused") or node.get("is_focused") or False),
        value=(str(node["value"]) if node.get("value") is not None
               else (str(node["text_content"]) if node.get("text_content") else None)),
        hint=(str(node.get("help_text") or node.get("keyboard_shortcut") or "") or None),
    )


class AccessibilityDriver:
    """Driver over the OS accessibility tree (OculOS daemon)."""

    name = "tree"
    surface = "native"

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        binary_path: Optional[str] = None,
        auto_start_daemon: bool = True,
        close_launched: bool = True,
    ) -> None:
        self.client = OculOSClient(base_url=base_url)
        self.daemon = OculOSDaemon(binary_path=binary_path, base_url=base_url)
        self._auto_start_daemon = auto_start_daemon
        self._started = False
        self._target_exe: Optional[str] = None
        # Exe names this driver launched via navigate. Closed on stop() via
        # the daemon's window-close (killing the launcher Popen is useless
        # for D-Bus-activated GNOME apps, which hand off to a session
        # service and exit).
        self._launched: List[subprocess.Popen] = []
        self._launched_exes: Set[str] = set()
        self._close_launched = close_launched
        self._last_elements: List[Element] = []

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Ensure the daemon is reachable, launching it if allowed."""
        if self._started:
            return
        try:
            self.client.health()
            self._started = True
            return
        except Exception:
            pass
        if not self._auto_start_daemon:
            raise SurfaceUnreadable(
                f"OculOS daemon not reachable at {self.client.base_url} "
                "and auto-start is disabled"
            )
        await self.daemon.start()
        self._started = True

    async def stop(self) -> None:
        if self._close_launched and self._launched_exes:
            # Find the real PIDs of windows we launched (the launcher Popen
            # is useless for D-Bus-activated apps). Prefer a graceful window
            # -close via the daemon; if that's unavailable (e.g. no xdotool),
            # signal the actual process the daemon reports.
            try:
                for w in self.client.list_windows():
                    exe = str(w.get("exe_name", "")).lower()
                    pid = w.get("pid")
                    if not any(name in exe or exe in name for name in self._launched_exes):
                        continue
                    try:
                        self.client.close_window(pid)
                    except Exception:
                        _kill_pid(pid)
            except Exception:
                pass
        for proc in self._launched:
            try:
                proc.terminate()
            except Exception:
                pass
        self._launched.clear()
        self._launched_exes.clear()
        self.daemon.stop()
        self._started = False

    # -- helpers ------------------------------------------------------------

    # Compositor / desktop-shell processes: their AT-SPI trees describe the
    # whole desktop's plumbing (thousands of 'generic' Wayland surface
    # nodes), never the app the user cares about.
    _SHELL_EXES = {
        "gnome-shell", "mutter", "kwin_wayland", "kwin_x11", "plasmashell",
        "xfwm4", "xfdesktop", "sway", "hyprland", "weston", "xwayland",
    }

    def _focused_window(self) -> dict:
        try:
            windows = self.client.list_windows()
        except Exception as exc:
            raise SurfaceUnreadable(f"cannot list windows: {exc}") from exc
        candidates = [
            w for w in windows
            if w.get("visible", True)
            and str(w.get("exe_name", "")).lower() not in self._SHELL_EXES
        ]
        if not candidates:
            raise SurfaceUnreadable("no visible application windows")
        # 1. the app we last launched wins
        if self._target_exe:
            for w in candidates:
                if self._target_exe in str(w.get("exe_name", "")).lower():
                    return w
        # 2. an explicitly focused window, if the daemon reports it
        focused = [w for w in candidates if w.get("is_focused") or w.get("focused")]
        if focused:
            return focused[0]
        # 3. otherwise the first real app window
        return candidates[0]

    @staticmethod
    def _flatten(node: dict, pid: int, out: List[Element], path: str = "0") -> None:
        if node.get("oculos_id"):
            out.append(_node_to_element(node, pid, path))
        for i, child in enumerate(node.get("children") or []):
            AccessibilityDriver._flatten(child, pid, out, f"{path}.{i}")

    def _window_rect(self, pid: int) -> Optional[Dict[str, float]]:
        try:
            for w in self.client.list_windows():
                if w.get("pid") == pid:
                    return w.get("rect") or None
        except Exception:
            pass
        return None

    @staticmethod
    def _is_browser_chrome(el: Element) -> bool:
        text = f"{el.name} {el.value or ''} {el.role}".lower()
        return any(marker in text for marker in _BROWSER_CHROME_MARKERS)

    def _check_bounds_sanity(self, el: Element, pid: int) -> None:
        """Coordinate sanity before pointer actions: element must sit inside
        the window rect, else the tree is stale/lying — raise rather than
        click a phantom location."""
        if el.bounds is None:
            return
        rect = self._window_rect(pid)
        if not rect:
            return
        wx, wy = float(rect.get("x", 0)), float(rect.get("y", 0))
        ww, wh = float(rect.get("width", 0)), float(rect.get("height", 0))
        local = Bounds(el.bounds.x - wx, el.bounds.y - wy,
                       el.bounds.width, el.bounds.height)
        if not local.sane_within(ww, wh):
            raise TargetObstructed(
                f"element {el.name!r} has bounds outside its window rect",
                reason="insane_bounds",
                element_bounds=vars(el.bounds),
                window_rect=rect,
            )

    def _resolve(self, description: str, pid: int, _retry: bool = True) -> Element:
        """Find the best interactive node matching the description.

        Tries the daemon-side query first (exact then lowercase — the old
        wait_for_element heuristic), then falls back to fuzzy ranking over
        the full interactive set."""
        # Rank the full flattened tree with our own matcher. Verified live:
        # the daemon's query search misses exact label matches, and its
        # interactive=True filter excludes GTK buttons entirely (they expose
        # no 'click' action) — the tree endpoint is the only honest source.
        try:
            tree = self.client.get_tree(pid)
        except Exception as exc:
            raise SurfaceUnreadable(f"cannot fetch tree for pid {pid}: {exc}") from exc
        elements: List[Element] = []
        self._flatten(tree, pid, elements)
        if not elements:
            raise SurfaceUnreadable("accessibility tree is empty")
        # Filter browser chrome unless nothing else matches (ui.py heuristic:
        # the address bar's AT-SPI position shifts with focus state and can
        # shadow page content).
        page_elements = [e for e in elements if not self._is_browser_chrome(e)]
        pool = page_elements or elements

        match = best_match(pool, description)
        if match is None and pool is page_elements and elements:
            match = best_match(elements, description)
        if match is None and _retry:
            # Verified live on GTK: the tree fetched immediately after an
            # action is partial while the toolkit rebuilds it (alternating
            # hit/miss pattern). One fresh fetch after a beat fixes it.
            time.sleep(0.6)
            return self._resolve(description, pid, _retry=False)
        if match is None:
            near = suggestions(pool)
            raise TargetNotFound(
                f"no accessible element matches {description!r} in window {pid}"
                + (f"; closest labels: {near}" if near else ""),
                target=description,
                suggestions=near,
            )
        return match

    # -- Driver protocol ----------------------------------------------------

    async def observe(self) -> Observation:
        await self.start()
        window = self._focused_window()
        pid = int(window.get("pid", 0))
        try:
            tree = self.client.get_tree(pid)
        except Exception as exc:
            raise SurfaceUnreadable(f"cannot fetch tree for pid {pid}: {exc}") from exc

        elements: List[Element] = []
        self._flatten(tree, pid, elements)
        # Cache for ref addressing: the model points at the index it was
        # shown, so we return that exact element instead of re-matching.
        self._last_elements = elements
        modal_count = sum(
            1 for e in elements if e.role.lower() in ("dialog", "alertdialog", "alert dialog")
        )
        focused_key = next((e.key for e in elements if e.focused), None)
        return Observation(
            surface=f"window:{pid}",
            kind="native",
            title=str(window.get("title") or ""),
            elements=elements,
            modal_count=modal_count,
            focused_key=focused_key,
        )

    @staticmethod
    def can_navigate(target: Optional[str]) -> bool:
        """This driver launches applications: any non-empty target the dom
        driver doesn't claim as a web address."""
        return bool(target and target.strip()) and not is_web_target(target)

    async def act(self, action: Action) -> Optional[Element]:
        await self.start()

        if action.kind is ActionKind.NAVIGATE:
            self._launch_app(action.value or "")
            return None

        if action.kind is ActionKind.PRESS:
            if not action.value:
                raise TargetNotFound("press requires a key chord in action.value")
            self._press_keys(action.value)
            return None

        window = self._focused_window()
        pid = int(window.get("pid", 0))

        if action.kind is ActionKind.SCROLL:
            if action.target:
                el = self._resolve(action.target, pid)
                eid = str(el.ref.get("node_id"))
                self.client.scroll_into_view(eid)
                return el
            self._scroll_fallback(action.value or "down")
            return None

        if action.kind is ActionKind.FILL:
            if action.value is None:
                raise TargetNotFound("fill requires a value")
            el = self._by_ref(action.ref) or self._resolve_editable(action.target, pid)
            self._type_into(el, pid, action)
            return el

        el = self._by_ref(action.ref)
        if el is None:
            if not action.target:
                raise TargetNotFound(f"{action.kind.value} requires a ref or target")
            el = self._resolve(action.target, pid)
        eid = str(el.ref.get("node_id"))

        if action.kind is ActionKind.CLICK:
            self._check_bounds_sanity(el, pid)
            if not el.enabled:
                raise TargetObstructed(
                    f"element {el.name!r} is disabled", reason="disabled",
                )
            self._do_with_stale_retry(action, el, pid, lambda e: self.client.click(e))
            return el

        raise TargetNotFound(f"unsupported action {action.kind.value}")

    # Editable text roles across AT-SPI/UIA toolkits. The main editing
    # widget of an app (a document, an entry, a terminal) is one of these
    # and frequently has an empty accessible name.
    _EDITABLE_ROLES = {
        "text", "entry", "edit", "textbox", "text box", "document",
        "document text", "document frame", "document web", "terminal",
        "paragraph", "source view", "multi line text", "multiline text",
    }

    def _is_editable_text(self, el: Element) -> bool:
        return el.role.lower() in self._EDITABLE_ROLES

    def _by_ref(self, ref: Optional[int]) -> Optional[Element]:
        """Element the model pointed at, from the last rendered observation."""
        if ref is None:
            return None
        if 0 <= ref < len(self._last_elements):
            return self._last_elements[ref]
        raise TargetNotFound(
            f"ref {ref} is out of range (observation had "
            f"{len(self._last_elements)} elements) — observe again",
            target=str(ref),
        )

    def _resolve_editable(self, description: Optional[str], pid: int) -> Element:
        """Resolve a fill target. Try the named description first, but fall
        back to the editable text widget itself — a document/entry field
        often has an empty accessible name, so requiring a name match makes
        the one thing you obviously want to type into unhittable. Prefer the
        focused editable, else the largest one."""
        if description:
            try:
                return self._resolve(description, pid)
            except TargetNotFound:
                pass
        try:
            tree = self.client.get_tree(pid)
        except Exception as exc:
            raise SurfaceUnreadable(f"cannot fetch tree for pid {pid}: {exc}") from exc
        elements: List[Element] = []
        self._flatten(tree, pid, elements)
        editable = [e for e in elements if self._is_editable_text(e) and e.enabled]
        if not editable:
            raise TargetNotFound(
                f"no editable text field found in window {pid}"
                + (f" for {description!r}" if description else ""),
                target=description,
            )

        def area(e: Element) -> float:
            b = e.bounds
            return (b.width * b.height) if b else 0.0

        editable.sort(key=lambda e: (e.focused, area(e)), reverse=True)
        return editable[0]

        if action.kind is ActionKind.SELECT:
            if action.value is None:
                raise TargetNotFound("select requires a value (option text)")
            return self._select_option(el, pid, action)

        raise TargetNotFound(f"unsupported action kind: {action.kind}")

    async def screenshot(self) -> Optional[bytes]:
        try:
            import io

            import pyautogui  # lazy

            img = pyautogui.screenshot()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    # -- action internals ---------------------------------------------------

    def _do_with_stale_retry(self, action: Action, el: Element, pid: int, fn: Any) -> None:
        """Perform fn(element_id); on failure re-find (stale a11y node) and
        retry once — transplanted from interact_with_element."""
        eid = str(el.ref.get("node_id"))
        try:
            fn(eid)
            return
        except Exception as first_exc:
            msg = str(first_exc)
            # Transient COM errors: retry same id first (Windows UIA quirk).
            if any(code in msg for code in ("0x80004005", "0x80040201")):
                try:
                    fn(eid)
                    return
                except Exception as exc2:
                    msg = str(exc2)
            try:
                fresh = self._resolve(action.target or el.name, pid)
                fn(str(fresh.ref.get("node_id")))
                return
            except (TargetNotFound, TargetObstructed):
                raise
            except Exception as exc3:
                raise TargetObstructed(
                    f"a11y interaction failed: {exc3}",
                    reason="action_failed", first_error=msg,
                ) from exc3

    def _type_into(self, el: Element, pid: int, action: Action) -> None:
        """set_text with the multiline heuristic from ui.py type_into:
        AT-SPI SetValue cannot insert mid-string newlines, so multiline
        values need real keyboard simulation."""
        text = str(action.value)
        stripped = text.rstrip("\n")
        trailing = len(text) - len(stripped)
        pyautogui = self._try_pyautogui()

        if "\n" in stripped and pyautogui is not None:
            eid = str(el.ref.get("node_id"))
            self.client.click(eid)
            time.sleep(0.15)
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("delete")
            segments = stripped.split("\n")
            for i, segment in enumerate(segments):
                if segment:
                    pyautogui.typewrite(segment, interval=0.03)
                if i < len(segments) - 1:
                    pyautogui.press("enter")
            for _ in range(trailing):
                pyautogui.press("enter")
        elif trailing and pyautogui is not None:
            self._do_with_stale_retry(
                action, el, pid, lambda e: self.client.set_text(e, stripped)
            )
            time.sleep(0.05)
            for _ in range(trailing):
                pyautogui.press("enter")
        else:
            try:
                self._do_with_stale_retry(
                    action, el, pid, lambda e: self.client.set_text(e, text)
                )
            except OrbitError:
                # Not every editable widget implements AT-SPI EditableText
                # (verified live: GtkSourceView in gnome-text-editor raises
                # UnknownMethod). Fall back to focusing it and typing for
                # real — the same thing a human does.
                self._type_by_keyboard(el, text)

    def _type_by_keyboard(self, el: Element, text: str) -> None:
        """Focus the element and type with real key events."""
        pyautogui = self._try_pyautogui()
        if pyautogui is None:
            raise TargetObstructed(
                "widget does not support accessibility text entry and "
                "pyautogui is unavailable for keyboard fallback",
                reason="no_text_interface",
            )
        eid = str(el.ref.get("node_id"))
        for attempt in (self.client.focus, self.client.click):
            try:
                attempt(eid)
                break
            except Exception:
                continue
        time.sleep(0.15)
        pyautogui.typewrite(text, interval=0.02)

    def _select_option(self, el: Element, pid: int, action: Action) -> Element:
        """Expand the dropdown, then find and select the option node."""
        eid = str(el.ref.get("node_id"))
        try:
            self.client.expand(eid)
        except Exception:
            try:
                self.client.click(eid)
            except Exception:
                pass
        time.sleep(0.3)
        try:
            options = self.client.find_elements(pid, query=str(action.value)) or []
        except Exception:
            options = []
        option_els = [_node_to_element(n, pid, "?") for n in options]
        ranked = rank_matches(option_els, str(action.value))
        if not ranked:
            raise TargetNotFound(
                f"option {action.value!r} not found in dropdown {action.target!r}",
                target=action.target,
                suggestions=suggestions(option_els),
            )
        opt = ranked[0].element
        try:
            self.client.select(str(opt.ref.get("node_id")))
        except Exception:
            self.client.click(str(opt.ref.get("node_id")))
        return opt

    def _launch_app(self, app_name: str) -> None:
        """Launch an app with accessibility flags (from ui.py manage_window)."""
        if not app_name:
            raise TargetNotFound("navigate requires an app name in action.value")
        # Remember what we launched so observation targets this app's
        # window, not whatever the window list happens to lead with.
        self._target_exe = os.path.basename(app_name.split()[0]).lower()
        self._launched_exes.add(self._target_exe)
        system = platform.system()
        if system == "Windows":
            try:
                os.startfile(app_name)  # type: ignore[attr-defined]
            except (FileNotFoundError, OSError):
                self._launched.append(subprocess.Popen(f"start {app_name}", shell=True))
        elif system == "Darwin":
            self._launched.append(subprocess.Popen(["open", "-a", app_name]))
        else:
            import shlex
            flags, env_vars = _launch_flags(app_name)
            env = {**os.environ, **env_vars}
            # No shell: a display name like "Text Editor" would shell-split
            # into a bogus command ("Text: not found") that fails silently.
            argv = shlex.split(app_name) + (shlex.split(flags) if flags else [])
            try:
                self._launched.append(subprocess.Popen(
                    argv, env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                ))
            except (FileNotFoundError, PermissionError, OSError) as exc:
                raise TargetNotFound(
                    f"cannot launch {app_name!r}: {exc}. Use the executable "
                    "name (e.g. 'gnome-text-editor'), not a display name.",
                    target=app_name,
                ) from exc

    def _press_keys(self, chord: str) -> None:
        pyautogui = self._try_pyautogui()
        if pyautogui is None:
            raise TargetObstructed(
                "pyautogui unavailable — cannot send raw keys",
                reason="no_keyboard_backend",
            )
        keys = [k.strip().lower() for k in chord.split("+") if k.strip()]
        if len(keys) > 1:
            pyautogui.hotkey(*keys)
        elif keys:
            pyautogui.press(keys[0])

    def _scroll_fallback(self, direction: str) -> None:
        pyautogui = self._try_pyautogui()
        if pyautogui is None:
            raise TargetObstructed(
                "pyautogui unavailable — cannot scroll without a target",
                reason="no_pointer_backend",
            )
        amount = 300 if direction.lower() == "up" else -300
        pyautogui.scroll(amount)

    @staticmethod
    def _try_pyautogui() -> Any:
        try:
            import pyautogui  # lazy

            return pyautogui
        except Exception:
            return None
