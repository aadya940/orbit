"""OculOS Python client."""

from __future__ import annotations

import requests
from typing import Any, Optional


class OculOS:
    """Thin wrapper around the OculOS REST API."""

    def __init__(self, base_url: str = "http://127.0.0.1:7878"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ── Discovery ──────────────────────────────────────────────

    def list_windows(self) -> list[dict]:
        """List all visible windows."""
        return self._get("/windows")

    def get_tree(self, pid: int) -> dict:
        """Get the full UI element tree for a window."""
        return self._get(f"/windows/{pid}/tree")

    def get_tree_hwnd(self, hwnd: int) -> dict:
        """Get the UI element tree by window handle."""
        return self._get(f"/hwnd/{hwnd}/tree")

    def find_elements(
        self,
        pid: int,
        *,
        query: Optional[str] = None,
        element_type: Optional[str] = None,
        interactive: Optional[bool] = None,
    ) -> list[dict]:
        """Search for UI elements in a window."""
        params: dict[str, Any] = {}
        if query is not None:
            params["q"] = query
        if element_type is not None:
            params["type"] = element_type
        if interactive is not None:
            params["interactive"] = str(interactive).lower()
        return self._get(f"/windows/{pid}/find", params=params)

    def find_elements_hwnd(
        self,
        hwnd: int,
        *,
        query: Optional[str] = None,
        element_type: Optional[str] = None,
        interactive: Optional[bool] = None,
    ) -> list[dict]:
        """Search for UI elements by window handle."""
        params: dict[str, Any] = {}
        if query is not None:
            params["q"] = query
        if element_type is not None:
            params["type"] = element_type
        if interactive is not None:
            params["interactive"] = str(interactive).lower()
        return self._get(f"/hwnd/{hwnd}/find", params=params)

    # ── Window operations ──────────────────────────────────────

    def focus_window(self, pid: int) -> None:
        """Bring a window to the foreground."""
        self._post(f"/windows/{pid}/focus")

    def close_window(self, pid: int) -> None:
        """Close a window gracefully."""
        self._post(f"/windows/{pid}/close")

    # ── Element interactions ───────────────────────────────────

    def click(self, element_id: str) -> dict:
        """Click an element."""
        return self._post(f"/interact/{element_id}/click")

    def set_text(self, element_id: str, text: str) -> dict:
        """Set the text content of an input field."""
        return self._post(f"/interact/{element_id}/set-text", json={"text": text})

    def send_keys(self, element_id: str, keys: str) -> dict:
        """Send keyboard input to an element."""
        return self._post(f"/interact/{element_id}/send-keys", json={"keys": keys})

    def focus(self, element_id: str) -> dict:
        """Move keyboard focus to an element."""
        return self._post(f"/interact/{element_id}/focus")

    def toggle(self, element_id: str) -> dict:
        """Toggle a checkbox or toggle button."""
        return self._post(f"/interact/{element_id}/toggle")

    def expand(self, element_id: str) -> dict:
        """Expand a dropdown, tree item, or menu."""
        return self._post(f"/interact/{element_id}/expand")

    def collapse(self, element_id: str) -> dict:
        """Collapse a dropdown, tree item, or menu."""
        return self._post(f"/interact/{element_id}/collapse")

    def select(self, element_id: str) -> dict:
        """Select a list item, radio button, or tab."""
        return self._post(f"/interact/{element_id}/select")

    def set_range(self, element_id: str, value: float) -> dict:
        """Set a slider or spinner value."""
        return self._post(f"/interact/{element_id}/set-range", json={"value": value})

    def scroll(self, element_id: str, direction: str) -> dict:
        """Scroll a container. Direction: up, down, left, right."""
        return self._post(
            f"/interact/{element_id}/scroll", json={"direction": direction}
        )

    def scroll_into_view(self, element_id: str) -> dict:
        """Scroll an element into the visible viewport."""
        return self._post(f"/interact/{element_id}/scroll-into-view")

    def highlight(self, element_id: str, duration_ms: int = 2000) -> dict:
        """Highlight an element on screen."""
        return self._post(
            f"/interact/{element_id}/highlight", json={"duration_ms": duration_ms}
        )

    # ── System ─────────────────────────────────────────────────

    def health(self) -> dict:
        """Get server status, version, and uptime."""
        return self._get("/health")

    # ── Internals ──────────────────────────────────────────────

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        r = self._session.get(f"{self.base_url}{path}", params=params)
        try:
            body = r.json()
        except ValueError:
            r.raise_for_status()
            raise OculOSError(f"HTTP {r.status_code}: non-JSON response")
        if not body.get("success"):
            raise OculOSError(body.get("error", f"HTTP {r.status_code}"))
        return body["data"]

    def _post(self, path: str, json: Optional[dict] = None) -> Any:
        r = self._session.post(f"{self.base_url}{path}", json=json)
        try:
            body = r.json()
        except ValueError:
            r.raise_for_status()
            raise OculOSError(f"HTTP {r.status_code}: non-JSON response")
        if not body.get("success"):
            raise OculOSError(body.get("error", f"HTTP {r.status_code}"))
        return body.get("data")


class OculOSError(Exception):
    """Raised when the OculOS API returns an error."""
