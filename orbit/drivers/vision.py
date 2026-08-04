"""VisionDriver — screenshot grounding rung.

The last-resort perception/action backend: when neither the DOM nor the
OS accessibility tree can see a surface (canvas apps, custom-drawn UI,
remote desktops, toolkits with broken a11y), this driver looks at pixels
the way a person does — screenshot, ground natural-language targets to
coordinates with a vision model, click there.

It is deliberately last in the ladder: a grounding call costs a model
round-trip and real latency, while a tree/DOM read is ~free. The cheap
paths run first and this one only pays when they fail.

Parsing/geometry are pure functions (unit-testable with no screen and no
model); only capture and the model call touch the outside world.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..types import (
    Action,
    ActionKind,
    Bounds,
    Element,
    Observation,
    Source,
    SurfaceUnreadable,
    TargetNotFound,
    TargetObstructed,
)

log = logging.getLogger("orbit.drivers.vision")

# Vision models emit boxes in a normalized 0-1000 space (Gemini convention:
# [ymin, xmin, ymax, xmax]). We scale to real pixels ourselves.
_NORM = 1000.0

_GROUND_PROMPT = """You are looking at a screenshot of a computer screen.

List the interactive UI elements you can see (buttons, text fields, links,
menu items, checkboxes, tabs, icons). For each, give:
- "role": button | textbox | link | menuitem | checkbox | tab | icon | other
- "name": its visible label, or a short description if it has no text
- "box": [ymin, xmin, ymax, xmax] normalized to 0-1000
- "enabled": true/false (false if it looks greyed out)

Also transcribe every piece of visible text on screen (labels, headings,
status lines, values, body copy) into "text" — read/verify tasks depend on
it, not just on the clickable elements.

Return ONLY a JSON object:
{"elements": [...], "title": "<window title if visible>", "text": "<all visible text>"}
Include every element a user could click or type into. Be precise with boxes.
"""


def _png_size(png: bytes) -> Tuple[int, int]:
    """Width/height straight from the PNG IHDR — no image library needed."""
    if len(png) >= 24 and png[:8] == b"\x89PNG\r\n\x1a\n":
        return (
            int.from_bytes(png[16:20], "big"),
            int.from_bytes(png[20:24], "big"),
        )
    return (0, 0)


def parse_grounding(raw: str) -> Tuple[List[dict], str, str]:
    """Extract (elements, title, visible_text) from a grounding reply.

    Tolerates fenced code blocks and leading prose — models wrap JSON in
    markdown often enough that failing on it would be a reliability bug,
    not strictness.
    """
    if not raw:
        return [], "", ""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], "", ""
    if not isinstance(data, dict):
        return [], "", ""
    elements = data.get("elements")
    if not isinstance(elements, list):
        return [], "", ""
    return (
        [e for e in elements if isinstance(e, dict)],
        str(data.get("title") or ""),
        str(data.get("text") or ""),
    )


def box_to_bounds(box: Any, width: int, height: int) -> Optional[Bounds]:
    """Convert a normalized [ymin, xmin, ymax, xmax] box to pixel Bounds."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    x = (xmin / _NORM) * width
    y = (ymin / _NORM) * height
    w = ((xmax - xmin) / _NORM) * width
    h = ((ymax - ymin) / _NORM) * height
    if w < 0 or h < 0:
        return None
    return Bounds(x=x, y=y, width=w, height=h)


def elements_from_grounding(
    raw_elements: List[dict], width: int, height: int
) -> List[Element]:
    """Build Elements from grounded boxes, dropping ones without sane geometry."""
    out: List[Element] = []
    for desc in raw_elements:
        bounds = box_to_bounds(desc.get("box"), width, height)
        if bounds is None or not bounds.sane_within(width, height):
            continue
        out.append(Element(
            role=str(desc.get("role") or "other"),
            name=str(desc.get("name") or ""),
            bounds=bounds,
            provenance=frozenset({Source.VISION}),
            ref={"cx": bounds.center[0], "cy": bounds.center[1]},
            enabled=bool(desc.get("enabled", True)),
        ))
    return out


class VisionDriver:
    """Pixel-grounding rung: sees what a person sees, clicks where they would."""

    name = "vision"
    surface = None  # grounds on whatever is on screen

    def __init__(self, llm: Any = None, *, model: Optional[str] = None) -> None:
        # `llm` is any object with async complete(messages, tools) (the
        # Session passes its own). `model` overrides it with a direct
        # litellm model id, useful for a cheap dedicated grounding model.
        self._llm = llm
        self._model = model
        self._last_elements: List[Element] = []
        self._last_size: Tuple[int, int] = (0, 0)

    def set_llm(self, llm: Any) -> None:
        """Session injects its LLM so vision doesn't need separate config."""
        if self._llm is None:
            self._llm = llm

    def set_capture(self, capture: Any) -> None:
        """Ground on the focused surface's own pixels.

        The World injects the current primary driver's screenshot(), so a
        browser surface is grounded from the page bitmap (crisp, and works
        without a reachable X display) and a native surface from the screen.
        Falls back to a full-screen grab when no source is injected.
        """
        self._capture = capture

    # -- capture ------------------------------------------------------------

    async def _capture_png(self) -> Tuple[bytes, int, int]:
        """Pixels of the focused surface, plus their size."""
        capture = getattr(self, "_capture", None)
        if capture is not None:
            png = await capture()
            if png:
                return (png, *_png_size(png))
        return self._grab()

    def _grab(self) -> Tuple[bytes, int, int]:
        try:
            # Not just ImportError: on Linux importing pyautogui opens an X
            # display and raises DisplayConnectionError when headless.
            import pyautogui  # lazy
        except Exception as exc:
            raise SurfaceUnreadable(
                f"no usable screenshot backend ({exc.__class__.__name__}: {exc}). "
                "Vision needs pyautogui + a reachable display."
            ) from exc
        try:
            img = pyautogui.screenshot()
        except Exception as exc:
            raise SurfaceUnreadable(f"screenshot failed: {exc}") from exc
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), img.width, img.height

    async def _ground(self, png: bytes) -> str:
        if self._llm is None and not self._model:
            raise SurfaceUnreadable(
                "vision has no model configured (Session normally injects one)"
            )
        b64 = base64.b64encode(png).decode()
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": _GROUND_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }]
        if self._model:
            import litellm  # lazy
            resp = await litellm.acompletion(
                model=self._model, messages=messages, timeout=90
            )
            return resp.choices[0].message.content or ""
        reply = await self._llm.complete(messages, [])
        return reply.text or ""

    # -- Driver protocol ----------------------------------------------------

    async def observe(self) -> Observation:
        png, width, height = await self._capture_png()
        raw = await self._ground(png)
        raw_elements, title, visible_text = parse_grounding(raw)
        elements = elements_from_grounding(raw_elements, width, height)
        if not elements:
            raise SurfaceUnreadable("vision grounding returned no usable elements")
        self._last_elements = elements
        self._last_size = (width, height)
        return Observation(
            surface="screen",
            kind="screen",
            title=title,
            elements=elements,
            text=visible_text,
        )

    def set_pointer(self, pointer: Any) -> None:
        """Inject a surface-local pointer (e.g. the browser page's mouse) so
        grounded coordinates are clicked in that surface's own space rather
        than as OS-global screen coordinates."""
        self._pointer = pointer

    async def act(self, action: Action) -> Optional[Element]:
        if action.kind is ActionKind.NAVIGATE:
            raise TargetNotFound("vision cannot open apps or urls", target=action.value)

        pointer = getattr(self, "_pointer", None)
        if pointer is not None:
            return await self._act_via_pointer(pointer, action)

        pyautogui = self._pyautogui()

        if action.kind is ActionKind.PRESS:
            if not action.value:
                raise TargetNotFound("press requires a key chord")
            keys = [k.strip().lower() for k in re.split(r"[+\-]", action.value) if k.strip()]
            if len(keys) > 1:
                pyautogui.hotkey(*keys)
            else:
                pyautogui.press(keys[0])
            return None

        el = self._resolve(action)
        if el is None or el.bounds is None:
            raise TargetNotFound(
                f"vision could not locate {action.target!r} on screen",
                target=action.target,
            )
        if not el.enabled:
            raise TargetObstructed(f"element {el.name!r} looks disabled", reason="disabled")

        cx, cy = el.bounds.center
        if action.kind is ActionKind.SCROLL:
            pyautogui.moveTo(cx, cy)
            pyautogui.scroll(-400 if (action.value or "down") == "down" else 400)
            return el

        pyautogui.click(cx, cy)
        if action.kind is ActionKind.FILL:
            if action.value is None:
                raise TargetNotFound("fill requires a value")
            pyautogui.hotkey("ctrl", "a")
            pyautogui.typewrite(str(action.value), interval=0.02)
        return el

    async def screenshot(self) -> Optional[bytes]:
        try:
            png, _, _ = self._grab()
            return png
        except SurfaceUnreadable:
            return None

    # -- helpers ------------------------------------------------------------

    async def _act_via_pointer(self, pointer: Any, action: Action) -> Optional[Element]:
        """Act inside the focused surface using its own input (page mouse/
        keyboard), with coordinates in that surface's space."""
        if action.kind is ActionKind.PRESS:
            if not action.value:
                raise TargetNotFound("press requires a key chord")
            await pointer.press(action.value)
            return None
        el = self._resolve(action)
        if el is None or el.bounds is None:
            raise TargetNotFound(
                f"vision could not locate {action.target!r}", target=action.target,
            )
        if not el.enabled:
            raise TargetObstructed(f"element {el.name!r} looks disabled", reason="disabled")
        cx, cy = el.bounds.center
        if action.kind is ActionKind.SCROLL:
            await pointer.scroll(cx, cy, action.value or "down")
            return el
        await pointer.click(cx, cy)
        if action.kind is ActionKind.FILL:
            if action.value is None:
                raise TargetNotFound("fill requires a value")
            await pointer.type(str(action.value))
        return el

    def _pyautogui(self):
        try:
            import pyautogui  # lazy (import itself can fail headless)
            return pyautogui
        except Exception as exc:
            raise TargetObstructed(
                f"vision needs pyautogui + a display to act ({exc})",
                reason="no_input_backend",
            ) from exc

    def _resolve(self, action: Action) -> Optional[Element]:
        """Ref first (the model points at what it was shown), then name match
        over the last grounded elements."""
        if action.ref is not None:
            if 0 <= action.ref < len(self._last_elements):
                return self._last_elements[action.ref]
            raise TargetNotFound(
                f"ref {action.ref} out of range for the last vision observation",
                target=str(action.ref),
            )
        if not action.target:
            raise TargetNotFound("vision act requires a ref or target")
        from .matching import best_match

        return best_match(self._last_elements, action.target)
