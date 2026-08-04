"""VisionDriver, the screenshot grounding rung.

The last resort perception and action backend. When neither the DOM nor
the OS accessibility tree can see a surface (canvas apps, custom drawn
UI, remote desktops, toolkits with broken accessibility), this driver
looks at pixels the way a person does: screenshot, ground natural
language targets to coordinates with a vision model, then click there.

Notes
-----
Vision is deliberately last in the ladder. A grounding call costs a
model round trip and real latency, while a tree or DOM read is close to
free. The cheap paths run first and this one only pays when they fail.

Parsing and geometry are pure functions, unit testable with no screen
and no model attached. Only capture and the model call touch the outside
world.

Grounding is done on the owner surface's own pixels wherever possible.
The World injects the primary driver's ``screenshot()``, so a browser
surface is grounded from the page bitmap, which is crisp and works
without a reachable X display, while a native surface is grounded from
the screen. Grounding on a full screen grab when a surface bitmap exists
would mismatch coordinate spaces and click the wrong place.
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
status lines, values, body copy) into "text", because read and verify tasks
depend on it, not just on the clickable elements.

Return ONLY a JSON object:
{"elements": [...], "title": "<window title if visible>", "text": "<all visible text>"}
Include every element a user could click or type into. Be precise with boxes.
"""


def _png_size(png: bytes) -> Tuple[int, int]:
    """Read width and height straight from the PNG IHDR chunk.

    Parameters
    ----------
    png : bytes
        Raw PNG bytes.

    Returns
    -------
    Tuple[int, int]
        The ``(width, height)`` in pixels, or ``(0, 0)`` if the bytes do
        not carry a PNG signature and a readable header.

    Notes
    -----
    Parsing the header directly avoids requiring an image library just
    to learn the dimensions needed to denormalize grounded boxes.
    """
    if len(png) >= 24 and png[:8] == b"\x89PNG\r\n\x1a\n":
        return (
            int.from_bytes(png[16:20], "big"),
            int.from_bytes(png[20:24], "big"),
        )
    return (0, 0)


def parse_grounding(raw: str) -> Tuple[List[dict], str, str]:
    """Extract elements, title and visible text from a grounding reply.

    Parameters
    ----------
    raw : str
        The raw model reply, which may be bare JSON, fenced JSON, or
        JSON surrounded by prose.

    Returns
    -------
    Tuple[List[dict], str, str]
        The element descriptors, the window title, and the transcribed
        visible text. All three are empty when the reply cannot be
        parsed into the expected shape.

    Notes
    -----
    Fenced code blocks and leading prose are tolerated rather than
    rejected. Models wrap JSON in markdown often enough that failing on
    it would be a reliability bug rather than useful strictness. A
    parse failure returns empty results instead of raising, so the
    caller reports an unreadable surface and the ladder moves on.
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
    """Convert a normalized ``[ymin, xmin, ymax, xmax]`` box to pixel Bounds.

    Parameters
    ----------
    box : Any
        The candidate box. Anything that is not a four item sequence of
        numbers is rejected.
    width : int
        Surface width in pixels to scale the x axis by.
    height : int
        Surface height in pixels to scale the y axis by.

    Returns
    -------
    Bounds or None
        The pixel space bounds, or None if the box is malformed or has
        negative extent.

    Notes
    -----
    Boxes arrive in the Gemini convention of a normalized 0 to 1000
    space with the y coordinates first, so the conversion cannot be a
    plain positional unpack into Bounds.
    """
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
    """Build Elements from grounded boxes, dropping ones without sane geometry.

    Parameters
    ----------
    raw_elements : List[dict]
        Element descriptors parsed from the grounding reply.
    width : int
        Surface width in pixels.
    height : int
        Surface height in pixels.

    Returns
    -------
    List[Element]
        Elements with pixel bounds and VISION provenance, in reply
        order. Descriptors whose boxes are malformed or fall outside the
        surface are skipped.

    Notes
    -----
    Each element carries its center point in ``ref`` so that a later act
    can click exactly where the model pointed, without re-grounding.
    Filtering on ``sane_within`` here rather than at click time means a
    hallucinated off-screen box never becomes a clickable target.
    """
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
    """Pixel grounding rung: sees what a person sees, clicks where they would.

    Attributes
    ----------
    name : str
        Driver id, ``"vision"``.
    surface : None
        None, because this driver is surface agnostic and grounds on
        whatever pixels it is given.
    """

    name = "vision"
    surface = None  # grounds on whatever is on screen

    def __init__(self, llm: Any = None, *, model: Optional[str] = None) -> None:
        """Create a vision driver, optionally bound to a model up front.

        Parameters
        ----------
        llm : Any, optional
            Any object exposing an async ``complete(messages, tools)``.
            The Session normally passes its own. Default is None, in
            which case one must be injected before observing.
        model : str, optional
            A direct litellm model id which takes precedence over
            ``llm``. Useful for pointing grounding at a cheap dedicated
            vision model instead of the main reasoning model. Default is
            None.

        Notes
        -----
        Construction performs no capture and no model call, so an
        instance that the ladder never reaches costs nothing.
        """
        # `llm` is any object with async complete(messages, tools) (the
        # Session passes its own). `model` overrides it with a direct
        # litellm model id, useful for a cheap dedicated grounding model.
        self._llm = llm
        self._model = model
        self._last_elements: List[Element] = []
        self._last_size: Tuple[int, int] = (0, 0)

    def set_llm(self, llm: Any) -> None:
        """Adopt the Session's LLM so vision needs no separate configuration.

        Parameters
        ----------
        llm : Any
            Any object exposing an async ``complete(messages, tools)``.

        Returns
        -------
        None

        Notes
        -----
        An explicitly constructed LLM wins: injection only fills the
        slot when it is still empty, so a caller that deliberately chose
        a grounding model is never silently overridden by the Session.
        """
        if self._llm is None:
            self._llm = llm

    def set_capture(self, capture: Any) -> None:
        """Bind the pixel source used for grounding.

        Parameters
        ----------
        capture : Any
            Async callable returning PNG bytes for the surface that
            currently owns the task.

        Returns
        -------
        None

        Notes
        -----
        The World injects the current primary driver's ``screenshot()``,
        so a browser surface is grounded from the page bitmap, which is
        crisp and works without a reachable X display, and a native
        surface is grounded from the screen. Grounding on the owner
        surface's own pixels keeps the coordinate space consistent with
        the pointer that will click them. When no source is injected the
        driver falls back to a full screen grab.
        """
        self._capture = capture

    # -- capture ------------------------------------------------------------

    async def _capture_png(self) -> Tuple[bytes, int, int]:
        """Capture pixels of the focused surface, plus their size.

        Returns
        -------
        Tuple[bytes, int, int]
            PNG bytes with their width and height in pixels.

        Raises
        ------
        SurfaceUnreadable
            If no injected capture produced bytes and the screen grab
            fallback is unavailable or fails.

        Notes
        -----
        The injected surface capture is preferred so that grounded
        coordinates land in the same space the acting pointer uses. A
        capture that returns nothing falls through to the screen grab
        rather than failing outright.
        """
        capture = getattr(self, "_capture", None)
        if capture is not None:
            png = await capture()
            if png:
                return (png, *_png_size(png))
        return self._grab()

    def _grab(self) -> Tuple[bytes, int, int]:
        """Grab the whole screen through pyautogui.

        Returns
        -------
        Tuple[bytes, int, int]
            PNG bytes with their width and height in pixels.

        Raises
        ------
        SurfaceUnreadable
            If pyautogui cannot be imported or the capture fails.

        Notes
        -----
        The import is guarded against every exception, not just
        ImportError. On Linux, importing pyautogui opens an X display
        and raises a display connection error when running headless,
        which would otherwise escape as an untyped crash instead of a
        failed rung.
        """
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
        """Ask the vision model to describe the interactive elements in an image.

        Parameters
        ----------
        png : bytes
            The screenshot to ground, sent inline as a base64 data URL.

        Returns
        -------
        str
            The raw model reply, to be handed to :func:`parse_grounding`.

        Raises
        ------
        SurfaceUnreadable
            If neither an injected LLM nor a direct model id is
            configured.

        Notes
        -----
        A configured model id takes the direct litellm path; otherwise
        the Session's own LLM is reused, which keeps credentials and
        retry policy in one place.
        """
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
        """Capture the surface and ground it into an observation.

        Returns
        -------
        Observation
            A screen observation carrying the grounded elements and the
            transcribed visible text.

        Raises
        ------
        SurfaceUnreadable
            If capture fails, no model is configured, or grounding
            yields no element with usable geometry.

        Notes
        -----
        The grounded elements and the surface size are cached so that a
        subsequent ``act`` can address an element by ref, clicking
        exactly what the model was shown rather than re-grounding and
        risking a different interpretation.
        """
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
        """Bind a surface local pointer for acting on grounded coordinates.

        Parameters
        ----------
        pointer : Any
            An object exposing async ``click``, ``scroll``, ``type`` and
            ``press``, for example the browser page's mouse and
            keyboard.

        Returns
        -------
        None

        Notes
        -----
        With a pointer bound, grounded coordinates are clicked in the
        owning surface's own space rather than as OS global screen
        coordinates. This matches the surface bitmap that grounding was
        performed on, and it works where no desktop pointer is
        available at all.
        """
        self._pointer = pointer

    async def act(self, action: Action) -> Optional[Element]:
        """Act on a grounded element, or send a key chord at current focus.

        Parameters
        ----------
        action : Action
            The action to perform. NAVIGATE is not supported.

        Returns
        -------
        Element or None
            The grounded element acted on, or None for PRESS.

        Raises
        ------
        TargetNotFound
            If the action is NAVIGATE, if PRESS carries no chord, if
            FILL carries no value, or if the target cannot be located
            among the last grounded elements.
        TargetObstructed
            If the resolved element looks disabled, or if no input
            backend is available to act with.

        Notes
        -----
        When a surface local pointer has been injected, all acting is
        delegated to it so that coordinates stay in the surface's own
        space. Otherwise the driver falls back to desktop input through
        pyautogui. FILL selects all before typing so that the new value
        replaces rather than appends to existing content.
        """
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
        """Capture the screen as PNG bytes.

        Returns
        -------
        bytes or None
            PNG bytes, or None when no screenshot backend is available.

        Notes
        -----
        SurfaceUnreadable is swallowed here because the protocol treats
        an absent screenshot as None rather than an error. A missing
        display should not turn an otherwise successful run into a
        failure.
        """
        try:
            png, _, _ = self._grab()
            return png
        except SurfaceUnreadable:
            return None

    # -- helpers ------------------------------------------------------------

    async def _act_via_pointer(self, pointer: Any, action: Action) -> Optional[Element]:
        """Act inside the focused surface using that surface's own input.

        Parameters
        ----------
        pointer : Any
            The surface local pointer, for example the page mouse and
            keyboard.
        action : Action
            The action to perform.

        Returns
        -------
        Element or None
            The grounded element acted on, or None for PRESS.

        Raises
        ------
        TargetNotFound
            If PRESS carries no chord, if FILL carries no value, or if
            the target cannot be located.
        TargetObstructed
            If the resolved element looks disabled.

        Notes
        -----
        Coordinates are used in the surface's own space, matching the
        bitmap that grounding ran on. Routing through the page pointer
        rather than the desktop one also avoids depending on window
        stacking or a reachable display.
        """
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
        """Import pyautogui on demand for desktop input.

        Returns
        -------
        Any
            The imported ``pyautogui`` module.

        Raises
        ------
        TargetObstructed
            If pyautogui is unavailable, including when the import
            itself fails for lack of a display.

        Notes
        -----
        This raises TargetObstructed rather than TargetNotFound because
        the target may well exist; it is the means of acting on it that
        is missing.
        """
        try:
            import pyautogui  # lazy (import itself can fail headless)
            return pyautogui
        except Exception as exc:
            raise TargetObstructed(
                f"vision needs pyautogui + a display to act ({exc})",
                reason="no_input_backend",
            ) from exc

    def _resolve(self, action: Action) -> Optional[Element]:
        """Resolve an action to one of the last grounded elements.

        Parameters
        ----------
        action : Action
            The action carrying either a ``ref`` index or a target
            description.

        Returns
        -------
        Element or None
            The resolved element, or None when a target description
            matches nothing well enough.

        Raises
        ------
        TargetNotFound
            If ``ref`` is out of range for the last observation, or if
            the action carries neither a ref nor a target.

        Notes
        -----
        Ref is tried first: the model points at exactly what it was
        shown, so addressing by index is unambiguous, whereas re-matching
        by name can silently pick a different element when several share
        a label or when the grounding pass worded them differently. Name
        matching remains as a fallback for callers that did not carry a
        ref through.
        """
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
