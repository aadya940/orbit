"""VisionDriver — STUB for the planned screenshot-grounding rung.

The full implementation will ground natural-language targets against
pixels via a vision grounding model (screenshot -> element boxes), making
it the fallback for unlabeled/custom-drawn UI that neither the DOM nor
the accessibility tree can see. Until then it only captures screenshots.
"""

from __future__ import annotations

from typing import Optional

from ..types import Action, Element, Observation, SurfaceUnreadable, TargetNotFound

_NOT_IMPLEMENTED = "vision grounding not yet implemented"


class VisionDriver:
    """Planned grounding-model rung — currently a stub."""

    name = "vision"
    surface = None  # grounds on the focused screen

    async def observe(self) -> Observation:
        raise SurfaceUnreadable(_NOT_IMPLEMENTED)

    async def act(self, action: Action) -> Optional[Element]:
        raise TargetNotFound(_NOT_IMPLEMENTED, target=action.target)

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
