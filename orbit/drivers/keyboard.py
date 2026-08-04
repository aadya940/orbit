"""KeyboardDriver — last-resort rung: acts via pure keyboard.

It cannot perceive (observe() raises SurfaceUnreadable — perception must
come from another driver) and cannot locate targets (CLICK raises
TargetNotFound). A caller can still advance flows with PRESS sequences
(Tab/Enter) and FILL typing at the current focus.
"""

from __future__ import annotations

from typing import Any, Optional

from ..types import (
    Action,
    ActionKind,
    Element,
    Observation,
    SurfaceUnreadable,
    TargetNotFound,
)


class KeyboardDriver:
    """Pure-keyboard action backend (pyautogui, lazily imported)."""

    name = "keyboard"
    surface = None  # acts on the focused OS window

    @staticmethod
    def _pyautogui() -> Any:
        try:
            import pyautogui  # lazy

            return pyautogui
        except Exception as exc:
            raise TargetNotFound(
                f"pyautogui unavailable: {exc}"
            ) from exc

    async def observe(self) -> Observation:
        raise SurfaceUnreadable(
            "keyboard driver cannot perceive — use another driver for observation"
        )

    async def act(self, action: Action) -> Optional[Element]:
        if action.kind is ActionKind.PRESS:
            if not action.value:
                raise TargetNotFound("press requires a key chord in action.value")
            pg = self._pyautogui()
            keys = [k.strip().lower() for k in action.value.split("+") if k.strip()]
            if len(keys) > 1:
                pg.hotkey(*keys)
            elif keys:
                pg.press(keys[0])
            return None

        if action.kind is ActionKind.FILL:
            if action.value is None:
                raise TargetNotFound("fill requires a value")
            pg = self._pyautogui()
            for i, segment in enumerate(str(action.value).split("\n")):
                if i:
                    pg.press("enter")
                if segment:
                    pg.typewrite(segment, interval=0.03)
            return None

        # CLICK / SELECT / SCROLL / NAVIGATE: this driver cannot locate targets.
        raise TargetNotFound(
            f"keyboard driver cannot resolve targets for {action.kind.value}; "
            "use PRESS sequences (Tab/Enter) instead",
            target=action.target,
        )

    async def screenshot(self) -> Optional[bytes]:
        return None
