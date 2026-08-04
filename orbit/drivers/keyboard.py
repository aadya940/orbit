"""KeyboardDriver, the last resort ladder rung, acting via pure keyboard.

Notes
-----
This driver cannot perceive: ``observe()`` raises SurfaceUnreadable, so
perception must come from another driver. It also cannot locate targets,
so CLICK raises TargetNotFound. A caller can still advance flows with
PRESS sequences such as Tab and Enter, and with FILL typing at the
current focus, which is often enough to finish a form when every richer
backend has failed.
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
    """Pure keyboard action backend, backed by a lazily imported pyautogui.

    Attributes
    ----------
    name : str
        Driver id, ``"keyboard"``.
    surface : None
        None, because this driver is surface agnostic: it acts on
        whatever OS window currently holds focus.
    """

    name = "keyboard"
    surface = None  # acts on the focused OS window

    @staticmethod
    def _pyautogui() -> Any:
        """Import pyautogui on demand.

        Returns
        -------
        Any
            The imported ``pyautogui`` module.

        Raises
        ------
        TargetNotFound
            If pyautogui cannot be imported, so that a missing optional
            dependency reads as an ordinary failed rung the ladder can
            escalate past rather than a crash.
        """
        try:
            import pyautogui  # lazy

            return pyautogui
        except Exception as exc:
            raise TargetNotFound(
                f"pyautogui unavailable: {exc}"
            ) from exc

    async def observe(self) -> Observation:
        """Refuse to perceive, since this backend has no perception.

        Returns
        -------
        Observation
            Never returns normally.

        Raises
        ------
        SurfaceUnreadable
            Always. Perception must come from another driver.
        """
        raise SurfaceUnreadable(
            "keyboard driver cannot perceive, use another driver for observation"
        )

    async def act(self, action: Action) -> Optional[Element]:
        """Perform a keyboard-only action at the current focus.

        Parameters
        ----------
        action : Action
            The action to perform. Only PRESS and FILL are supported.

        Returns
        -------
        Element or None
            Always None, because this driver never resolves an element.

        Raises
        ------
        TargetNotFound
            If PRESS carries no key chord, if FILL carries no value, if
            pyautogui is unavailable, or if the action kind requires
            locating a target.

        Notes
        -----
        FILL splits on newlines and presses Enter between segments
        rather than typing the newline character, because many native
        toolkits treat a typed newline as a form submission at
        unpredictable points. Typing is paced with a small per character
        interval so that toolkits with input throttling do not drop
        characters.
        """
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
        """Report that this backend cannot capture pixels.

        Returns
        -------
        bytes or None
            Always None.
        """
        return None
