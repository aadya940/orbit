"""Driver backends for Orbit v2.

Exports are lazy so ``import orbit`` works with zero optional deps
installed (playwright/patchright, requests, pyautogui are all imported
only inside driver methods).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from .base import CapabilityScore, Driver, LadderOutcome, run_ladder

if TYPE_CHECKING:  # pragma: no cover
    from .accessibility import AccessibilityDriver
    from .dom import DomDriver
    from .keyboard import KeyboardDriver
    from .vision import VisionDriver

__all__ = [
    "Driver",
    "CapabilityScore",
    "LadderOutcome",
    "run_ladder",
    "DomDriver",
    "AccessibilityDriver",
    "KeyboardDriver",
    "VisionDriver",
    "default_drivers",
]

_LAZY = {
    "DomDriver": ("orbit.drivers.dom", "DomDriver"),
    "AccessibilityDriver": ("orbit.drivers.accessibility", "AccessibilityDriver"),
    "KeyboardDriver": ("orbit.drivers.keyboard", "KeyboardDriver"),
    "VisionDriver": ("orbit.drivers.vision", "VisionDriver"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def default_drivers(browser: str = "chrome") -> Dict[str, "Driver"]:
    """Instantiate (but do not start) the standard driver set, keyed by name."""
    from .accessibility import AccessibilityDriver
    from .dom import DomDriver
    from .keyboard import KeyboardDriver
    from .vision import VisionDriver

    drivers = [DomDriver(browser=browser), AccessibilityDriver(),
               KeyboardDriver(), VisionDriver()]
    return {d.name: d for d in drivers}
