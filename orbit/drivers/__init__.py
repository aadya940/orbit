"""Driver backends for Orbit v2.

Exports are lazy so ``import orbit`` works with zero optional deps
installed (playwright/patchright, requests, pyautogui are all imported
only inside driver methods).

Notes
-----
Lazy attribute resolution via module-level ``__getattr__`` keeps the
import graph flat: naming a driver in ``__all__`` does not drag its
optional third-party dependency into every ``import orbit``.
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
    """Resolve a lazily exported driver class on first attribute access.

    Parameters
    ----------
    name : str
        Attribute name requested on this module.

    Returns
    -------
    type
        The driver class imported from its defining module.

    Raises
    ------
    AttributeError
        If ``name`` is not one of the lazily exported drivers.

    Notes
    -----
    Deferring the import until access is what allows the package to be
    imported without playwright, patchright, requests or pyautogui
    installed.
    """
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def default_drivers(browser: str = "chrome") -> Dict[str, "Driver"]:
    """Instantiate (but do not start) the standard driver set, keyed by name.

    Parameters
    ----------
    browser : str, optional
        Browser engine passed through to the DOM driver. Default is
        ``"chrome"``.

    Returns
    -------
    Dict[str, Driver]
        Mapping of ``driver.name`` to a fresh, unstarted driver instance,
        in ladder order: dom, accessibility, keyboard, vision.

    Notes
    -----
    Construction is deliberately side effect free. Browsers and
    accessibility connections are only established when a rung is
    actually reached, so an unused driver never costs a process launch.
    """
    from .accessibility import AccessibilityDriver
    from .dom import DomDriver
    from .keyboard import KeyboardDriver
    from .vision import VisionDriver

    drivers = [DomDriver(browser=browser), AccessibilityDriver(),
               KeyboardDriver(), VisionDriver()]
    return {d.name: d for d in drivers}
