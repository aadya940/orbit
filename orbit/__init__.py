"""Orbit v2: computer-use automation with typed contracts.

This package exposes the public surface of Orbit: :class:`Session` and its
verb methods, the :class:`Policy` side-effect gate, the typed action and
run contracts, and the :class:`OrbitError` hierarchy.

Examples
--------
>>> import orbit
>>> orbit.__version__
'2.0.0a0'
"""

from contextlib import asynccontextmanager

from .policy import Policy
from .session import Session
from .types import (
    Action,
    ActionHadNoEffect,
    ActionKind,
    ActionResult,
    BudgetExhausted,
    NeedsHuman,
    OrbitError,
    OutputInvalid,
    PolicyDenied,
    RunResult,
    RunStatus,
    SurfaceUnreadable,
    TargetNotFound,
    TargetObstructed,
    TargetUnresolvable,
)

__version__ = "2.0.0a0"


@asynccontextmanager
async def session(**kwargs):
    """Open a :class:`Session` as an async context manager.

    Convenience wrapper so callers can write
    ``async with orbit.session(...) as s:`` without importing
    :class:`Session` directly.

    Parameters
    ----------
    **kwargs : Any
        Forwarded verbatim to the :class:`Session` constructor, for example
        ``llm``, ``policy``, ``max_steps``, ``drivers``, ``tools``,
        ``include_default_tools`` and ``browser``.

    Yields
    ------
    Session
        An entered session. Its drivers are stopped on exit, but only the
        ones that were actually started.

    Examples
    --------
    >>> async def main():
    ...     async with orbit.session(llm="gpt-4o") as s:
    ...         return await s.check("the page has loaded")
    """
    async with Session(**kwargs) as s:
        yield s


__all__ = [
    "Session", "session", "Policy",
    "Action", "ActionKind", "ActionResult",
    "RunResult", "RunStatus",
    "OrbitError", "TargetNotFound", "TargetObstructed", "TargetUnresolvable",
    "ActionHadNoEffect", "SurfaceUnreadable", "BudgetExhausted",
    "NeedsHuman", "PolicyDenied", "OutputInvalid",
    "__version__",
]
