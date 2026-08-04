"""Orbit v2 — computer-use automation with typed contracts."""

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
    """`async with orbit.session(...) as s:` convenience wrapper."""
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
