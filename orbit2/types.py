"""Core typed contracts for Orbit v2.

Every component speaks these types. No free-form status dicts anywhere:
tools raise OrbitError subclasses, actions return ActionResult, runs
return RunResult.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OrbitError(Exception):
    """Base for all typed Orbit failures."""

    code: str = "orbit_error"

    def __init__(self, message: str = "", **context: Any) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        self.context = context


class TargetNotFound(OrbitError):
    code = "target_not_found"


class TargetObstructed(OrbitError):
    """Element exists but is covered / invisible / disabled / off-screen."""

    code = "target_obstructed"


class TargetUnresolvable(OrbitError):
    """Whole fallback ladder exhausted for this target."""

    code = "target_unresolvable"


class ActionHadNoEffect(OrbitError):
    """The action API succeeded but the observed state did not change."""

    code = "action_had_no_effect"


class SurfaceUnreadable(OrbitError):
    """No usable perception channel for the focused surface."""

    code = "surface_unreadable"


class BudgetExhausted(OrbitError):
    code = "budget_exhausted"


class NeedsHuman(OrbitError):
    """Agent-initiated escalation (CAPTCHA, blocked step, approval)."""

    code = "needs_human"


class PolicyDenied(OrbitError):
    code = "policy_denied"


class OutputInvalid(OrbitError):
    """Final output failed schema validation after retries."""

    code = "output_invalid"


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

class Source(str, Enum):
    """Which perception channel an element was seen through."""

    TREE = "tree"      # OS accessibility tree
    DOM = "dom"        # live DOM via CDP/Playwright
    VISION = "vision"  # screenshot grounding


@dataclass
class Bounds:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def sane_within(self, w: float, h: float) -> bool:
        """Coordinate sanity: bounds must fall inside the surface rect."""
        cx, cy = self.center
        return 0 <= cx <= w and 0 <= cy <= h and self.width >= 0 and self.height >= 0


@dataclass
class Element:
    """A fused interactive element with provenance and confidence.

    Cross-confirmed (seen by 2+ sources) elements are trustworthy.
    Tree-only elements are suspect (possibly stale); vision-only means
    unlabeled/custom-drawn and needs pixel targeting.
    """

    role: str
    name: str
    bounds: Optional[Bounds] = None
    provenance: frozenset = frozenset()          # set[Source]
    ref: Dict[str, Any] = field(default_factory=dict)  # backend handles (node id, selector, ...)
    enabled: bool = True
    focused: bool = False
    value: Optional[str] = None
    # Secondary accessible text (help_text, tooltip, keyboard shortcut).
    # Crucial for symbol-labeled controls: a calculator's '×' button often
    # carries hint "Multiply [*]" — the only place its purpose is in words.
    hint: Optional[str] = None

    @property
    def confidence(self) -> float:
        n = len(self.provenance)
        return {0: 0.0, 1: 0.5, 2: 0.85}.get(n, 1.0)

    @property
    def key(self) -> str:
        b = self.bounds
        pos = f"{int(b.x)},{int(b.y)}" if b else "?"
        return f"{self.role}|{self.name}|{pos}"


@dataclass
class Observation:
    """Fused snapshot of the focused surface."""

    surface: str                                  # e.g. "browser:main", "window:1234"
    kind: str                                     # "browser" | "native"
    title: str = ""
    url: Optional[str] = None
    elements: List[Element] = field(default_factory=list)
    text: str = ""                                # visible text content (may be truncated upstream)
    modal_count: int = 0
    focused_key: Optional[str] = None
    captured_at: float = field(default_factory=time.time)

    @property
    def content_hash(self) -> str:
        payload = "|".join(
            [self.surface, self.title, self.url or "", str(self.modal_count)]
            + sorted(e.key for e in self.elements)
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class StateDiff:
    """Generalized consequences block: what changed after an action."""

    changed: bool = False
    url_changed: bool = False
    title_changed: bool = False
    new_modals: int = 0
    closed_modals: int = 0
    appeared: List[str] = field(default_factory=list)   # element keys
    disappeared: List[str] = field(default_factory=list)
    value_changes: List[str] = field(default_factory=list)
    focus_changed: bool = False

    def summary(self) -> str:
        if not self.changed:
            return "no observable change"
        parts: List[str] = []
        if self.url_changed:
            parts.append("url changed")
        if self.title_changed:
            parts.append("title changed")
        if self.new_modals:
            parts.append(f"{self.new_modals} modal(s) opened")
        if self.closed_modals:
            parts.append(f"{self.closed_modals} modal(s) closed")
        if self.appeared:
            parts.append(f"{len(self.appeared)} element(s) appeared")
        if self.disappeared:
            parts.append(f"{len(self.disappeared)} element(s) disappeared")
        if self.value_changes:
            parts.append(f"values changed: {', '.join(self.value_changes[:5])}")
        if self.focus_changed:
            parts.append("focus moved")
        return "; ".join(parts)


def diff_observations(before: Observation, after: Observation) -> StateDiff:
    """Pure diff between two observations. The referee for effect verification."""
    b_keys = {e.key for e in before.elements}
    a_keys = {e.key for e in after.elements}
    b_vals = {e.key: e.value for e in before.elements}
    value_changes = [
        e.key for e in after.elements
        if e.key in b_vals and b_vals[e.key] != e.value
    ]
    d = StateDiff(
        url_changed=before.url != after.url,
        title_changed=before.title != after.title,
        new_modals=max(0, after.modal_count - before.modal_count),
        closed_modals=max(0, before.modal_count - after.modal_count),
        appeared=sorted(a_keys - b_keys),
        disappeared=sorted(b_keys - a_keys),
        value_changes=value_changes,
        focus_changed=before.focused_key != after.focused_key,
    )
    d.changed = bool(
        d.url_changed or d.title_changed or d.new_modals or d.closed_modals
        or d.appeared or d.disappeared or d.value_changes or d.focus_changed
    )
    return d


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class ActionKind(str, Enum):
    CLICK = "click"
    FILL = "fill"       # set value on a field
    SELECT = "select"   # choose option in a dropdown
    PRESS = "press"     # key or chord, no target needed
    SCROLL = "scroll"
    NAVIGATE = "navigate"  # url or app launch


@dataclass
class Action:
    kind: ActionKind
    target: Optional[str] = None   # natural-language target description
    value: Optional[str] = None    # text to type / option / url / key chord
    # Some effects legitimately produce no diff (e.g. copy). Callers may
    # declare that so verification doesn't false-positive.
    expects_effect: bool = True


@dataclass
class ActionResult:
    landed: bool
    action: Action
    strategy: str = ""              # which ladder rung succeeded ("tree", "dom", "vision", "keyboard")
    diff: StateDiff = field(default_factory=StateDiff)
    element: Optional[Element] = None
    attempts: int = 1
    duration_ms: float = 0.0
    note: str = ""


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class RunStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NEEDS_HUMAN = "needs_human"
    TIMEOUT = "timeout"


@dataclass
class RunResult:
    status: RunStatus
    output: Any = None                 # validated pydantic instance, bool (Check), str, or None
    steps_used: int = 0
    error: Optional[OrbitError] = None
    journal: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.SUCCESS


def validate_output(raw: Any, schema: Optional[Type[BaseModel]]) -> Any:
    """Strict validation — raises OutputInvalid, never coerces silently."""
    if schema is None:
        return raw
    try:
        if isinstance(raw, schema):
            return raw
        if isinstance(raw, dict):
            return schema.model_validate(raw)
        if isinstance(raw, str):
            return schema.model_validate_json(raw)
    except Exception as exc:  # pydantic ValidationError et al.
        raise OutputInvalid(str(exc), raw=raw) from exc
    raise OutputInvalid(f"cannot validate {type(raw).__name__} against {schema.__name__}", raw=raw)
