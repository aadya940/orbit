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
    """Base for all typed Orbit failures.

    Every failure mode carries a stable machine-readable :attr:`code` plus
    free-form context, so callers can branch on the code without parsing
    the message.

    Attributes
    ----------
    code : str
        Stable identifier for the failure mode. Subclasses override it.
    message : str
        Human-readable description. Falls back to :attr:`code` when the
        caller supplies no message.
    context : dict
        Arbitrary structured detail about the failure, such as the target
        that could not be resolved.

    Examples
    --------
    >>> err = OrbitError("something broke", surface="browser:main")
    >>> err.code, err.context["surface"]
    ('orbit_error', 'browser:main')
    """

    code: str = "orbit_error"

    def __init__(self, message: str = "", **context: Any) -> None:
        """Build the error.

        Parameters
        ----------
        message : str, optional
            Human-readable description. Default is the empty string, in
            which case :attr:`code` is used as the message.
        **context : Any
            Structured detail attached to :attr:`context`.
        """
        super().__init__(message or self.code)
        self.message = message or self.code
        self.context = context


class TargetNotFound(OrbitError):
    """No element matching the requested target exists on the surface.

    Attributes
    ----------
    code : str
        Always ``"target_not_found"``.
    """

    code = "target_not_found"


class TargetObstructed(OrbitError):
    """Element exists but cannot be acted on.

    Raised when the element is covered, invisible, disabled or off-screen.

    Attributes
    ----------
    code : str
        Always ``"target_obstructed"``.
    """

    code = "target_obstructed"


class TargetUnresolvable(OrbitError):
    """The whole fallback ladder was exhausted for this target.

    Attributes
    ----------
    code : str
        Always ``"target_unresolvable"``.

    Notes
    -----
    This is the terminal targeting failure: every rung (tree, DOM, vision,
    keyboard) was tried and none landed the action.
    """

    code = "target_unresolvable"


class ActionHadNoEffect(OrbitError):
    """The action API succeeded but the observed state did not change.

    Attributes
    ----------
    code : str
        Always ``"action_had_no_effect"``.

    Notes
    -----
    A backend reporting success is not evidence that anything happened: a
    click can be swallowed by an overlay, and the API still returns fine.
    Effect verification diffs the surface before and after so a silently
    lost action becomes a real failure instead of a false success.
    """

    code = "action_had_no_effect"


class SurfaceUnreadable(OrbitError):
    """No usable perception channel for the focused surface.

    Attributes
    ----------
    code : str
        Always ``"surface_unreadable"``.
    """

    code = "surface_unreadable"


class BudgetExhausted(OrbitError):
    """The run consumed its entire step budget.

    Attributes
    ----------
    code : str
        Always ``"budget_exhausted"``.
    """

    code = "budget_exhausted"


class NeedsHuman(OrbitError):
    """Agent-initiated escalation to a person.

    Raised for CAPTCHAs, logins, genuine ambiguity, or a destructive step
    that needs approval.

    Attributes
    ----------
    code : str
        Always ``"needs_human"``.
    """

    code = "needs_human"


class PolicyDenied(OrbitError):
    """A side-effecting action was refused by :class:`~orbit.policy.Policy`.

    Attributes
    ----------
    code : str
        Always ``"policy_denied"``.
    """

    code = "policy_denied"


class OutputInvalid(OrbitError):
    """Final output failed schema validation after retries.

    Attributes
    ----------
    code : str
        Always ``"output_invalid"``.
    """

    code = "output_invalid"


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

class Source(str, Enum):
    """Which perception channel an element was seen through.

    Attributes
    ----------
    TREE : str
        The OS accessibility tree.
    DOM : str
        The live DOM, read via CDP or Playwright.
    VISION : str
        Screenshot grounding.
    """

    TREE = "tree"      # OS accessibility tree
    DOM = "dom"        # live DOM via CDP/Playwright
    VISION = "vision"  # screenshot grounding


@dataclass
class Bounds:
    """Axis-aligned rectangle in surface coordinates.

    Attributes
    ----------
    x : float
        Left edge.
    y : float
        Top edge.
    width : float
        Horizontal extent.
    height : float
        Vertical extent.

    Examples
    --------
    >>> Bounds(10, 20, 100, 40).center
    (60.0, 40.0)
    """

    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Tuple[float, float]:
        """Midpoint of the rectangle.

        Returns
        -------
        tuple of (float, float)
            The ``(x, y)`` centre, which is where a pointer action aims.
        """
        return (self.x + self.width / 2, self.y + self.height / 2)

    def sane_within(self, w: float, h: float) -> bool:
        """Check that these bounds fall inside a surface rectangle.

        Parameters
        ----------
        w : float
            Surface width.
        h : float
            Surface height.

        Returns
        -------
        bool
            True when the centre lies within the surface and the extents
            are non-negative.

        Notes
        -----
        Backends occasionally report stale or garbage geometry, for example
        an element at a negative offset or far past the viewport. Clicking
        such coordinates hits the wrong thing, so bounds are sanity-checked
        before they are trusted for pixel targeting.

        Examples
        --------
        >>> Bounds(10, 20, 100, 40).sane_within(800, 600)
        True
        >>> Bounds(-500, 20, 10, 10).sane_within(800, 600)
        False
        """
        cx, cy = self.center
        return 0 <= cx <= w and 0 <= cy <= h and self.width >= 0 and self.height >= 0


@dataclass
class Element:
    """A fused interactive element with provenance and confidence.

    Attributes
    ----------
    role : str
        Accessible role, for example ``"button"`` or ``"textbox"``.
    name : str
        Accessible name, that is, the visible or announced label.
    bounds : Bounds or None, optional
        Screen geometry, when a channel reported it. Default is None.
    provenance : frozenset, optional
        The set of :class:`Source` values that observed this element.
        Default is the empty frozenset.
    ref : dict, optional
        Backend handles for re-addressing the element, such as a node id
        or a selector. Default is an empty dict.
    enabled : bool, optional
        Whether the element accepts interaction. Default is True.
    focused : bool, optional
        Whether the element currently holds focus. Default is False.
    value : str or None, optional
        Current value for fields and other value-bearing controls.
        Default is None.
    hint : str or None, optional
        Secondary accessible text such as help text, a tooltip or a
        keyboard shortcut. Default is None.

    Notes
    -----
    Provenance drives trust. An element cross-confirmed by two or more
    sources is trustworthy. A tree-only element is suspect, since the
    accessibility tree can lag behind the real UI. A vision-only element is
    unlabelled or custom-drawn and needs pixel targeting.

    The hint matters for symbol-labelled controls: a calculator's
    multiplication button often carries the hint ``"Multiply [*]"``, which
    is the only place its purpose is spelled out in words.
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
    # Crucial for symbol-labeled controls: a calculator's multiply button
    # often carries hint "Multiply [*]", the only place its purpose is
    # written out in words.
    hint: Optional[str] = None

    @property
    def confidence(self) -> float:
        """How much to trust this element, based on cross-confirmation.

        Returns
        -------
        float
            0.0 for no source, 0.5 for a single source, 0.85 for two, and
            1.0 for three or more.
        """
        n = len(self.provenance)
        return {0: 0.0, 1: 0.5, 2: 0.85}.get(n, 1.0)

    @property
    def key(self) -> str:
        """Stable identity string used to diff observations.

        Returns
        -------
        str
            ``"role|name|x,y"``, with ``"?"`` in place of the position when
            the element has no bounds. Coordinates are truncated to whole
            pixels so sub-pixel jitter does not register as a change.
        """
        b = self.bounds
        pos = f"{int(b.x)},{int(b.y)}" if b else "?"
        return f"{self.role}|{self.name}|{pos}"


@dataclass
class Observation:
    """Fused snapshot of the focused surface.

    Attributes
    ----------
    surface : str
        Identifier of the observed surface, for example
        ``"browser:main"`` or ``"window:1234"``.
    kind : str
        Surface family, either ``"browser"`` or ``"native"``.
    title : str, optional
        Window or document title. Default is the empty string.
    url : str or None, optional
        Current URL for browser surfaces. Default is None.
    elements : list of Element, optional
        Interactive elements in render order. The index into this list is
        the ref the model points at. Default is an empty list.
    text : str, optional
        Visible text content, possibly truncated upstream. Default is the
        empty string.
    modal_count : int, optional
        Number of open modal dialogs. Default is 0.
    focused_key : str or None, optional
        :attr:`Element.key` of the focused element. Default is None.
    captured_at : float, optional
        Unix timestamp of capture. Defaults to the current time.
    """

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
        """Short digest identifying the observed state.

        Returns
        -------
        str
            The first 16 hex characters of a SHA-256 over the surface,
            title, URL, modal count and sorted element keys.

        Notes
        -----
        The loop compares hashes to decide whether to re-render the full
        observation or send only a delta, so the digest deliberately covers
        structure rather than free text. Element keys are sorted, which
        makes the hash insensitive to backend ordering churn.
        """
        payload = "|".join(
            [self.surface, self.title, self.url or "", str(self.modal_count)]
            + sorted(e.key for e in self.elements)
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def by_ref(self, ref: int) -> Optional["Element"]:
        """Look up the element at a rendered ref index.

        Parameters
        ----------
        ref : int
            Index into :attr:`elements`, matching the bracketed number
            shown to the model.

        Returns
        -------
        Element or None
            The element, or None when the index is out of range.

        Notes
        -----
        Out-of-range refs return None instead of raising, because a model
        pointing at a stale index is an expected condition the loop reports
        back rather than a programming error.
        """
        if 0 <= ref < len(self.elements):
            return self.elements[ref]
        return None


@dataclass
class StateDiff:
    """Generalized consequences block: what changed after an action.

    Attributes
    ----------
    changed : bool, optional
        True when any other field indicates a change. Default is False.
    url_changed : bool, optional
        Whether the URL differs. Default is False.
    title_changed : bool, optional
        Whether the title differs. Default is False.
    new_modals : int, optional
        Count of modals opened. Default is 0.
    closed_modals : int, optional
        Count of modals closed. Default is 0.
    appeared : list of str, optional
        Element keys present only after the action. Default is an empty
        list.
    disappeared : list of str, optional
        Element keys present only before the action. Default is an empty
        list.
    value_changes : list of str, optional
        Keys of elements whose value changed. Default is an empty list.
    focus_changed : bool, optional
        Whether focus moved to a different element. Default is False.

    Notes
    -----
    This is the evidence behind effect verification. A backend reporting a
    successful click proves nothing, so the diff, not the API return value,
    decides whether an action actually landed.
    """

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
        """Render the diff as a short phrase for the model.

        Returns
        -------
        str
            A semicolon-separated description of what changed, or
            ``"no observable change"`` when nothing did. At most five
            value changes are named, to keep the line bounded.

        Examples
        --------
        >>> StateDiff().summary()
        'no observable change'
        >>> StateDiff(changed=True, url_changed=True).summary()
        'url changed'
        """
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
    """Compute the difference between two observations.

    Parameters
    ----------
    before : Observation
        Snapshot taken before the action.
    after : Observation
        Snapshot taken after the action.

    Returns
    -------
    StateDiff
        The consequences block, with :attr:`StateDiff.changed` set when
        any individual signal fired.

    Notes
    -----
    This function is pure and is the referee for effect verification: it is
    the single place that decides whether an action changed anything, which
    keeps that judgement out of the individual drivers.

    Examples
    --------
    >>> a = Observation(surface="s", kind="browser", title="One")
    >>> b = Observation(surface="s", kind="browser", title="Two")
    >>> diff_observations(a, b).summary()
    'title changed'
    """
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
    """The kinds of action a driver can perform.

    Attributes
    ----------
    CLICK : str
        Activate an element.
    FILL : str
        Set the value of a field.
    SELECT : str
        Choose an option in a dropdown.
    PRESS : str
        Send a key or chord. Needs no target.
    SCROLL : str
        Scroll the surface.
    NAVIGATE : str
        Open a URL or launch an application.
    """

    CLICK = "click"
    FILL = "fill"       # set value on a field
    SELECT = "select"   # choose option in a dropdown
    PRESS = "press"     # key or chord, no target needed
    SCROLL = "scroll"
    NAVIGATE = "navigate"  # url or app launch


@dataclass
class Action:
    """One requested interaction with the focused surface.

    Attributes
    ----------
    kind : ActionKind
        What to do.
    target : str or None, optional
        Natural-language description of the element. Default is None.
    value : str or None, optional
        Text to type, option to choose, URL to open, or key chord to
        press, depending on :attr:`kind`. Default is None.
    ref : int or None, optional
        Direct index into the last observation's element list. Default is
        None.
    expects_effect : bool, optional
        Whether the action should produce an observable state change.
        Default is True.

    Notes
    -----
    ``ref`` is the preferred way to address an element. The model is shown
    numbered elements and points at one, so the driver never has to re-find
    by fuzzy string what was already rendered to it. ``target`` remains the
    description-based path, used for durable workflows and as a fallback
    when no ref fits.

    Some actions legitimately produce no diff, a clipboard copy being the
    clearest case. Setting ``expects_effect`` to False declares that, so
    effect verification does not report a false failure.
    """

    kind: ActionKind
    target: Optional[str] = None   # natural-language target description
    value: Optional[str] = None    # text to type / option / url / key chord
    ref: Optional[int] = None
    expects_effect: bool = True


@dataclass
class ActionResult:
    """Outcome of attempting an :class:`Action`.

    Attributes
    ----------
    landed : bool
        Whether the action was performed and verified.
    action : Action
        The action that was attempted.
    strategy : str, optional
        Which ladder rung succeeded: ``"tree"``, ``"dom"``, ``"vision"``
        or ``"keyboard"``. Default is the empty string.
    diff : StateDiff, optional
        Observed consequences. Defaults to an empty diff.
    element : Element or None, optional
        The element that was acted on, when one was resolved. Default is
        None.
    attempts : int, optional
        Number of rungs tried, including the successful one. Default is 1.
    duration_ms : float, optional
        Wall-clock time spent. Default is 0.0.
    note : str, optional
        Extra detail, typically the failure message. Default is the empty
        string.
    """

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
    """Terminal state of a run.

    Attributes
    ----------
    SUCCESS : str
        The agent finished and its output validated.
    FAILED : str
        The run ended in an unrecoverable error.
    BUDGET_EXHAUSTED : str
        The step budget ran out first.
    NEEDS_HUMAN : str
        The agent escalated to a person.
    TIMEOUT : str
        The wall-clock deadline elapsed.
    """

    SUCCESS = "success"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NEEDS_HUMAN = "needs_human"
    TIMEOUT = "timeout"


@dataclass
class RunResult:
    """Everything a finished run produced.

    Attributes
    ----------
    status : RunStatus
        How the run ended.
    output : Any, optional
        The result: a validated pydantic instance, a bool for a check, a
        string, or None. Default is None.
    steps_used : int, optional
        Steps consumed from the budget. Default is 0.
    error : OrbitError or None, optional
        The failure, for any non-success status. Default is None.
    journal : list of dict, optional
        The full audit trail. Default is an empty list.

    Examples
    --------
    >>> RunResult(status=RunStatus.SUCCESS, output=42).ok
    True
    """

    status: RunStatus
    output: Any = None                 # validated pydantic instance, bool (Check), str, or None
    steps_used: int = 0
    error: Optional[OrbitError] = None
    journal: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the run succeeded.

        Returns
        -------
        bool
            True only when :attr:`status` is :attr:`RunStatus.SUCCESS`.
        """
        return self.status is RunStatus.SUCCESS


def validate_output(raw: Any, schema: Optional[Type[BaseModel]]) -> Any:
    """Validate a raw final output against an optional schema.

    Accepts an instance of the schema, a dict, or a JSON string, and
    returns a validated model instance.

    Parameters
    ----------
    raw : Any
        The output the agent produced.
    schema : type of pydantic.BaseModel or None
        Expected output model. When None, ``raw`` is returned unchanged.

    Returns
    -------
    Any
        ``raw`` itself when no schema was given, otherwise a validated
        instance of ``schema``.

    Raises
    ------
    OutputInvalid
        If validation fails, or if ``raw`` is of a type that cannot be
        validated against ``schema``.

    Notes
    -----
    Validation is strict on purpose: it raises rather than coercing
    silently. A quietly coerced output would hand the caller data that does
    not match the contract they asked for, and the mismatch would surface
    far from its cause. Raising instead lets the loop show the model the
    validation error and ask for a corrected output.

    Examples
    --------
    >>> validate_output({"a": 1}, None)
    {'a': 1}
    """
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
