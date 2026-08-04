"""Driver protocol and the fallback ladder.

A Driver is one perception/action backend (DOM, accessibility tree,
vision, keyboard). The model never chooses a driver; the World routes by
surface and the ladder escalates mechanically on failure, so no LLM
calls are spent on fallback.

Notes
-----
Contract rules that every driver implementation must honour:

* Resolve and act is atomic inside ``act()``: never resolve now, click
  later. A reference captured before a repaint is already stale.
* ``act()`` reports what the backend did; it does NOT verify effect.
  Effect verification (observe, act, observe, diff) is the World's job,
  so that it is uniform across every driver.
* Failures are typed ``OrbitError`` raises, never status dicts.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from ..types import (
    Action,
    ActionResult,
    Element,
    Observation,
    OrbitError,
    SurfaceUnreadable,
    TargetNotFound,
    TargetObstructed,
    TargetUnresolvable,
    diff_observations,
)


def is_web_target(target: Optional[str]) -> bool:
    """Report whether a navigate target is a web address.

    Accepts either an explicit URL scheme or a bare ``host[:port][/path]``
    such as ``localhost:3000`` or ``example.com``.

    Parameters
    ----------
    target : str or None
        The navigate target string to classify. May be None or blank.

    Returns
    -------
    bool
        True if the target should be routed to the DOM driver, False if
        it should be treated as a native application target.

    Notes
    -----
    This is the single definition that both the dom driver (which claims
    web targets) and the accessibility driver (which claims the rest)
    route by. Keeping it in one place avoids per-driver string sniffing
    drifting apart.
    """
    if not target or not target.strip():
        return False
    from urllib.parse import urlparse
    t = target.strip()
    if urlparse(t).scheme in ("http", "https", "file", "about", "data"):
        return True
    if " " in t:
        return False
    head = t.split("/")[0]
    return head == "localhost" or head.startswith("localhost:") or (
        "." in head and not head.endswith(".")
    )


@runtime_checkable
class Driver(Protocol):
    """One perception and action backend.

    Attributes
    ----------
    name : str
        Short id used in ``ActionResult.strategy`` and in probe caches.
    surface : str or None
        Which surface this driver acts on: ``"web"`` (dom),
        ``"native"`` (accessibility tree), or None for surface agnostic
        action fallbacks (keyboard and vision act on whatever OS window
        is focused).

    Notes
    -----
    A driver bound to one surface is never a fallback rung for a
    different one. The dom driver must not "help" a native task by
    typing into a background browser page, which is why ``surface`` is
    part of the protocol rather than an implementation detail.
    """

    name: str

    surface: Optional[str] = None

    async def observe(self) -> Observation:
        """Capture the focused surface.

        Returns
        -------
        Observation
            Snapshot of the surface this driver perceives.

        Raises
        ------
        SurfaceUnreadable
            If this backend cannot perceive the surface at all.
        """
        ...

    async def act(self, action: Action) -> Optional[Element]:
        """Resolve the target (if any) and perform the action atomically.

        Parameters
        ----------
        action : Action
            The action to perform, including its target description and
            any value.

        Returns
        -------
        Element or None
            The element acted on, or None for targetless actions such as
            PRESS.

        Raises
        ------
        TargetNotFound
            If no element matches the action's target description.
        TargetObstructed
            If the element was found but could not be acted on.

        Notes
        -----
        Resolution and action happen in one step by contract. Returning a
        handle for a caller to act on later would reintroduce staleness
        across repaints.
        """
        ...

    async def screenshot(self) -> Optional[bytes]:
        """Capture the surface as an image.

        Returns
        -------
        bytes or None
            PNG bytes, or None if this backend cannot capture pixels.
        """
        ...


@dataclass
class CapabilityScore:
    """Result of probing one surface through one driver.

    Attributes
    ----------
    driver : str
        Name of the driver that produced the probe.
    usable : bool
        Whether the driver perceives the surface well enough to be the
        primary perception backend. Default is False.
    element_count : int
        Number of elements the driver reported. Default is 0.
    label_coverage : float
        Fraction of interactive elements that carry names. Default is
        0.0.
    bounds_sane : float
        Fraction of elements with sane coordinates. Default is 0.0.
    score : float
        Weighted quality score combining the three measures above.
        Default is 0.0.
    """

    driver: str
    usable: bool = False
    element_count: int = 0
    label_coverage: float = 0.0
    bounds_sane: float = 0.0
    score: float = 0.0

    @staticmethod
    def from_observation(driver: str, obs: Observation, surface_w: float = 0, surface_h: float = 0) -> "CapabilityScore":
        """Score an observation to decide whether a driver can see a surface.

        Parameters
        ----------
        driver : str
            Name of the driver that produced the observation.
        obs : Observation
            The probe observation to score.
        surface_w : float, optional
            Surface width in pixels, used to sanity check element
            bounds. Default is 0, which disables the width check.
        surface_h : float, optional
            Surface height in pixels, used to sanity check element
            bounds. Default is 0, which disables the height check.

        Returns
        -------
        CapabilityScore
            The populated score. An observation with no elements yields
            an unusable score immediately.

        Notes
        -----
        The weighting favours element count only up to a saturation
        point of fifteen elements, so a large but unlabeled tree does not
        outrank a small, well labeled one. When surface dimensions are
        unknown, bounds are counted as sane whenever they are present at
        all.
        """
        els = obs.elements
        if not els:
            return CapabilityScore(driver=driver, usable=False)
        labeled = sum(1 for e in els if e.name.strip())
        if surface_w and surface_h:
            sane = sum(1 for e in els if e.bounds and e.bounds.sane_within(surface_w, surface_h))
        else:
            sane = sum(1 for e in els if e.bounds is not None)
        cs = CapabilityScore(
            driver=driver,
            element_count=len(els),
            label_coverage=labeled / len(els),
            bounds_sane=sane / len(els),
        )
        cs.score = 0.4 * min(1.0, len(els) / 15) + 0.4 * cs.label_coverage + 0.2 * cs.bounds_sane
        cs.usable = cs.score >= 0.25
        return cs


@dataclass
class LadderOutcome:
    """The successful result of a ladder run plus the rungs it burned through.

    Attributes
    ----------
    result : ActionResult
        The verified result from the rung that landed.
    errors : List[OrbitError]
        Typed failures from every rung that was tried and escalated
        past, in order. Empty when the first rung succeeded.

    Notes
    -----
    Failed rungs are retained rather than discarded because the eventual
    error message (or the journal entry for a success) is far more
    useful when it can explain what was attempted first.
    """

    result: ActionResult
    errors: List[OrbitError] = field(default_factory=list)


async def _observe_settled(
    driver: Driver,
    max_polls: int = 6,
    interval: float = 0.15,
) -> Observation:
    """Observe until the surface stops changing, then return it.

    Parameters
    ----------
    driver : Driver
        The perception driver to observe through.
    max_polls : int, optional
        Maximum number of additional observations to take before giving
        up on settling. Default is 6.
    interval : float, optional
        Seconds to wait between consecutive observations. Default is
        0.15.

    Returns
    -------
    Observation
        The first observation whose content hash matched its
        predecessor, or the latest observation if the surface never
        settled within the poll bound.

    Notes
    -----
    A post action observation can be captured mid repaint (a GTK toolkit
    rebuilding its tree, a browser still laying out), which reads as
    either a spurious diff or no diff at all. Instead of a fixed sleep
    and hope, poll until two consecutive observations agree by content
    hash, meaning the surface has settled. The poll count is bounded so
    that a genuinely animating page still returns rather than hanging.
    """
    obs = await driver.observe()
    for _ in range(max_polls):
        await asyncio.sleep(interval)
        nxt = await driver.observe()
        if nxt.content_hash == obs.content_hash:
            return nxt          # stable: two observations agree
        obs = nxt
    return obs                  # still moving after the bound, use latest


async def run_ladder(
    drivers: Sequence[Driver],
    action: Action,
    observe_via: Driver,
    max_rounds: int = 1,
    ensure: Optional[Any] = None,
) -> LadderOutcome:
    """Run the fallback ladder with built in effect verification.

    Tries each driver in order until one both reports success and
    demonstrably changes the world.

    Parameters
    ----------
    drivers : Sequence[Driver]
        Candidate backends, in escalation order. The first is the
        preferred rung.
    action : Action
        The action to perform on each attempt.
    observe_via : Driver
        The primary perception driver used to capture the before and
        after observations. This is deliberately independent of the
        acting driver.
    max_rounds : int, optional
        How many times to cycle through the full driver sequence.
        Default is 1.
    ensure : Any, optional
        Async callable taking a driver name, invoked immediately before
        that rung acts. Used to start a backend lazily so that unused
        rungs never pay for setup. Default is None.

    Returns
    -------
    LadderOutcome
        The verified result plus every typed error escalated past.

    Raises
    ------
    TargetUnresolvable
        When every rung in every round has been exhausted without an
        action landing.

    Notes
    -----
    After each attempt the ladder re observes through ``observe_via``
    and diffs against the previous baseline. An action that reports
    success but produces no observable change, when the action kind
    expects one, did NOT actually land, and the ladder escalates to the
    next rung. This is why verification lives here rather than in each
    driver: every backend gets the same, uniform proof of effect, and no
    LLM call is spent deciding whether to fall back.

    The baseline is advanced to the latest observation after each
    non-landing attempt, so a slow surface that settles between rungs
    does not produce a false positive on a later one.
    """
    errors: List[OrbitError] = []
    attempts = 0
    before = await observe_via.observe()

    for _ in range(max_rounds):
        for driver in drivers:
            attempts += 1
            start = time.monotonic()
            try:
                if ensure is not None:
                    await ensure(driver.name)  # start this rung only now
                element = await driver.act(action)
            except OrbitError as exc:
                # Any typed failure from a backend is a failed rung to
                # escalate past, never a run-killing crash.
                errors.append(exc)
                continue

            after = await _observe_settled(observe_via)
            diff = diff_observations(before, after)
            duration = (time.monotonic() - start) * 1000

            if diff.changed or not action.expects_effect:
                return LadderOutcome(
                    result=ActionResult(
                        landed=True,
                        action=action,
                        strategy=driver.name,
                        diff=diff,
                        element=element,
                        attempts=attempts,
                        duration_ms=duration,
                    ),
                    errors=errors,
                )

            # API said success, world says nothing happened: stale target.
            errors.append(
                OrbitError(
                    f"{driver.name}: action reported success but produced no observable change",
                )
            )
            before = after  # keep freshest baseline

    raise TargetUnresolvable(
        f"all strategies exhausted for {action.kind.value} on {action.target!r}",
        attempts=attempts,
        errors=[str(e) for e in errors],
    )
