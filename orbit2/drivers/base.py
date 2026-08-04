"""Driver protocol and the fallback ladder.

A Driver is one perception/action backend (DOM, accessibility tree,
vision, keyboard). The model never chooses a driver; the World routes by
surface and the ladder escalates mechanically on failure — no LLM calls
are spent on fallback.

Contract rules:
- resolve-and-act is atomic inside `act()`: never resolve now, click later.
- `act()` reports what the backend did; it does NOT verify effect. Effect
  verification (observe → act → observe → diff) is the World's job, so it
  is uniform across every driver.
- Failures are typed OrbitError raises, never status dicts.
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


@runtime_checkable
class Driver(Protocol):
    """One perception/action backend."""

    #: short id used in ActionResult.strategy and probe caches
    name: str

    async def observe(self) -> Observation:
        """Capture the focused surface. Raise SurfaceUnreadable if blind."""
        ...

    async def act(self, action: Action) -> Optional[Element]:
        """Resolve the target (if any) and perform the action atomically.

        Returns the element acted on (None for targetless actions like
        PRESS). Raises TargetNotFound / TargetObstructed on failure.
        """
        ...

    async def screenshot(self) -> Optional[bytes]:
        """PNG bytes, or None if this backend cannot capture pixels."""
        ...


@dataclass
class CapabilityScore:
    """Result of probing one surface through one driver."""

    driver: str
    usable: bool = False
    element_count: int = 0
    label_coverage: float = 0.0   # fraction of interactive elements with names
    bounds_sane: float = 0.0      # fraction with sane coordinates
    score: float = 0.0

    @staticmethod
    def from_observation(driver: str, obs: Observation, surface_w: float = 0, surface_h: float = 0) -> "CapabilityScore":
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
    result: ActionResult
    errors: List[OrbitError] = field(default_factory=list)


async def run_ladder(
    drivers: Sequence[Driver],
    action: Action,
    observe_via: Driver,
    max_rounds: int = 1,
    ensure: Optional[Any] = None,
) -> LadderOutcome:
    """The fallback ladder with built-in effect verification.

    Tries each driver in order. After each attempt, re-observes through
    `observe_via` (the primary perception driver) and diffs. An action
    that "succeeds" but produces no observable change when one was
    expected did NOT land — escalate to the next rung.

    Raises TargetUnresolvable when every rung is exhausted.
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
                # escalate past — never a run-killing crash.
                errors.append(exc)
                continue

            after = await observe_via.observe()
            diff = diff_observations(before, after)

            if not diff.changed and action.expects_effect:
                # Never escalate on a single no-effect verdict: post-action
                # observations can be partial while the toolkit rebuilds its
                # tree (verified live on GTK — a re-clicked '8' turned 7x8
                # into 7x88). Settle, observe again, and only treat the
                # action as not-landed if two observations agree.
                await asyncio.sleep(0.5)
                after = await observe_via.observe()
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
