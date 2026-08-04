"""World: per-session container replacing all globals.

Holds drivers, journal, policy, probe cache and the step budget. Two
Worlds share nothing, which is what makes everything testable."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from .drivers.base import CapabilityScore, Driver, run_ladder
from .journal import Journal
from .policy import Policy
from .types import (
    Action,
    ActionKind,
    ActionResult,
    BudgetExhausted,
    Observation,
    OrbitError,
    SurfaceUnreadable,
    TargetUnresolvable,
)


class World:
    def __init__(
        self,
        drivers: Dict[str, Driver],
        primary: Optional[str] = None,
        policy: Optional[Policy] = None,
        journal: Optional[Journal] = None,
        max_steps: int = 40,
        runtime: Optional[dict] = None,
    ) -> None:
        if not drivers:
            raise ValueError("World needs at least one driver")
        self.drivers: Dict[str, Driver] = dict(drivers)
        # When primary is given, respect it and don't auto-route. When it
        # isn't, route perception to whichever driver can actually see the
        # focused surface (browser -> dom, native window -> tree), re-picked
        # whenever the current one goes blind.
        self.auto_route: bool = primary is None
        self.primary: str = primary or next(iter(self.drivers))
        self.policy: Policy = policy or Policy()
        self.journal: Journal = journal or Journal()
        self.probe_cache: Dict[str, List[CapabilityScore]] = {}
        self.ladder_order: List[str] = list(self.drivers)
        self.max_steps: int = max_steps
        self.used: int = 0
        # Driver runtime (which backends are started, last surface hint)
        # is shared across the session's verbs so a native task doesn't
        # relaunch a browser on its second verb. Fresh dict = fresh session.
        self._runtime: dict = runtime if runtime is not None else {}
        self._runtime.setdefault("started", set())
        self._routed: bool = False

    # Session-shared driver runtime, exposed as a plain attribute.
    @property
    def _started(self) -> Set[str]:
        return self._runtime["started"]

    # -- budget ------------------------------------------------------------
    @property
    def remaining(self) -> int:
        return max(0, self.max_steps - self.used)

    def spend(self, n: int = 1) -> None:
        if self.used + n > self.max_steps:
            self.journal.append("budget_exhausted", max_steps=self.max_steps, used=self.used)
            raise BudgetExhausted(
                f"budget of {self.max_steps} steps exhausted", max_steps=self.max_steps
            )
        self.used += n

    # -- perception / action ----------------------------------------------
    async def observe(self) -> Observation:
        if self.auto_route and not self._routed:
            # Nothing opened yet: don't probe-launch every backend to look at
            # an empty desktop. Return an empty observation; the first
            # navigate opens a surface and the next observe routes to it.
            if not self._started:
                return Observation(surface="none", kind="none")
            await self._route()
        try:
            return await self.drivers[self.primary].observe()
        except SurfaceUnreadable:
            if not self.auto_route:
                raise
            # Current perception driver went blind (surface changed under
            # us, e.g. focus moved browser -> native app). Re-route once.
            await self._route()
            return await self.drivers[self.primary].observe()

    async def _ensure_started(self, name: str) -> None:
        if name in self._started:
            return
        driver = self.drivers[name]
        start = getattr(driver, "start", None)
        if start is not None:
            res = start()
            if hasattr(res, "__await__"):
                await res
        self._started.add(name)

    def _route_order(self) -> List[str]:
        """Perception drivers to probe: already-running ones first, cold
        ones last. Preferring a live backend means we never cold-start a
        browser when a running driver already sees the surface — no
        target-string guessing required."""
        started = [n for n in self.drivers if n in self._started]
        cold = [n for n in self.drivers if n not in self._started]
        return started + cold

    async def _route(self) -> None:
        """Pick the perception driver that actually sees the focused surface.
        Probe running drivers first, cold ones only if none running can see,
        and commit to the first usable one."""
        scores: List[CapabilityScore] = []
        chosen: Optional[str] = None
        for name in self._route_order():
            try:
                await self._ensure_started(name)
                obs = await self.drivers[name].observe()
                cs = CapabilityScore.from_observation(name, obs)
            except Exception:
                cs = CapabilityScore(driver=name, usable=False)
            scores.append(cs)
            if cs.usable:
                chosen = name
                break  # first driver that sees the surface wins
        if chosen:
            self.primary = chosen
        rest = [n for n in self.drivers if n != self.primary]
        self.ladder_order = [self.primary] + rest
        self._routed = True
        self.journal.append(
            "route", primary=self.primary,
            scores={s.driver: round(s.score, 3) for s in scores},
        )

    def reroute(self) -> None:
        """Force re-selection of the perception driver on next observe."""
        self._routed = False

    async def _navigate(self, action: Action) -> ActionResult:
        """Open a URL or app. The driver that *can* open this target decides
        the surface — dom opens URLs, tree launches apps — so we ask each
        driver rather than pattern-matching the target string ourselves."""
        target_name = self._pick_navigator(action.value)
        await self._ensure_started(target_name)
        self.primary = target_name
        self._routed = False  # next observe re-confirms against the real surface
        try:
            await self.drivers[target_name].act(action)
            landed, note = True, ""
        except OrbitError as exc:
            landed, note = False, exc.message
        self.journal.append(
            "action", kind_="navigate", target=action.value, value=action.value,
            landed=landed, strategy=target_name, attempts=1, diff="navigated",
            errors=[note] if note else [],
        )
        return ActionResult(landed=landed, action=action, strategy=target_name, note=note)

    def _pick_navigator(self, target: Optional[str]) -> str:
        """Ask each driver whether it can open this target; first yes wins.
        Drivers own the definition ('is this a URL?' lives in the dom driver,
        not in a hardcoded classifier here)."""
        for name, driver in self.drivers.items():
            can = getattr(driver, "can_navigate", None)
            if can is not None and can(target):
                return name
        # No driver claimed it: fall back to the current primary.
        return self.primary

    async def act(self, action: Action) -> ActionResult:
        # Navigate opens a surface that doesn't exist yet, so it can't be
        # routed by observation. The driver that *can* open the target
        # handles it; the next observe then routes perception normally.
        if action.kind is ActionKind.NAVIGATE and self.auto_route:
            return await self._navigate(action)

        if self.auto_route and not self._routed:
            await self._route()
        await self._ensure_started(self.primary)  # observe_via / first rung
        ordered = [self.drivers[n] for n in self.ladder_order if n in self.drivers]
        try:
            outcome = await run_ladder(
                ordered, action,
                observe_via=self.drivers[self.primary],
                ensure=self._ensure_started,  # start each rung only when reached
            )
        except TargetUnresolvable as exc:
            self.journal.append(
                "action_failed",
                kind_=action.kind.value, target=action.target,
                error=exc.code, message=exc.message, context=exc.context,
            )
            raise
        result = outcome.result
        self.journal.append(
            "action",
            kind_=action.kind.value, target=action.target, value=action.value,
            landed=result.landed, strategy=result.strategy,
            attempts=result.attempts, diff=result.diff.summary(),
            errors=[str(e) for e in outcome.errors],
        )
        # A navigation may switch the focused surface (browser <-> native
        # app); force perception to re-route on the next observe.
        if self.auto_route and action.kind is ActionKind.NAVIGATE:
            self._routed = False
        return result

    async def probe(self, surface: str) -> List[str]:
        """Score every driver against `surface`, cache, and reorder the
        ladder best-first."""
        scores: List[CapabilityScore] = []
        for name, driver in self.drivers.items():
            try:
                obs = await driver.observe()
                scores.append(CapabilityScore.from_observation(name, obs))
            except Exception:
                scores.append(CapabilityScore(driver=name, usable=False))
        scores.sort(key=lambda s: s.score, reverse=True)
        self.probe_cache[surface] = scores
        self.ladder_order = [s.driver for s in scores]
        self.journal.append(
            "probe", surface=surface,
            order=self.ladder_order,
            scores={s.driver: round(s.score, 3) for s in scores},
        )
        return self.ladder_order
