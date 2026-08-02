"""World: per-session container replacing all globals.

Holds drivers, journal, policy, probe cache and the step budget. Two
Worlds share nothing, which is what makes everything testable."""

from __future__ import annotations

from typing import Dict, List, Optional

from .drivers.base import CapabilityScore, Driver, run_ladder
from .journal import Journal
from .policy import Policy
from .types import (
    Action,
    ActionResult,
    BudgetExhausted,
    Observation,
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
    ) -> None:
        if not drivers:
            raise ValueError("World needs at least one driver")
        self.drivers: Dict[str, Driver] = dict(drivers)
        self.primary: str = primary or next(iter(self.drivers))
        self.policy: Policy = policy or Policy()
        self.journal: Journal = journal or Journal()
        self.probe_cache: Dict[str, List[CapabilityScore]] = {}
        self.ladder_order: List[str] = list(self.drivers)
        self.max_steps: int = max_steps
        self.used: int = 0

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
        return await self.drivers[self.primary].observe()

    async def act(self, action: Action) -> ActionResult:
        ordered = [self.drivers[n] for n in self.ladder_order if n in self.drivers]
        try:
            outcome = await run_ladder(ordered, action, observe_via=self.drivers[self.primary])
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
