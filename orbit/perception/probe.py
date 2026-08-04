"""Surface probing: rank drivers by how well they can see a surface."""

from __future__ import annotations

from typing import Dict, List

from ..drivers.base import CapabilityScore, Driver
from ..types import SurfaceUnreadable


async def probe_surface(drivers: Dict[str, Driver], surface: str) -> List[str]:
    """Observe `surface` through every driver and return driver names
    ordered best-first by CapabilityScore. Blind drivers score 0 and
    sort last."""
    scores: List[CapabilityScore] = []
    for name, driver in drivers.items():
        try:
            obs = await driver.observe()
            scores.append(CapabilityScore.from_observation(name, obs))
        except SurfaceUnreadable:
            scores.append(CapabilityScore(driver=name, usable=False))
    scores.sort(key=lambda s: s.score, reverse=True)
    return [s.driver for s in scores]
