"""Side-effect policy with pluggable human-in-the-loop approval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Literal, Optional

from .types import PolicyDenied

Decision = Literal["allow", "ask", "deny"]

CATEGORIES = ("disk_writes", "deletes", "uploads", "external_sends")


@dataclass
class Policy:
    """What side-effecting actions are permitted.

    Each category is "allow", "ask" (route through `approver`), or "deny".
    """

    disk_writes: Decision = "ask"
    deletes: Decision = "deny"
    uploads: Decision = "ask"
    external_sends: Decision = "ask"
    #: async callback asked to approve "ask"-gated actions; None means deny.
    approver: Optional[Callable[[str], Awaitable[bool]]] = None

    def check(self, category: str) -> Decision:
        """Return the decision for a category; raise PolicyDenied on "deny"."""
        if category not in CATEGORIES:
            raise PolicyDenied(f"unknown policy category: {category}", category=category)
        decision: Decision = getattr(self, category)
        if decision == "deny":
            raise PolicyDenied(f"policy denies {category}", category=category)
        return decision

    async def gate(self, category: str, description: str) -> None:
        """Gate an action: pass on "allow", consult approver on "ask"."""
        decision = self.check(category)
        if decision == "allow":
            return
        if self.approver is None:
            raise PolicyDenied(
                f"{category} requires approval but no approver is configured",
                category=category, description=description,
            )
        if not await self.approver(description):
            raise PolicyDenied(
                f"approver rejected {category}: {description}",
                category=category, description=description,
            )
