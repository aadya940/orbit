"""Side-effect policy with pluggable human-in-the-loop approval.

Side-effecting work is grouped into a small fixed set of categories. Each
category carries a decision: allow it outright, route it through an
approver callback, or refuse it. Refusal and rejected approval both raise
:class:`~orbit.types.PolicyDenied` rather than returning a status, so a
denied action can never be mistaken for a completed one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Literal, Optional

from .types import PolicyDenied

#: Permitted decision values for any policy category.
Decision = Literal["allow", "ask", "deny"]

#: The side-effect categories a :class:`Policy` can gate.
CATEGORIES = ("disk_writes", "deletes", "uploads", "external_sends")


@dataclass
class Policy:
    """What side-effecting actions are permitted.

    Each category holds one of three decisions: ``"allow"`` to permit it
    silently, ``"ask"`` to route it through :attr:`approver`, or
    ``"deny"`` to refuse it.

    Attributes
    ----------
    disk_writes : {'allow', 'ask', 'deny'}, optional
        Decision for writing to disk. Default is ``"ask"``.
    deletes : {'allow', 'ask', 'deny'}, optional
        Decision for destructive removal. Default is ``"deny"``, since a
        delete is the one category that cannot be undone by retrying.
    uploads : {'allow', 'ask', 'deny'}, optional
        Decision for sending local files outward. Default is ``"ask"``.
    external_sends : {'allow', 'ask', 'deny'}, optional
        Decision for messages leaving the machine, for example email or
        chat posts. Default is ``"ask"``.
    approver : callable or None, optional
        Async callback consulted for ``"ask"``-gated actions. It receives
        the action description and returns a bool. Default is None, which
        makes every ``"ask"`` category behave as a denial.

    Examples
    --------
    >>> policy = Policy(disk_writes="allow")
    >>> policy.check("disk_writes")
    'allow'
    """

    disk_writes: Decision = "ask"
    deletes: Decision = "deny"
    uploads: Decision = "ask"
    external_sends: Decision = "ask"
    #: async callback asked to approve "ask"-gated actions; None means deny.
    approver: Optional[Callable[[str], Awaitable[bool]]] = None

    def check(self, category: str) -> Decision:
        """Return the decision for a category.

        Parameters
        ----------
        category : str
            One of the names in :data:`CATEGORIES`.

        Returns
        -------
        {'allow', 'ask'}
            The configured decision. ``"deny"`` is never returned, because
            it raises instead.

        Raises
        ------
        PolicyDenied
            If ``category`` is not a known category, or if it is
            configured as ``"deny"``.

        Examples
        --------
        >>> Policy().check("deletes")
        Traceback (most recent call last):
            ...
        orbit.types.PolicyDenied: policy denies deletes
        """
        if category not in CATEGORIES:
            raise PolicyDenied(f"unknown policy category: {category}", category=category)
        decision: Decision = getattr(self, category)
        if decision == "deny":
            raise PolicyDenied(f"policy denies {category}", category=category)
        return decision

    async def gate(self, category: str, description: str) -> None:
        """Gate an action, returning only if it is permitted.

        Passes straight through on ``"allow"``. On ``"ask"``, consults
        :attr:`approver` with ``description`` and proceeds only if it
        returns a truthy value.

        Parameters
        ----------
        category : str
            One of the names in :data:`CATEGORIES`.
        description : str
            Human-readable description of the specific action, shown to
            the approver so the decision is made with full context rather
            than on the category name alone.

        Returns
        -------
        None
            Returns normally when the action is permitted.

        Raises
        ------
        PolicyDenied
            If the category is unknown or denied, if the category needs
            approval but no approver is configured, or if the approver
            rejects the action.

        Notes
        -----
        A missing approver is treated as a denial rather than an implicit
        allow, so forgetting to wire one up fails closed.
        """
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
