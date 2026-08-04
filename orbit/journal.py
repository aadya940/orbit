"""Append-only run journal.

Every action, decision, error, and policy gate is recorded here. The
journal is the audit trail attached to each :class:`~orbit.types.RunResult`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List


class Journal:
    """Append-only log of everything that happened during a run.

    Entries are never mutated or removed once appended, so the sequence is
    a faithful replay of the run.

    Attributes
    ----------
    entries : list of dict
        Recorded entries in append order. Each entry carries at least a
        ``ts`` timestamp and a ``kind`` label.

    Examples
    --------
    >>> journal = Journal()
    >>> entry = journal.append("action", target="Submit")
    >>> entry["kind"]
    'action'
    """

    def __init__(self) -> None:
        """Create an empty journal."""
        self.entries: List[Dict[str, Any]] = []

    def append(self, kind: str, **data: Any) -> Dict[str, Any]:
        """Record a new entry and return it.

        Parameters
        ----------
        kind : str
            Category label for the entry, for example ``"action"``,
            ``"route"`` or ``"run_end"``.
        **data : Any
            Arbitrary structured fields merged into the entry.

        Returns
        -------
        dict
            The stored entry, including the generated ``ts`` timestamp and
            the given ``kind``.

        Examples
        --------
        >>> journal = Journal()
        >>> sorted(journal.append("probe", surface="browser:main"))
        ['kind', 'surface', 'ts']
        """
        entry: Dict[str, Any] = {"ts": time.time(), "kind": kind, **data}
        self.entries.append(entry)
        return entry

    def tail(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent entries.

        Parameters
        ----------
        n : int, optional
            Maximum number of entries to return. Default is 10.

        Returns
        -------
        list of dict
            Up to ``n`` entries, oldest first.
        """
        return self.entries[-n:]

    def to_list(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of every entry.

        Returns
        -------
        list of dict
            All entries in append order. The list is a copy, so callers
            cannot extend the journal through it.
        """
        return list(self.entries)
