"""Append-only run journal. Every action, decision, error, and policy
gate is recorded here — the audit trail attached to each RunResult."""

from __future__ import annotations

import time
from typing import Any, Dict, List


class Journal:
    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def append(self, kind: str, **data: Any) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"ts": time.time(), "kind": kind, **data}
        self.entries.append(entry)
        return entry

    def tail(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.entries[-n:]

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.entries)
