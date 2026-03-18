from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import json


def _truncate(obj: Any, max_len: int = 1200) -> Any:
    """Truncate large tool args/responses to keep the journal compact."""
    try:
        s = json.dumps(obj, ensure_ascii=True)
    except Exception:
        s = str(obj)

    if len(s) <= max_len:
        return obj

    # Return a string so the JSON stays small.
    return s[: max_len - 10] + " ... (truncated)"


@dataclass
class JournalEntry:
    core_key: str
    phase_instruction: str = ""

    llm_start: list[str] = field(default_factory=list)
    llm_end: list[str] = field(default_factory=list)

    # Evidence: recent tool calls + outcomes.
    actions: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    # Internal helpers (not injected to LLM verbatim).
    _last_interactions: list[str] = field(default_factory=list)
    _first_interactions_targeted: bool = False


class Journal:
    """Ephemeral per-attempt OS action journal."""

    def __init__(self, core_key: str = "desktop_attempt_0"):
        self.entry = JournalEntry(core_key=core_key)
        self._pending_calls: dict[str, dict[str, Any]] = {}
        self._element_index: dict[str, str] = {}  # oculos_id -> label

    def reset(self, *, core_key: str, phase_instruction: str = "") -> None:
        self.entry = JournalEntry(
            core_key=core_key, phase_instruction=phase_instruction
        )
        self._pending_calls.clear()
        self._element_index.clear()

    def record_call(
        self,
        *,
        call_id: Optional[str],
        tool_name: str,
        tool_args: Optional[dict[str, Any]] = None,
    ) -> None:
        args = tool_args or {}
        if call_id:
            self._pending_calls[call_id] = {
                "tool_name": tool_name,
                "args": args,
            }

        # Track “first/final elements interacted with” from the CALL params.
        self._maybe_record_interaction_from_call(tool_name, args)

    def record_response(
        self,
        *,
        call_id: Optional[str],
        tool_name: str,
        response: Any,
    ) -> None:
        args = {}
        if call_id and call_id in self._pending_calls:
            args = self._pending_calls[call_id].get("args") or {}

        # Update element label index from discovery tools.
        if tool_name in ("find_ui_elements", "find_ui_elements_hwnd"):
            self._update_element_index_from_discovery(response)

        status = None
        message = None
        if isinstance(response, dict):
            status = response.get("status")
            message = response.get("message") or response.get("error")

        action = {
            "tool": tool_name,
            "args": _truncate(args),
            "response_status": status,
            "response_message": (
                _truncate(message, 500) if message is not None else None
            ),
        }
        self.entry.actions.append(action)

        if status in ("error", "timeout"):
            self.entry.errors.append(action)

        # Cap actions to keep session.state small.
        if len(self.entry.actions) > 30:
            self.entry.actions = self.entry.actions[-30:]
        if len(self.entry.errors) > 10:
            self.entry.errors = self.entry.errors[-10:]

    def finalize_end_interactions(self, *, last_k: int = 4) -> None:
        self.entry.llm_end = self.entry._last_interactions[-last_k:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_key": self.entry.core_key,
            "phase_instruction": self.entry.phase_instruction,
            "llm_start": self.entry.llm_start,
            "llm_end": self.entry.llm_end,
            "actions": self.entry.actions[-12:],  # keep injected prompt compact
            "errors": self.entry.errors,
        }

    def to_prompt_block(self) -> str:
        d = self.to_dict()
        return (
            "OS Action Journal (ephemeral, attempt-scoped):\n"
            f"- core_key: {d['core_key']}\n"
            f"- phase_instruction: {d['phase_instruction']}\n"
            f"- llm_start: {d['llm_start']}\n"
            f"- llm_end: {d['llm_end']}\n"
            "- recent_actions:\n"
            + "\n".join(
                f"  - {i+1}. {a.get('tool')} status={a.get('response_status')} msg={a.get('response_message')}"
                for i, a in enumerate(d["actions"])
            )
            + (
                "\n- errors:\n"
                + "\n".join(
                    f"  - {e.get('tool')} msg={e.get('response_message')}"
                    for e in d["errors"]
                )
                if d["errors"]
                else "\n- errors: []"
            )
        )

    def _maybe_record_interaction_from_call(
        self, tool_name: str, args: dict[str, Any]
    ) -> None:
        # The “elements interacted with” are primarily oculos UI interactions.
        if tool_name == "interact_with_element":
            element_id = args.get("element_id")
            action = args.get("action")
            label = self._element_index.get(element_id) or element_id or "?"
            desc = f"interact_with_element(action={action}, target={label})"
        elif tool_name in ("mouse_click", "mouse_right_click"):
            desc = f"{tool_name}(x={args.get('x')}, y={args.get('y')})"
        elif tool_name == "mouse_drag":
            desc = f"mouse_drag(({args.get('x1')},{args.get('y1')})->({args.get('x2')},{args.get('y2')}))"
        elif tool_name == "mouse_type":
            desc = f"mouse_type(text_len={len(str(args.get('text', '') or ''))})"
        else:
            return

        # Track rolling tail.
        self.entry._last_interactions.append(desc)
        if len(self.entry._last_interactions) > 30:
            self.entry._last_interactions = self.entry._last_interactions[-30:]

        # Track prefix once.
        if not self.entry._first_interactions_targeted:
            self.entry.llm_start.append(desc)
            if len(self.entry.llm_start) >= 4:
                self.entry._first_interactions_targeted = True

    def _update_element_index_from_discovery(self, response: Any) -> None:
        if not isinstance(response, dict):
            return
        elements = response.get("elements") or []
        for el in elements:
            if not isinstance(el, dict):
                continue
            element_id = el.get("oculos_id")
            if not element_id:
                continue
            label = (
                el.get("title")
                or el.get("label")
                or el.get("name")
                or el.get("element_type")
                or str(element_id)
            )
            self._element_index[str(element_id)] = str(label)
