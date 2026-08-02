"""Thin LLM layer: one protocol, one litellm-backed implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMReply:
    text: Optional[str] = None
    tool_call: Optional[ToolCall] = None


@runtime_checkable
class LLM(Protocol):
    async def complete(self, messages: List[dict], tools: List[dict]) -> "LLMReply":
        ...


class LiteLLMClient:
    """LLM implemented over `litellm.acompletion` (imported lazily)."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs

    async def complete(self, messages: List[dict], tools: List[dict]) -> LLMReply:
        import litellm  # lazy: tests never need it

        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            tools=tools or None,
            **self.kwargs,
        )
        message = response.choices[0].message
        text = getattr(message, "content", None)
        tool_call: Optional[ToolCall] = None
        raw_calls = getattr(message, "tool_calls", None) or []
        if raw_calls:
            first = raw_calls[0]
            args = first.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_call = ToolCall(name=first.function.name, arguments=args or {})
        return LLMReply(text=text, tool_call=tool_call)
