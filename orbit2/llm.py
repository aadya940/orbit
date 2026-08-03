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

    def __init__(self, model: str, *, timeout: float = 120.0, retries: int = 2, **kwargs: Any) -> None:
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.kwargs = kwargs

    async def complete(self, messages: List[dict], tools: List[dict]) -> LLMReply:
        import asyncio

        import litellm  # lazy: tests never need it

        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    tools=tools or None,
                    timeout=self.timeout,
                    **self.kwargs,
                )
                break
            except (litellm.Timeout, litellm.RateLimitError, litellm.APIConnectionError,
                    litellm.InternalServerError, litellm.ServiceUnavailableError) as exc:
                last_exc = exc
                if attempt < self.retries:
                    await asyncio.sleep(2 * (attempt + 1))
        else:
            raise last_exc  # type: ignore[misc]
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
