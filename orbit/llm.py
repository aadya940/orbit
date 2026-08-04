"""Thin LLM layer: one protocol, one litellm-backed implementation.

The loop depends on the :class:`LLM` protocol only, so tests can inject a
scripted client and never import a provider SDK.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """A single function call requested by the model.

    Attributes
    ----------
    name : str
        Name of the tool the model chose to invoke.
    arguments : dict
        Decoded keyword arguments for the call. Always a dict, even when
        the provider returned the arguments as a JSON string.
    """

    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMReply:
    """One model turn, normalised across providers.

    Attributes
    ----------
    text : str or None, optional
        Assistant prose, if any. Default is None.
    tool_call : ToolCall or None, optional
        The single tool call the model requested, or None for a text-only
        reply. Default is None.

    Notes
    -----
    At most one tool call is carried, because the loop enforces one action
    per turn and simply ignores any extras a provider returns.
    """

    text: Optional[str] = None
    tool_call: Optional[ToolCall] = None


@runtime_checkable
class LLM(Protocol):
    """Structural protocol for anything the loop can think with."""

    async def complete(self, messages: List[dict], tools: List[dict]) -> "LLMReply":
        """Produce the next model turn.

        Parameters
        ----------
        messages : list of dict
            Chat history in OpenAI message format.
        tools : list of dict
            Tool definitions in OpenAI function-calling format.

        Returns
        -------
        LLMReply
            The model's text, its requested tool call, or both.
        """
        ...


class LiteLLMClient:
    """LLM implemented over ``litellm.acompletion``.

    litellm is imported lazily inside :meth:`complete`, so installing
    Orbit and running its tests never requires the provider stack.

    Attributes
    ----------
    model : str
        litellm model identifier, for example ``"gpt-4o"``.
    timeout : float
        Per-request timeout in seconds.
    retries : int
        Number of retries after the first attempt.
    kwargs : dict
        Extra keyword arguments forwarded to every completion call.

    Examples
    --------
    >>> client = LiteLLMClient("gpt-4o", timeout=30.0)
    >>> client.model
    'gpt-4o'
    """

    def __init__(self, model: str, *, timeout: float = 120.0, retries: int = 2, **kwargs: Any) -> None:
        """Configure the client.

        Parameters
        ----------
        model : str
            litellm model identifier.
        timeout : float, optional
            Per-request timeout in seconds. Default is 120.0.
        retries : int, optional
            Retries after the first attempt, applied only to transient
            provider failures. Default is 2.
        **kwargs : Any
            Extra keyword arguments passed through to
            ``litellm.acompletion`` on every call, for example
            ``temperature`` or ``api_base``.
        """
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.kwargs = kwargs

    async def complete(self, messages: List[dict], tools: List[dict]) -> LLMReply:
        """Call the provider and normalise the reply.

        Parameters
        ----------
        messages : list of dict
            Chat history in OpenAI message format.
        tools : list of dict
            Tool definitions in OpenAI function-calling format. An empty
            list is sent as None so providers do not see an empty toolset.

        Returns
        -------
        LLMReply
            The assistant text and at most one decoded tool call.

        Raises
        ------
        Exception
            The last transient provider error, re-raised once the retry
            budget is spent. Non-transient errors propagate immediately.

        Notes
        -----
        Only transient failures (timeout, rate limit, connection loss,
        server errors) are retried, with a linear backoff. Everything else,
        such as a malformed request, would fail identically on retry and so
        is surfaced at once.

        Tool-call arguments arrive as a JSON string from most providers. If
        that string fails to parse, it is preserved under a ``_raw`` key
        instead of being dropped, so the failure stays visible downstream.
        """
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
