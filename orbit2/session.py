"""Public API: Session with verb methods (do / read / check / navigate / fill)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel, create_model

from . import loop
from .llm import LLM, LiteLLMClient
from .policy import Policy
from .types import RunResult, RunStatus
from .world import World

DEFAULT_MODEL = "gpt-4o"


def _default_drivers() -> Dict[str, Any]:
    try:
        from .drivers import default_drivers
        return default_drivers()
    except ImportError as exc:
        raise ImportError(
            "Default Orbit drivers are not available in this build. "
            "Pass drivers explicitly: Session(drivers={'name': driver}). "
            f"(underlying error: {exc})"
        ) from exc


async def _call_if_present(obj: Any, *names: str) -> None:
    for name in names:
        fn = getattr(obj, name, None)
        if fn is not None:
            res = fn()
            if hasattr(res, "__await__"):
                await res
            return


class Session:
    def __init__(
        self,
        llm: Union[str, LLM] = DEFAULT_MODEL,
        policy: Optional[Policy] = None,
        max_steps: int = 40,
        drivers: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.llm: LLM = LiteLLMClient(llm) if isinstance(llm, str) else llm
        self.policy = policy or Policy()
        self.max_steps = max_steps
        self._drivers = drivers
        # Shared across verbs: which backends are started + last surface
        # hint. Drivers start lazily (on first use) so a native-only task
        # never launches a browser, and a web-only task never starts the
        # accessibility daemon.
        self._runtime: Dict[str, Any] = {}

    async def __aenter__(self) -> "Session":
        if self._drivers is None:
            self._drivers = _default_drivers()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        # Only stop backends that were actually started.
        started = self._runtime.get("started", set())
        for name, driver in (self._drivers or {}).items():
            if name in started:
                await _call_if_present(driver, "stop", "close")

    # -- internals ---------------------------------------------------------
    def _world(self, max_steps: Optional[int]) -> World:
        if self._drivers is None:
            self._drivers = _default_drivers()
        return World(
            drivers=self._drivers,
            policy=self.policy,
            max_steps=max_steps if max_steps is not None else self.max_steps,
            runtime=self._runtime,
        )

    async def _run(
        self,
        task: str,
        *,
        llm: Optional[LLM] = None,
        max_steps: Optional[int] = None,
        guidance: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
        timeout: Optional[float] = None,
    ) -> RunResult:
        return await loop.run(
            task=task,
            world=self._world(max_steps),
            llm=llm or self.llm,
            schema=schema,
            guidance=guidance,
            timeout=timeout,
        )

    # -- verbs -------------------------------------------------------------
    async def do(self, task: str, *, llm: Optional[LLM] = None,
                 max_steps: Optional[int] = None, guidance: Optional[str] = None,
                 timeout: Optional[float] = None) -> RunResult:
        return await self._run(f"ACTION: {task}", llm=llm, max_steps=max_steps,
                               guidance=guidance, timeout=timeout)

    async def read(self, task: str, *, schema: Optional[Type[BaseModel]] = None,
                   llm: Optional[LLM] = None, max_steps: Optional[int] = None,
                   guidance: Optional[str] = None, timeout: Optional[float] = None) -> RunResult:
        return await self._run(
            f"READ (observe only, do not change anything): {task}",
            llm=llm, max_steps=max_steps, guidance=guidance,
            schema=schema, timeout=timeout,
        )

    async def check(self, condition: str, *, llm: Optional[LLM] = None,
                    max_steps: Optional[int] = None, guidance: Optional[str] = None,
                    timeout: Optional[float] = None) -> bool:
        schema = create_model("CheckResult", result=(bool, ...))
        run = await self._run(
            f"CHECK (observe only): is the following true? {condition}",
            llm=llm, max_steps=max_steps, guidance=guidance,
            schema=schema, timeout=timeout,
        )
        if run.status is not RunStatus.SUCCESS or run.output is None:
            return False
        return bool(run.output.result)

    async def navigate(self, target: str, *, llm: Optional[LLM] = None,
                       max_steps: Optional[int] = None, guidance: Optional[str] = None,
                       timeout: Optional[float] = None) -> RunResult:
        return await self._run(
            f"NAVIGATE: open {target}, then finish. No further interaction.",
            llm=llm, max_steps=max_steps, guidance=guidance, timeout=timeout,
        )

    async def fill(self, form_name: str, data: Dict[str, Any], *, llm: Optional[LLM] = None,
                   max_steps: Optional[int] = None, guidance: Optional[str] = None,
                   timeout: Optional[float] = None) -> RunResult:
        return await self._run(
            f"FILL the form {form_name!r} with these values, then finish:\n"
            f"{json.dumps(data, default=str, indent=2)}",
            llm=llm, max_steps=max_steps, guidance=guidance, timeout=timeout,
        )
