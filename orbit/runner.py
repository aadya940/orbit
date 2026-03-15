"""Minimal Agent interface: Agent(llm=..., task=...)."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.artifacts import InMemoryArtifactService

from .agents import parent_agent
from .daemon import OculOSManager
from ._tools.hitl import APPROVAL_TOOLS
from ._ui import default_human_in_the_loop


def _get_long_running_function_call(event: Any) -> Optional[Any]:
    """Get the function call from the event if it is a long-running tool call."""
    if (
        not getattr(event, "long_running_tool_ids", None)
        or not getattr(event, "content", None)
        or not event.content.parts
    ):
        return None
    for part in event.content.parts:
        if (
            getattr(part, "function_call", None)
            and event.long_running_tool_ids
            and part.function_call.id in event.long_running_tool_ids
        ):
            return part.function_call
    return None


def _get_function_response(event: Any, function_call_id: str) -> Optional[Any]:
    """Get the function response for the function call with the given id."""
    if not getattr(event, "content", None) or not event.content.parts:
        return None
    for part in event.content.parts:
        if (
            getattr(part, "function_response", None)
            and getattr(part.function_response, "id", None) == function_call_id
        ):
            return part.function_response
    return None


class _LatencyTracker:
    def __init__(self):
        self.run_start = None
        self.step_start = None
        self.tool_call_times = []
        self.tool_latencies = []
        self.llm_step_latencies = []
        self.final_response_at = None

    def start_run(self):
        self.run_start = time.perf_counter()
        self.step_start = self.run_start

    def on_function_call(self, name: str, args: dict) -> float:
        now = time.perf_counter()
        step_sec = now - self.step_start
        self.llm_step_latencies.append(step_sec)
        self.tool_call_times.append((name, now))
        self.step_start = now
        return step_sec

    def on_function_response(self, name: str) -> float:
        now = time.perf_counter()
        latency = 0.0
        if self.tool_call_times:
            call_name, start = self.tool_call_times.pop(0)
            latency = now - start
            self.tool_latencies.append((call_name, latency))
        self.step_start = now
        return latency

    def on_final_response(self):
        self.final_response_at = time.perf_counter()

    def summary(self):
        total = (self.final_response_at or time.perf_counter()) - (self.run_start or 0)
        tool_total = sum(t for _, t in self.tool_latencies)
        llm_total = sum(self.llm_step_latencies) if self.llm_step_latencies else 0.0
        return {
            "total_sec": round(total, 3),
            "tool_calls": len(self.tool_latencies),
            "tool_time_sec": round(tool_total, 3),
            "llm_steps": len(self.llm_step_latencies),
            "llm_time_sec": round(llm_total, 3),
            "per_tool_sec": [(n, round(t, 3)) for n, t in self.tool_latencies],
        }

    def print_report(self):
        s = self.summary()
        print("AGENT LATENCY REPORT")
        print("--------------------------------")
        print(f"  Total run:           {s['total_sec']:.3f}s")
        print(
            f"  LLM steps:           {s['llm_steps']} (total {s['llm_time_sec']:.3f}s)"
        )
        print(
            f"  Tool calls:          {s['tool_calls']} (total {s['tool_time_sec']:.3f}s)"
        )
        for name, sec in s.get("per_tool_sec", []):
            print(f"    {name}: {sec:.3f}s")
        print("--------------------------------")


HumanInTheLoopHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class Agent:
    """Orbit agent. Pass llm (model name) and task, then await `agent.run`."""

    def __init__(
        self,
        task: str,
        llm: str = "gemini-3-pro-preview",
        measure_latency: bool = True,
        verbose: bool = False,
        human_in_the_loop: Optional[HumanInTheLoopHandler] = None,
    ):
        self.task = task
        self.llm = llm
        self.measure_latency = measure_latency
        self.verbose = verbose
        self._human_in_the_loop = human_in_the_loop

    async def run(self):
        daemon = OculOSManager(verbose=self.verbose)
        await daemon.start()
        try:
            return await self._run()
        finally:
            daemon.stop()

    async def _run(self):
        prompt = self.task
        if self.verbose:
            print(f"\n[User]: {prompt}\n--------------------------------")

        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="desktop_app", user_id="local_admin", session_id="session_001"
        )
        runner = Runner(
            agent=parent_agent,
            app_name="desktop_app",
            session_service=session_service,
            artifact_service=InMemoryArtifactService(),
        )
        user_id = "local_admin"
        content = types.Content(role="user", parts=[types.Part(text=prompt)])
        events = runner.run_async(
            session_id=session.id, user_id=user_id, new_message=content
        )

        latency = _LatencyTracker() if self.measure_latency else None
        if latency:
            latency.start_run()

        handler = self._human_in_the_loop or default_human_in_the_loop
        _last = time.time()

        while True:
            long_running_call = None
            long_running_response = None

            async for event in events:
                if not long_running_call:
                    long_running_call = _get_long_running_function_call(event)
                    if long_running_call and long_running_response is None:
                        long_running_response = _get_function_response(
                            event, long_running_call.id
                        )

                if event.is_final_response():
                    if latency:
                        latency.on_final_response()
                    if getattr(event, "content", None) and event.content.parts:
                        print(f"\n{event.content.parts[0].text}")
                elif getattr(event, "content", None) and event.content.parts:
                    for part in event.content.parts:
                        if getattr(part, "function_call", None):
                            now = time.time()
                            name = part.function_call.name
                            args = (
                                dict(part.function_call.args)
                                if part.function_call.args
                                else {}
                            )
                            if (
                                long_running_call is None
                                and long_running_response is None
                                and getattr(event, "long_running_tool_ids", None)
                            ):
                                long_running_response = _get_function_response(
                                    event, part.function_call.id
                                )
                            if latency:
                                step_sec = latency.on_function_call(name, args)
                                if self.verbose:
                                    print(
                                        f"[{step_sec:.3f}s LLM→tool] [Action]: {name}({args})"
                                    )
                            else:
                                if self.verbose:
                                    print(
                                        f"[{round(now - _last, 2)}s] [Action]: {name}({args})"
                                    )
                            _last = now
                        elif getattr(part, "function_response", None):
                            if (
                                long_running_call
                                and getattr(part.function_response, "id", None)
                                == long_running_call.id
                            ):
                                long_running_response = part.function_response
                            name = getattr(part.function_response, "name", "?")
                            if latency:
                                tool_sec = latency.on_function_response(name)
                                if self.verbose:
                                    print(
                                        f"[tool {tool_sec:.3f}s] [Result]: {part.function_response.response}"
                                    )
                            else:
                                if self.verbose:
                                    print(
                                        f"[Result]: {part.function_response.response}"
                                    )
                            _last = time.time()

            if long_running_call is None:
                break

            # Paused on long-running tool: ask human and resume
            name = long_running_call.name
            args = dict(long_running_call.args) if long_running_call.args else {}
            kind = "approval" if name in APPROVAL_TOOLS else "help"
            context = {"tool": name, **args}

            result = await handler(kind, context)

            if (
                kind == "approval"
                and result.get("status") == "approved"
                and name in APPROVAL_TOOLS
            ):
                try:
                    impl = APPROVAL_TOOLS[name]
                    if asyncio.iscoroutinefunction(impl):
                        response_body = await impl(**args)
                    else:
                        response_body = impl(**args)
                except Exception as e:
                    response_body = {"status": "error", "message": str(e)}
            else:
                response_body = result

            if long_running_response is not None and hasattr(
                long_running_response, "model_copy"
            ):
                updated_response = long_running_response.model_copy(deep=True)
                updated_response.response = response_body
            else:
                updated_response = types.FunctionResponse(
                    id=long_running_call.id, name=name, response=response_body
                )

            resume_content = types.Content(
                role="user",
                parts=[types.Part(function_response=updated_response)],
            )
            events = runner.run_async(
                session_id=session.id, user_id=user_id, new_message=resume_content
            )

        if latency and self.verbose:
            latency.print_report()
        return latency.summary() if latency else None
