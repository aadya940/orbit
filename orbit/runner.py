"""Minimal Agent interface: Agent(llm=..., task=...)."""

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.genai.errors import ClientError
from google.adk.artifacts import InMemoryArtifactService

from .agents import (
    build_agents,
    DESKTOP_EXECUTOR_AGENT_NAME,
)
from .daemon import OculOSManager
from ._ui.console import OrbitConsole
from .journal import Journal

log = logging.getLogger("orbit")


@dataclass
class RunResult:
    """Structured return type for Agent.run()."""

    status: Literal["success", "failed", "needs_human", "error"]
    summary: str = ""
    output: Any = None
    errors: list[str] = field(default_factory=list)
    latency: dict = field(default_factory=dict)
    journal: dict = field(default_factory=dict)


def _console_safe(obj: Any) -> str:
    """Return an ASCII-only string for console logging."""
    s = str(obj)
    return s.encode("ascii", errors="backslashreplace").decode("ascii")


def _setup_logging(*, verbose: bool = False) -> None:
    """Configure the 'orbit' logger. Called once per Agent init."""
    logger = logging.getLogger("orbit")
    if logger.handlers:
        return
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False


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


HumanInTheLoopHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class Agent:
    """Orbit agent. Pass llm (model name) and task, then await `agent.run`."""

    def __init__(
        self,
        task: str,
        llm: str = "gemini-3-pro-preview",
        desktop_llm: Optional[str] = None,
        planner_llm: Optional[str] = None,
        measure_latency: bool = True,
        verbose: bool = False,
        max_steps: int = 30,
        planner: bool = True,
        session: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        extra_info: Optional[str] = None,
        extra_tools: Optional[list] = None,
        human_in_the_loop: Optional[HumanInTheLoopHandler] = None,
        timeout: Optional[int] = None,
    ):
        self.task = task
        self.extra_info = extra_info.strip() if isinstance(extra_info, str) else None
        self.llm = llm
        self.desktop_llm = desktop_llm
        self.planner_llm = planner_llm
        self.measure_latency = measure_latency
        self.verbose = verbose
        self.max_steps = max_steps
        self.planner = bool(planner)
        self._session = session
        self._output_schema = output_schema
        self._extra_tools = extra_tools or []
        self._human_in_the_loop = human_in_the_loop
        self._timeout = timeout  # seconds; None = no timeout
        self._owns_session = False

        _setup_logging(verbose=verbose)

    # ── Lifecycle ─────────────────────────────────────────────────

    async def run(self) -> RunResult:
        if self._session and self._session.started:
            try:
                return await self._run()
            except Exception as e:
                log.error("Agent run failed: %s", e, exc_info=True)
                return RunResult(status="error", summary=str(e), errors=[str(e)])
        else:
            # Ephemeral session — backward compatible path.
            from .session import Session

            async with Session() as s:
                self._session = s
                try:
                    return await self._run()
                except Exception as e:
                    log.error("Agent run failed: %s", e, exc_info=True)
                    return RunResult(status="error", summary=str(e), errors=[str(e)])

    async def __aenter__(self):
        if not self._session:
            from .session import Session

            self._session = Session()
            await self._session.__aenter__()
            self._owns_session = True
        return self

    async def __aexit__(self, *exc):
        if self._owns_session and self._session:
            await self._session.__aexit__(*exc)
            self._session = None
            self._owns_session = False

    # ── Core orchestration ────────────────────────────────────────

    async def _run(self) -> RunResult:
        prompt = self._compose_prompt(self.task, self.extra_info)
        self._ui = OrbitConsole(verbose=self.verbose)
        self._ui.task_start(prompt)

        # Run state.
        self._final_text = ""
        self._final_output: Any = None
        self._errors: list[str] = []
        self._saw_request_human = False

        # Session reuse strategy:
        # - planner=True  → reuse ADK session so the planner retains context across verbs
        # - planner=False → fresh session per verb (no planner to benefit from history,
        #   and accumulated tool responses can cause Gemini INVALID_ARGUMENT errors)
        reuse_session = (
            self.planner and self._session and hasattr(self._session, "session_service")
        )

        if reuse_session:
            session_service = self._session.session_service
            if self._session.adk_session is not None:
                cached = self._session.adk_session
                session = await session_service.get_session(
                    app_name="desktop_app",
                    user_id="local_admin",
                    session_id=cached.id,
                )
                if session is None:
                    session = cached
                self._session.adk_session = session
            else:
                session = await session_service.create_session(
                    app_name="desktop_app",
                    user_id="local_admin",
                    session_id="session_001",
                    state={},
                )
                self._session.adk_session = session
        elif self._session and hasattr(self._session, "session_service"):
            # planner=False with a session: fresh ADK session each time,
            # but still use the session's service for consistency.
            session_service = self._session.session_service
            session_id = f"verb_{int(time.time() * 1000)}"
            session = await session_service.create_session(
                app_name="desktop_app",
                user_id="local_admin",
                session_id=session_id,
                state={},
            )
        else:
            session_service = InMemorySessionService()
            session = await session_service.create_session(
                app_name="desktop_app",
                user_id="local_admin",
                session_id="session_001",
                state={},
            )

        self._journal = Journal(core_key="desktop_attempt_0")
        self._desktop_attempt_idx = 0
        self._journal_active = False
        self._session_obj = session

        desktop_model = self.desktop_llm or self.llm
        planner_model = self.planner_llm or self.llm
        build_kwargs: dict[str, Any] = {}
        if desktop_model is not None:
            build_kwargs["desktop_model"] = desktop_model
        if planner_model is not None:
            build_kwargs["planner_model"] = planner_model
        build_kwargs["planner"] = self.planner
        if self._extra_tools:
            build_kwargs["extra_tools"] = self._extra_tools
        if self._output_schema is not None:
            # ADK-native structured output path:
            # - ADK validates the model reply against output_schema
            # - ADK stores the validated JSON string into session.state[output_key]
            output_key = f"orbit_output_{int(time.time() * 1000)}"
            self._output_schema_output_key = output_key
            build_kwargs["output_schema"] = self._output_schema
            build_kwargs["output_key"] = output_key
        self._budget_counter = {"call_count": 0}
        build_kwargs["max_calls"] = self.max_steps
        build_kwargs["budget_counter"] = self._budget_counter

        root_agent, _desktop_agent = build_agents(**build_kwargs)

        # NOTE: EventsCompactionConfig disabled — ADK compaction can corrupt
        # function_call/function_response pairs, causing Gemini to reject
        # subsequent requests with 400 INVALID_ARGUMENT. See ADK issue #4740.
        app = App(
            name="desktop_app",
            root_agent=root_agent,
        )
        # Reuse artifact service from orbit Session so screenshots persist
        # across verbs; fall back to a fresh one for standalone Agent usage.
        if self._session and hasattr(self._session, "artifact_service"):
            artifact_service = self._session.artifact_service
        else:
            artifact_service = InMemoryArtifactService()

        runner = Runner(
            app=app,
            session_service=session_service,
            artifact_service=artifact_service,
        )
        content = types.Content(role="user", parts=[types.Part(text=prompt)])
        events = runner.run_async(
            session_id=session.id,
            user_id="local_admin",
            new_message=content,
        )

        self._latency = _LatencyTracker() if self.measure_latency else None
        if self._latency:
            self._latency.start_run()
        self._last_time = time.time()

        async def _consume_events():
            async for event in events:
                self._dispatch_event(event)

        try:
            if self._timeout:
                await asyncio.wait_for(_consume_events(), timeout=self._timeout)
            else:
                await _consume_events()
        except asyncio.TimeoutError:
            log.error("Global timeout (%ds) reached. Stopping agent.", self._timeout)
            self._errors.append(
                f"Global timeout ({self._timeout}s) reached. Agent stopped."
            )
        except ClientError as e:
            log.error(
                "Gemini API error at call %d: %s",
                (
                    self._budget_counter.get("call_count", 0)
                    if self._budget_counter
                    else "?"
                ),
                e,
            )
            self._errors.append(f"API error: {e}")

        self._ui.step_done()

        # Determine status.
        if self._saw_request_human:
            status = "needs_human"
        elif self._errors:
            status = "failed"
        else:
            status = "success"

        if self._output_schema is not None and status == "success":
            # ADK-native structured output path:
            # - output_schema validates the model's reply
            # - output_key stores the validated JSON string in session.state
            output_key = self._output_schema_output_key
            try:
                updated_session = await session_service.get_session(
                    app_name="desktop_app",
                    user_id="local_admin",
                    session_id=session.id,
                )
                state = (
                    updated_session.state if updated_session else session.state
                ) or {}
                json_text = state.get(output_key)
                if json_text is None:
                    raise KeyError(
                        f"Missing ADK output_key={output_key!r} in session.state"
                    )
                self._final_output = self._validate_schema_json(json_text)
            except Exception as e:
                # Fallback: parse from the final response text if output_key is missing.
                # Keep strict validation semantics: if validation fails, mark run failed.
                try:
                    payload = self._extract_json_payload(self._final_text)
                    if payload is None:
                        raise ValueError(
                            "Final response text is not valid JSON payload"
                        )
                    self._final_output = self._validate_schema_json(payload)
                except Exception as e2:
                    status = "failed"
                    self._errors.append(
                        f"structured_output_decode_failed: state={e}; text={e2}"
                    )
                    self._final_output = None
        else:
            self._final_output = self._final_text

        latency_summary = self._latency.summary() if self._latency else {}
        llm_calls_used = int(self._budget_counter.get("call_count", 0))
        latency_summary["llm_calls"] = llm_calls_used
        latency_summary["max_llm_calls"] = self.max_steps
        if self._latency:
            self._ui.latency(latency_summary)

        return RunResult(
            status=status,
            summary=self._final_text,
            output=self._final_output,
            errors=self._errors,
            latency=latency_summary,
            journal=self._journal.to_dict(),
        )

    @staticmethod
    def _compose_prompt(task: str, extra_info: Optional[str]) -> str:
        """Build the ADK user prompt from task plus optional advisory hints."""
        base_task = (task or "").strip()
        if not extra_info:
            return base_task
        return (
            f"PRIMARY_TASK:\n{base_task}\n\n"
            f"EXTRA_INFO (advisory context):\n{extra_info}"
        )

    # ── Event dispatch ────────────────────────────────────────────

    def _dispatch_event(self, event) -> None:
        if event.is_final_response():
            self._on_final_response(event)
        elif getattr(event, "content", None) and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "function_call", None):
                    self._on_function_call(event, part)
                elif getattr(part, "function_response", None):
                    self._on_function_response(event, part)

    def _on_final_response(self, event) -> None:
        is_desktop = event.author == DESKTOP_EXECUTOR_AGENT_NAME

        # Finalize journal when desktop executor finishes.
        if (
            is_desktop
            and self._journal_active
            and getattr(event, "content", None)
            and getattr(event.content, "parts", None)
            and event.content.parts
            and getattr(event.content.parts[0], "text", None) is not None
        ):
            self._journal.finalize_end_interactions()
            self._session_obj.state["journal"] = self._journal.to_dict()
            self._journal_active = False

        if self._latency:
            self._latency.on_final_response()

        text = None
        if getattr(event, "content", None) and event.content.parts:
            text = getattr(event.content.parts[0], "text", None)

        if is_desktop and self.planner:
            self._ui.step_done()
        else:
            self._final_text = _console_safe(text) if text else ""
            self._ui.agent_done(self._final_text)

    def _validate_schema_json(self, json_text: str) -> Any:
        """Strict schema validation from a JSON string with light dict coercion."""
        schema = self._output_schema
        if not schema:
            return json_text

        model_validate_json = getattr(schema, "model_validate_json", None)
        if callable(model_validate_json):
            try:
                return schema.model_validate_json(json_text)
            except Exception:
                # Fall through to dict-level coercion path.
                pass

        data = json.loads(json_text)
        validate = getattr(schema, "model_validate", None)
        if not callable(validate):
            return data

        try:
            return schema.model_validate(data)
        except Exception:
            if isinstance(data, dict):
                coerced = self._coerce_dict_for_schema(schema, data)
                if coerced is not None:
                    return schema.model_validate(coerced)
            raise

    @staticmethod
    def _coerce_dict_for_schema(schema: Any, data: dict) -> Optional[dict]:
        """
        Heuristics to coerce common LLM JSON shapes into the schema shape.

        Common failure patterns:
        - Model returns {"items": [...]} but schema expects {"products": [...]}
        - Model returns {"products": {"items": [...]}} but schema expects {"products": [...]}
        """
        model_fields = getattr(schema, "model_fields", None)
        if not isinstance(model_fields, dict):
            return None

        out = dict(data)

        # 1) Unwrap nested {"field": {"items": [...]}} -> {"field": [...]}
        for fname in list(model_fields.keys()):
            v = out.get(fname)
            if isinstance(v, dict) and isinstance(v.get("items"), list):
                out[fname] = v["items"]

        # 2) If schema has exactly one field and payload is {"items": [...]}, map it.
        if (
            "items" in out
            and isinstance(out.get("items"), list)
            and len(model_fields) == 1
        ):
            only_field = next(iter(model_fields.keys()))
            if only_field not in out:
                out[only_field] = out["items"]

        # 3) Common alias mapping (items/results -> products)
        if "products" in model_fields and "products" not in out:
            for k in ("items", "results", "data"):
                if k in out and isinstance(out.get(k), list):
                    out["products"] = out[k]
                    break

        return out

    @staticmethod
    def _extract_json_payload(text: str) -> Optional[str]:
        if not text:
            return None
        s = text.strip()
        if s.startswith("```"):
            m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL)
            if m:
                s = m.group(1).strip()
        if (s.startswith("{") and s.endswith("}")) or (
            s.startswith("[") and s.endswith("]")
        ):
            return s
        return None

    def _on_function_call(self, event, part) -> None:
        name = part.function_call.name
        args = dict(part.function_call.args) if part.function_call.args else {}
        is_desktop = event.author == DESKTOP_EXECUTOR_AGENT_NAME

        # Start journal for new desktop phase.
        if is_desktop and not self._journal_active:
            self._desktop_attempt_idx += 1
            phase_instruction = self._session_obj.state.get(
                "journal_phase_instruction", ""
            )
            self._journal.reset(
                core_key=f"desktop_attempt_{self._desktop_attempt_idx}",
                phase_instruction=str(phase_instruction or ""),
            )
            self._session_obj.state["journal"] = self._journal.to_dict()
            self._journal_active = True

        if name == "request_human":
            self._saw_request_human = True

        # UI updates.
        if name == DESKTOP_EXECUTOR_AGENT_NAME:
            self._ui.step_start(args.get("request", str(args)))
        elif is_desktop:
            self._ui.step_tool(name)

        # Journal.
        if is_desktop:
            self._journal.record_call(
                call_id=getattr(part.function_call, "id", None),
                tool_name=name,
                tool_args=args,
            )

        # Latency / logging.
        if self._latency:
            step_sec = self._latency.on_function_call(name, args)
            log.debug("[%.3fs] %s(%s)", step_sec, name, _console_safe(args))
        else:
            now = time.time()
            log.debug(
                "[%.2fs] %s(%s)",
                round(now - self._last_time, 2),
                name,
                _console_safe(args),
            )
            self._last_time = now

    def _on_function_response(self, event, part) -> None:
        name = getattr(part.function_response, "name", "?")
        is_desktop = event.author == DESKTOP_EXECUTOR_AGENT_NAME

        # Journal.
        if is_desktop:
            self._journal.record_response(
                call_id=getattr(part.function_response, "id", None),
                tool_name=name,
                response=getattr(part.function_response, "response", None),
            )

        # Collect errors.
        resp = getattr(part.function_response, "response", None)
        if isinstance(resp, dict) and resp.get("status") == "error":
            self._errors.append(f"{name}: {resp.get('message', 'unknown error')}")

        # Latency / logging.
        if self._latency:
            tool_sec = self._latency.on_function_response(name)
            log.debug(
                "[tool %.3fs] %s -> %s",
                tool_sec,
                name,
                _console_safe(part.function_response.response),
            )
        else:
            log.debug(
                "%s -> %s",
                name,
                _console_safe(part.function_response.response),
            )
            self._last_time = time.time()
