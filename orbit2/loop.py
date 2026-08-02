"""The owned agent loop: observe -> think -> act, with delta prompting.

The model sees a uniform toolset (act/press/navigate/observe/ask_human/
finish) and never picks drivers — fallback is mechanical in the World.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, List, Optional, Type

from pydantic import BaseModel

from .llm import LLM, LLMReply
from .types import (
    Action,
    ActionKind,
    BudgetExhausted,
    NeedsHuman,
    Observation,
    OrbitError,
    OutputInvalid,
    RunResult,
    RunStatus,
    TargetUnresolvable,
    validate_output,
)
from .world import World

SYSTEM_PROMPT = """You are Orbit, an agent operating a computer to complete one task.

You interact only through tools. Each turn, review the latest observation
or tool result, then call exactly one tool:
- act(kind, target, value): perform click / fill / select / scroll on the
  element best matching the natural-language `target`. For fill/select,
  `value` is the text or option.
- press(keys): press a key or chord, e.g. "Enter", "ctrl+s".
- navigate(value): open a URL or application.
- observe(): request a fresh snapshot of the screen.
- ask_human(reason): escalate when blocked (CAPTCHA, login, ambiguity,
  destructive step needing approval).
- finish(output): end the run with your final result. If a schema was
  given, `output` must be a JSON object matching it exactly.

Rules:
- One tool call per turn. Never invent elements not in the observation.
- After each action you get a state-change summary; if nothing changed,
  try a different target phrasing or approach rather than repeating.
- Prefer the fewest steps that reliably complete the task.
- When the task is done (or is read-only), call finish immediately.
"""

_MAX_OUTPUT_RETRIES = 2


def _tool_defs(schema: Optional[Type[BaseModel]]) -> List[dict]:
    def tool(name: str, desc: str, props: dict, required: List[str]) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }

    finish_output: dict = {"description": "final result"}
    if schema is not None:
        finish_output = schema.model_json_schema()
    return [
        tool("act", "Perform an action on the current surface", {
            "kind": {"type": "string", "enum": ["click", "fill", "select", "scroll"]},
            "target": {"type": "string", "description": "natural-language element description"},
            "value": {"type": "string", "description": "text / option / scroll direction"},
        }, ["kind"]),
        tool("press", "Press a key or chord", {"keys": {"type": "string"}}, ["keys"]),
        tool("navigate", "Open a URL or app", {"value": {"type": "string"}}, ["value"]),
        tool("observe", "Force a fresh observation", {}, []),
        tool("ask_human", "Escalate to a human", {"reason": {"type": "string"}}, ["reason"]),
        tool("finish", "End the run with the final output", {"output": finish_output}, []),
    ]


def _render_full(obs: Observation) -> str:
    lines = [
        f"[observation {obs.content_hash}] surface={obs.surface} kind={obs.kind}",
        f"title: {obs.title}",
    ]
    if obs.url:
        lines.append(f"url: {obs.url}")
    if obs.modal_count:
        lines.append(f"modals open: {obs.modal_count}")
    lines.append("elements:")
    for e in obs.elements[:150]:
        val = f" value={e.value!r}" if e.value is not None else ""
        flags = "" if e.enabled else " (disabled)"
        lines.append(f"- {e.role} {e.name!r}{val}{flags}")
    if obs.text:
        lines.append(f"visible text: {obs.text[:2000]}")
    return "\n".join(lines)


async def run(
    task: str,
    world: World,
    llm: LLM,
    schema: Optional[Type[BaseModel]] = None,
    max_steps: Optional[int] = None,
    guidance: Optional[str] = None,
    timeout: Optional[float] = None,
) -> RunResult:
    if max_steps is not None:
        world.max_steps = max_steps
    try:
        if timeout is not None:
            return await asyncio.wait_for(_run(task, world, llm, schema, guidance), timeout)
        return await _run(task, world, llm, schema, guidance)
    except asyncio.TimeoutError:
        world.journal.append("timeout", task=task)
        return RunResult(
            status=RunStatus.TIMEOUT, steps_used=world.used,
            journal=world.journal.to_list(),
        )


async def _run(
    task: str,
    world: World,
    llm: LLM,
    schema: Optional[Type[BaseModel]],
    guidance: Optional[str],
) -> RunResult:
    journal = world.journal
    journal.append("run_start", task=task)
    tools = _tool_defs(schema)

    user = f"TASK: {task}"
    if schema is not None:
        user += f"\n\nFinish with output matching this schema:\n{json.dumps(schema.model_json_schema())}"
    if guidance:
        user += f"\n\n<user_guidance>\n{guidance}\n</user_guidance>"
    messages: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    last_obs: Optional[Observation] = None
    output_retries = 0

    def result(status: RunStatus, output: Any = None, error: Optional[OrbitError] = None) -> RunResult:
        journal.append("run_end", status=status.value)
        return RunResult(
            status=status, output=output, steps_used=world.used,
            error=error, journal=journal.to_list(),
        )

    while True:
        # Observe: full render first time, deltas afterwards.
        try:
            obs = await world.observe()
        except OrbitError as exc:
            journal.append("observe_error", error=exc.code, message=exc.message)
            messages.append({"role": "user", "content": f"OBSERVE ERROR: {exc.code}: {exc.message}"})
            obs = None
        if obs is not None:
            if last_obs is None:
                messages.append({"role": "user", "content": _render_full(obs)})
            elif obs.content_hash != last_obs.content_hash:
                from .types import diff_observations
                d = diff_observations(last_obs, obs)
                messages.append({"role": "user", "content": (
                    f"[observation {obs.content_hash}] changes since last: {d.summary()}"
                )})
            last_obs = obs

        # Think (one budget step per LLM call).
        try:
            world.spend(1)
        except BudgetExhausted as exc:
            return result(RunStatus.BUDGET_EXHAUSTED, error=exc)
        reply: LLMReply = await llm.complete(messages, tools)
        journal.append(
            "llm_reply", text=reply.text,
            tool=reply.tool_call.name if reply.tool_call else None,
            arguments=reply.tool_call.arguments if reply.tool_call else None,
        )
        if reply.text:
            messages.append({"role": "assistant", "content": reply.text})
        if reply.tool_call is None:
            messages.append({"role": "user", "content": "Please respond with exactly one tool call."})
            continue

        name = reply.tool_call.name
        args = reply.tool_call.arguments or {}
        messages.append({"role": "assistant", "content": f"[tool call] {name}({json.dumps(args, default=str)})"})

        if name == "finish":
            raw = args.get("output")
            try:
                validated = validate_output(raw, schema)
            except OutputInvalid as exc:
                output_retries += 1
                journal.append("output_invalid", error=exc.message, retry=output_retries)
                if output_retries > _MAX_OUTPUT_RETRIES:
                    return result(RunStatus.FAILED, error=exc)
                messages.append({"role": "user", "content": (
                    f"finish() output failed validation: {exc.message}. "
                    "Call finish again with a corrected output."
                )})
                continue
            return result(RunStatus.SUCCESS, output=validated)

        if name == "ask_human":
            reason = args.get("reason", "")
            journal.append("needs_human", reason=reason)
            return result(RunStatus.NEEDS_HUMAN, error=NeedsHuman(reason))

        if name == "observe":
            if last_obs is not None:
                messages.append({"role": "user", "content": _render_full(last_obs)})
            continue

        # Action-mapping tools.
        if name == "act":
            try:
                kind = ActionKind(args.get("kind", "click"))
            except ValueError:
                messages.append({"role": "user", "content": f"unknown action kind: {args.get('kind')!r}"})
                continue
            action = Action(kind=kind, target=args.get("target"), value=args.get("value"))
        elif name == "press":
            action = Action(kind=ActionKind.PRESS, value=args.get("keys"), expects_effect=False)
        elif name == "navigate":
            action = Action(kind=ActionKind.NAVIGATE, value=args.get("value"))
        else:
            messages.append({"role": "user", "content": f"unknown tool: {name}"})
            continue

        try:
            action_result = await world.act(action)
        except TargetUnresolvable as exc:
            messages.append({"role": "user", "content": (
                f"TOOL ERROR ({exc.code}): {exc.message}. "
                "Rephrase the target, try another approach, or finish/ask_human."
            )})
            continue
        except NeedsHuman as exc:
            journal.append("needs_human", reason=exc.message)
            return result(RunStatus.NEEDS_HUMAN, error=exc)
        messages.append({"role": "user", "content": (
            f"TOOL RESULT: landed via {action_result.strategy}; {action_result.diff.summary()}"
        )})
