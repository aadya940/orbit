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
  `value` is the text or option. To enter or paste text (into a field, a
  text editor, a document), ALWAYS use act(kind="fill", target=<the text
  area>, value=<the text>) — never press.
- press(keys): press a SINGLE key or shortcut chord only, e.g. "Enter",
  "ctrl+s", "Tab". Never pass free text here.
- navigate(value): open a URL or application.
- observe(detail, offset, query): request screen content at the detail you
  need. Default is a compact summary. If the answer should be on-screen
  but is not in what you were shown, do NOT scroll or click around —
  call observe(detail="text", offset=...) to page through the full text,
  or observe(query="...") to get text around matches. Elision notices
  tell you exactly how much more exists.
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
        tool("observe", "Request screen content at a chosen detail level", {
            "detail": {
                "type": "string",
                "enum": ["summary", "text", "elements", "full"],
                "description": "summary (default): elements + start of text. "
                               "text: page through the full visible text. "
                               "elements: the complete element list. "
                               "full: everything (expensive).",
            },
            "offset": {"type": "integer",
                       "description": "char offset into visible text (with detail='text')"},
            "query": {"type": "string",
                      "description": "return text surrounding matches of this string"},
        }, []),
        tool("ask_human", "Escalate to a human", {"reason": {"type": "string"}}, ["reason"]),
        tool("finish", "End the run with the final output", {"output": finish_output}, []),
    ]


# Hard ceilings: the model chooses *within* these; it never gets unbounded
# output. Allocation is the model's call, the total is the loop's.
_SUMMARY_TEXT = 2_000
_TEXT_PAGE = 8_000
_MAX_ELEMENTS_SUMMARY = 80
_MAX_ELEMENTS_FULL = 400
_QUERY_WINDOW = 400
_QUERY_MAX_HITS = 5


def _render_elements(obs: Observation, limit: int) -> List[str]:
    lines = ["elements:"]
    for e in obs.elements[:limit]:
        val = f" value={e.value!r}" if e.value is not None else ""
        hint = f" ({e.hint})" if getattr(e, "hint", None) else ""
        flags = "" if e.enabled else " (disabled)"
        lines.append(f"- {e.role} {e.name!r}{hint}{val}{flags}")
    if len(obs.elements) > limit:
        lines.append(
            f"[... {len(obs.elements) - limit} more elements elided — "
            "observe(detail='elements') for all]"
        )
    return lines


def _render_text(obs: Observation, offset: int, limit: int) -> List[str]:
    total = len(obs.text)
    offset = max(0, min(offset, total))
    chunk = obs.text[offset:offset + limit]
    lines = [f"visible text (chars {offset}-{offset + len(chunk)} of {total}):", chunk]
    if offset + len(chunk) < total:
        lines.append(
            f"[... {total - offset - len(chunk)} more chars — "
            f"observe(detail='text', offset={offset + len(chunk)}) for the next page]"
        )
    return lines


def _render_query(obs: Observation, query: str) -> List[str]:
    text, needle = obs.text, query.lower()
    hits, start = [], 0
    while len(hits) < _QUERY_MAX_HITS:
        i = text.lower().find(needle, start)
        if i < 0:
            break
        lo, hi = max(0, i - _QUERY_WINDOW), min(len(text), i + len(needle) + _QUERY_WINDOW)
        hits.append(f"...{text[lo:hi]}...")
        start = hi
    if not hits:
        return [f"no matches for {query!r} in visible text ({len(text)} chars total)"]
    return [f"text around {len(hits)} match(es) for {query!r}:"] + hits


def _render(obs: Observation, detail: str = "summary", offset: int = 0,
            query: Optional[str] = None) -> str:
    lines = [
        f"[observation {obs.content_hash}] surface={obs.surface} kind={obs.kind}",
        f"title: {obs.title}",
    ]
    if obs.url:
        lines.append(f"url: {obs.url}")
    if obs.modal_count:
        lines.append(f"modals open: {obs.modal_count}")

    if query:
        lines += _render_elements(obs, _MAX_ELEMENTS_SUMMARY)
        lines += _render_query(obs, query)
    elif detail == "text":
        lines += _render_text(obs, offset, _TEXT_PAGE)
    elif detail == "elements":
        lines += _render_elements(obs, _MAX_ELEMENTS_FULL)
    elif detail == "full":
        lines += _render_elements(obs, _MAX_ELEMENTS_FULL)
        lines += _render_text(obs, 0, _TEXT_PAGE + _SUMMARY_TEXT)
    else:  # summary
        lines += _render_elements(obs, _MAX_ELEMENTS_SUMMARY)
        if obs.text:
            lines += _render_text(obs, 0, _SUMMARY_TEXT)
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
    step_idx = 0

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
                messages.append({"role": "user", "content": _render(obs)})
            elif obs.content_hash != last_obs.content_hash:
                from .types import diff_observations
                d = diff_observations(last_obs, obs)
                messages.append({"role": "user", "content": (
                    f"[observation {obs.content_hash}] changes since last: {d.summary()}"
                )})
            last_obs = obs

        # Think (one budget step per LLM call).
        step_idx += 1
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
        if reply.tool_call is None:
            # Text-only reply: record it verbatim, demand a real function
            # call. Never echo a fake "[tool call] ..." text format — the
            # model imitates whatever the history looks like.
            if reply.text:
                messages.append({"role": "assistant", "content": reply.text})
            messages.append({"role": "user", "content": (
                "Respond with exactly one function call (act / press / navigate / "
                "observe / ask_human / finish), not text."
            )})
            continue

        name = reply.tool_call.name
        args = reply.tool_call.arguments or {}
        call_id = f"call_{step_idx}"
        messages.append({
            "role": "assistant",
            "content": reply.text or None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, default=str)},
            }],
        })

        def tool_result(content: str) -> None:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

        if name == "finish":
            raw = args.get("output")
            try:
                validated = validate_output(raw, schema)
            except OutputInvalid as exc:
                output_retries += 1
                journal.append("output_invalid", error=exc.message, retry=output_retries)
                if output_retries > _MAX_OUTPUT_RETRIES:
                    return result(RunStatus.FAILED, error=exc)
                tool_result(
                    f"finish() output failed validation: {exc.message}. "
                    "Call finish again with a corrected output."
                )
                continue
            return result(RunStatus.SUCCESS, output=validated)

        if name == "ask_human":
            reason = args.get("reason", "")
            journal.append("needs_human", reason=reason)
            return result(RunStatus.NEEDS_HUMAN, error=NeedsHuman(reason))

        if name == "observe":
            if last_obs is None:
                tool_result("no observation available")
            else:
                tool_result(_render(
                    last_obs,
                    detail=str(args.get("detail") or "summary"),
                    offset=int(args.get("offset") or 0),
                    query=args.get("query"),
                ))
            continue

        # Action-mapping tools.
        if name == "act":
            try:
                kind = ActionKind(args.get("kind", "click"))
            except ValueError:
                tool_result(f"unknown action kind: {args.get('kind')!r}")
                continue
            action = Action(kind=kind, target=args.get("target"), value=args.get("value"))
        elif name == "press":
            action = Action(kind=ActionKind.PRESS, value=args.get("keys"), expects_effect=False)
        elif name == "navigate":
            action = Action(kind=ActionKind.NAVIGATE, value=args.get("value"))
        else:
            tool_result(f"unknown tool: {name}")
            continue

        try:
            action_result = await world.act(action)
        except TargetUnresolvable as exc:
            tool_result(
                f"TOOL ERROR ({exc.code}): {exc.message}. "
                "Rephrase the target, try another approach, or finish/ask_human."
            )
            continue
        except NeedsHuman as exc:
            journal.append("needs_human", reason=exc.message)
            return result(RunStatus.NEEDS_HUMAN, error=exc)
        tool_result(
            f"landed via {action_result.strategy}; {action_result.diff.summary()}"
        )
