"""The owned agent loop: observe -> think -> act, with delta prompting.

The model sees a uniform toolset (act/press/navigate/observe/ask_human/
finish) and never picks drivers. Fallback is mechanical in the World.
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
- act(kind, ref, value): perform click / fill / select / scroll on an
  element. Address it by `ref`, the [number] shown next to it in the
  observation. Only fall back to `target` (a description) if nothing in
  the list fits. For fill/select, `value` is the text or option. To enter
  text (a field, a text editor, a document), ALWAYS use
  act(kind="fill", ref=<the text area>, value=<the text>), never press.
- press(keys): press a SINGLE key or shortcut chord only, e.g. "Enter",
  "ctrl+s", "Tab". Never pass free text here.
- navigate(value): open a URL or application.
- observe(detail, offset, query): request screen content at the detail you
  need. Default is a compact summary. If the answer should be on-screen
  but is not in what you were shown, do NOT scroll or click around:
  call observe(detail="text", offset=...) to page through the full text,
  or observe(query="...") to get text around matches. Elision notices
  tell you exactly how much more exists.
- ask_human(reason): escalate when blocked (CAPTCHA, login, ambiguity,
  destructive step needing approval).
- finish(output): end the run with your final result. If a schema was
  given, `output` must be a JSON object matching it exactly.

You may also have tools for files, code, the clipboard and the web
(listed in your tool schema). Prefer them when they are the shorter
path: read a file directly instead of opening it in an editor, search
or fetch a page instead of browsing to it, run_python for parsing and
math. Drive the screen when the task genuinely needs the app.

Rules:
- One tool call per turn. Never invent elements not in the observation.
- After each action you get a state-change summary; if nothing changed,
  try a different target phrasing or approach rather than repeating.
- Prefer the fewest steps that reliably complete the task.
- When the task is done (or is read-only), call finish immediately.
"""

_MAX_OUTPUT_RETRIES = 2


def _tool_defs(schema: Optional[Type[BaseModel]],
               registry: Optional[dict] = None) -> List[dict]:
    """Build the OpenAI-style tool schema list offered to the model.

    Parameters
    ----------
    schema : Type[BaseModel], optional
        Pydantic model describing the required final output. When given, its
        JSON schema becomes the `output` parameter of `finish`. Default is
        None, which leaves `finish` output free-form.
    registry : dict, optional
        Extra tools by name, each exposing a `schema()` method, appended
        after the built-in tools. Default is None.

    Returns
    -------
    List[dict]
        Tool definitions in function-calling format: the fixed verbs (act,
        press, navigate, observe, ask_human, finish) plus registry tools.

    Notes
    -----
    The model sees a uniform toolset and never picks drivers; fallback is
    mechanical in the World. Tool calls must reach the model as real
    tool_calls, so these definitions are the only contract it is given.
    """
    def tool(name: str, desc: str, props: dict, required: List[str]) -> dict:
        """Assemble one function-calling tool definition.

        Parameters
        ----------
        name : str
            Tool name as the model will call it.
        desc : str
            Human-readable description shown to the model.
        props : dict
            JSON schema properties for the tool's parameters.
        required : List[str]
            Names of the required parameters.

        Returns
        -------
        dict
            A single tool definition in function-calling format.
        """
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
            "ref": {"type": "integer",
                    "description": "PREFERRED: the [number] of an element from the "
                                   "observation. Exact, with no name guessing."},
            "target": {"type": "string",
                       "description": "fallback when no ref fits: natural-language "
                                      "element description"},
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
    ] + [t.schema() for t in (registry or {}).values()]


# Hard ceilings: the model chooses *within* these; it never gets unbounded
# output. Allocation is the model's call, the total is the loop's.
_SUMMARY_TEXT = 2_000
_TEXT_PAGE = 8_000
_MAX_ELEMENTS_SUMMARY = 80
_MAX_ELEMENTS_FULL = 400
_QUERY_WINDOW = 400
_QUERY_MAX_HITS = 5


def _render_elements(obs: Observation, limit: int) -> List[str]:
    """Render an observation's elements with their ref indices.

    Parameters
    ----------
    obs : Observation
        Observation whose elements are rendered.
    limit : int
        Maximum number of elements to render before eliding the rest.

    Returns
    -------
    List[str]
        Rendered lines, starting with a header and ending with an elision
        notice when elements were dropped.

    Notes
    -----
    Elements carry their ref index so the model points at a ref instead of
    re-describing an element we already showed it. The elision notice states
    exactly how much more exists, since observation detail is model-controlled
    within loop-owned ceilings and the model needs to know what it can ask for.
    """
    lines = ["elements (act on one with ref=<number>):"]
    for i, e in enumerate(obs.elements[:limit]):
        val = f" value={e.value!r}" if e.value is not None else ""
        hint = f" ({e.hint})" if getattr(e, "hint", None) else ""
        marks = []
        if not e.enabled:
            marks.append("disabled")
        if e.focused:
            marks.append("focused")
        flags = f" [{', '.join(marks)}]" if marks else ""
        lines.append(f"[{i}] {e.role} {e.name!r}{hint}{val}{flags}")
    if len(obs.elements) > limit:
        lines.append(
            f"[... {len(obs.elements) - limit} more elements elided, "
            "observe(detail='elements') for all]"
        )
    return lines


def _render_text(obs: Observation, offset: int, limit: int) -> List[str]:
    """Render one page of an observation's visible text.

    Parameters
    ----------
    obs : Observation
        Observation whose visible text is paged.
    offset : int
        Character offset to start from. Clamped into the valid range.
    limit : int
        Maximum number of characters to include in this page.

    Returns
    -------
    List[str]
        A header line naming the character range, the text chunk, and a
        continuation notice when more text remains.

    Notes
    -----
    Observation detail is model-controlled within loop-owned ceilings: the
    model chooses the offset, the loop caps the page size. The continuation
    notice spells out the next offset so paging never requires guessing.
    """
    total = len(obs.text)
    offset = max(0, min(offset, total))
    chunk = obs.text[offset:offset + limit]
    lines = [f"visible text (chars {offset}-{offset + len(chunk)} of {total}):", chunk]
    if offset + len(chunk) < total:
        lines.append(
            f"[... {total - offset - len(chunk)} more chars, "
            f"observe(detail='text', offset={offset + len(chunk)}) for the next page]"
        )
    return lines


def _render_query(obs: Observation, query: str) -> List[str]:
    """Render windows of visible text surrounding matches of a query.

    Parameters
    ----------
    obs : Observation
        Observation whose visible text is searched.
    query : str
        Substring to search for, matched case-insensitively.

    Returns
    -------
    List[str]
        A header line plus one snippet per match, or a single line reporting
        no matches and the total text length.

    Notes
    -----
    Both the number of hits and the window around each are bounded by
    loop-owned ceilings: observation detail is model-controlled, but the
    totals belong to the loop. Reporting the total character count on a miss
    lets the model tell "not present" from "not yet paged in".
    """
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
    """Render an observation as the text block shown to the model.

    Parameters
    ----------
    obs : Observation
        Observation to render.
    detail : str, optional
        Detail level: "summary", "text", "elements" or "full". Unknown values
        fall through to the summary rendering. Default is "summary".
    offset : int, optional
        Character offset into the visible text, used with detail "text".
        Default is 0.
    query : str, optional
        When given, overrides `detail` and renders the element list plus text
        around matches of this string. Default is None.

    Returns
    -------
    str
        The rendered observation, newline joined.

    Notes
    -----
    Observation detail is model-controlled within loop-owned ceilings: the
    model asks for a level, the module constants cap what any level can
    return. The header always carries the content hash so the model can tell
    a fresh observation from a repeat of one it already has.
    """
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
    tools: Optional[dict] = None,
) -> RunResult:
    """Run one task to completion, with an optional wall-clock timeout.

    Parameters
    ----------
    task : str
        Natural-language description of the task to complete.
    world : World
        The World providing perception, action and the step budget.
    llm : LLM
        Model client used for every think step.
    schema : Type[BaseModel], optional
        Pydantic model the final output must validate against. Default is
        None, which accepts free-form output.
    max_steps : int, optional
        Overrides the World's step budget when given. Default is None.
    guidance : str, optional
        Extra user guidance appended to the task message. Default is None.
    timeout : float, optional
        Wall-clock timeout in seconds for the whole run. Default is None,
        meaning no timeout.
    tools : dict, optional
        Extra tools by name, offered alongside the built-in verbs. Default
        is None.

    Returns
    -------
    RunResult
        The run outcome, including status, output, steps used and journal. On
        timeout the status is TIMEOUT and the timeout is journalled.

    Notes
    -----
    This wrapper owns only budget override and timeout; the loop itself lives
    in `_run`. A timeout is reported as a normal result rather than raised,
    so callers always get a journal back for the partial run.
    """
    if max_steps is not None:
        world.max_steps = max_steps
    try:
        if timeout is not None:
            return await asyncio.wait_for(
                _run(task, world, llm, schema, guidance, tools), timeout)
        return await _run(task, world, llm, schema, guidance, tools)
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
    registry: Optional[dict] = None,
) -> RunResult:
    """Drive the observe, think, act loop until the run ends.

    Parameters
    ----------
    task : str
        Natural-language description of the task to complete.
    world : World
        The World providing perception, action and the step budget.
    llm : LLM
        Model client used for every think step.
    schema : Type[BaseModel] or None
        Pydantic model the final output must validate against, or None to
        accept free-form output.
    guidance : str or None
        Extra user guidance appended to the task message, or None.
    registry : dict, optional
        Extra tools by name, each exposing `schema()` and `call()`. Default
        is None.

    Returns
    -------
    RunResult
        The terminal result: SUCCESS with validated output, NEEDS_HUMAN,
        BUDGET_EXHAUSTED, or FAILED when output validation keeps failing.

    Notes
    -----
    Tool calls go into history as real tool_calls with matching tool results.
    Echoing a fake "[tool call] ..." text format taught the model to imitate
    that format and stop calling functions, so a text-only reply is recorded
    verbatim and answered with a demand for a real function call. Observation
    detail is model-controlled within loop-owned ceilings, which is why the
    observe verb re-renders the last observation rather than re-perceiving.
    Output is validated strictly with retries fed back to the model, never
    coerced; after `_MAX_OUTPUT_RETRIES` the run fails rather than returning
    something that does not match the schema. Perception is delta-rendered:
    full render the first time, then only changes since the last observation.
    """
    journal = world.journal
    journal.append("run_start", task=task)
    tools = _tool_defs(schema, registry)

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
        """Journal the end of the run and build the terminal RunResult.

        Parameters
        ----------
        status : RunStatus
            Terminal status of the run.
        output : Any, optional
            Validated final output, when the run succeeded. Default is None.
        error : OrbitError, optional
            Error explaining a non-successful status. Default is None.

        Returns
        -------
        RunResult
            Result carrying the status, output, steps used, error and the
            journal snapshot.

        Notes
        -----
        Every exit path goes through here so no run can end without a
        "run_end" entry and a complete journal.
        """
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
            # call. Never echo a fake "[tool call] ..." text format, since the
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
            """Append a tool result matching the call just made.

            Parameters
            ----------
            content : str
                Text result to hand back to the model for this tool call.

            Returns
            -------
            None
                This closure appends to the message history in place.

            Notes
            -----
            Results are appended as real tool-role messages keyed to the
            current `call_id`. Echoing a fake "[tool call] ..." text format
            taught the model to imitate it and stop calling functions, so
            every outcome (including errors) comes back through this channel.
            """
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
            ref = args.get("ref")
            action = Action(
                kind=kind,
                ref=int(ref) if ref is not None else None,
                target=args.get("target"),
                value=args.get("value"),
            )
        elif name == "press":
            action = Action(kind=ActionKind.PRESS, value=args.get("keys"), expects_effect=False)
        elif name == "navigate":
            action = Action(kind=ActionKind.NAVIGATE, value=args.get("value"))
        elif registry and name in registry:
            try:
                out = await registry[name].call(**args)
            except Exception as exc:  # tool errors are the model's to handle
                out = f"{type(exc).__name__}: {exc}"
            journal.append("tool", name=name, args=args, result=out[:400])
            tool_result(out)
            continue
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
