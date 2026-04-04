from google.adk.agents import Agent
from google.genai import types

from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.tools import AgentTool
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm
from typing import Any, Optional
import logging

log = logging.getLogger("orbit.agents")

from .prompts import SYSTEM_PROMPT, PARENT_SYSTEM_PROMPT
from ._tools.ui import (
    list_active_windows,
    manage_window,
    take_screenshot,
    fill_form_fields,
    find_ui_elements,
    find_ui_elements_hwnd,
    get_window_tree,
    get_window_tree_hwnd,
    interact_with_element,
    wait_for_element,
    click_first,
    type_into,
    navigate_to_url,
    launch_and_get_pid,
    scroll_page,
    get_form_fields,
    select_dropdown_option,
    select_option_by_label,
    get_popuphost_menu_window,
    get_page_text,
    wait_for_text,
)
from ._tools.clipboard import (
    clipboard_get,
    clipboard_set,
)
from ._tools.search import duckduckgo_search
from ._tools.filesystem import (
    get_system_info,
    find_installed_apps,
    read_file,
    read_pdf,
    read_csv,
    list_directory,
    search_files,
    file_exists,
    get_file_info,
    find_in_file,
)
from ._tools.hitl import (
    write_file as write_file_approval,
    append_to_file as append_to_file_approval,
    write_csv as write_csv_approval,
    copy_file as copy_file_approval,
    move_file as move_file_approval,
    move_files as move_files_approval,
    create_directory_and_move as create_directory_and_move_approval,
    delete_file,
    create_directory as create_directory_approval,
    upload_file as upload_file_approval,
    request_human,
    run_shell as run_shell_approval,
)
from ._tools.hotkey import press_hotkey

DEFAULT_DESKTOP_MODEL = "gemini-3-flash-preview"
DEFAULT_PLANNER_MODEL = "gemini-3-flash-preview"

DESKTOP_EXECUTOR_AGENT_NAME = "desktop_agent"


def make_lite_llm(model: str):
    """
    Create an ADK LiteLlm from the user-provided model string.

    ADK + LiteLLM typically expects provider-prefixed model strings (`provider/model-name`).
    To keep user experience simple, we normalize the common raw Gemini format
    `gemini-3-pro-preview` into `gemini/gemini-3-pro-preview` (provider prefix).
    For any already provider-prefixed model (contains `/`), we pass it through unchanged.
    """
    m = (model or "").strip()
    # Use ADK native Gemini to preserve thought signatures/tool-calling behavior.
    # Accept common forms:
    # - gemini-3-pro-preview
    # - gemini/gemini-3-pro-preview
    # - google/gemini-3-pro-preview
    # - openrouter/google/gemini-3-pro-preview-customtools
    if "gemini-" in m:
        if m.startswith("gemini-"):
            return Gemini(model=m)
        parts = [p for p in m.split("/") if p]
        for part in reversed(parts):
            if part.startswith("gemini-"):
                return Gemini(model=part)
    return LiteLlm(model=model)


_BUDGET_WARNING_THRESHOLD = 5  # Warn when this many calls remain.
_LOOP_WINDOW = 10  # Look at last N tool calls for repetition.
_LOOP_THRESHOLD = 3  # Same tool+args returning error this many times → loop.


def make_inject_screenshot_callback(
    *, max_calls: int, budget_counter: Optional[dict[str, int]] = None
):
    # Closure state for loop detection: only tracks calls that returned errors.
    _failed_calls: list[str] = []

    async def _inject_screenshot_callback(
        callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        """
        Before each desktop agent LLM call:
        1. Track call count; hard-stop when budget exhausted, warn when running low.
        2. Detect tool-call loops and inject recovery hint.
        3. Inject screenshot artifacts as inline images.
        """
        # ── Budget tracking ────────────────────────────────────────────
        # Keep this per-run using a closure-backed counter. This avoids relying on
        # direct Session.state mutations outside callback/tool context.
        if budget_counter is None:
            call_num = (
                int(callback_context.state.get("temp:orbit_call_count", 0) or 0) + 1
            )
            callback_context.state["temp:orbit_call_count"] = call_num
        else:
            budget_counter["call_count"] = (
                int(budget_counter.get("call_count", 0) or 0) + 1
            )
            call_num = budget_counter["call_count"]
            callback_context.state["temp:orbit_call_count"] = call_num

        remaining = max(0, int(max_calls) - call_num)

        # ── Log recent tool calls ─────────────────────────────────────
        # ADK does not stream sub-agent events, so the runner can't log
        # desktop agent tool calls. We log them here instead.
        if llm_request.contents:
            for c in llm_request.contents[-2:]:
                for p in c.parts or []:
                    fc = getattr(p, "function_call", None)
                    if fc and fc.name:
                        args_summary = str(dict(fc.args))[:200] if fc.args else "{}"
                        log.info(
                            "[call %d/%d] %s(%s)",
                            call_num,
                            max_calls,
                            fc.name,
                            args_summary,
                        )
                    fr = getattr(p, "function_response", None)
                    if fr and fr.name:
                        resp_summary = str(fr.response)[:200] if fr.response else ""
                        log.info(
                            "[call %d/%d] %s -> %s",
                            call_num,
                            max_calls,
                            fr.name,
                            resp_summary,
                        )

        # Hard-stop: return a canned response so the LLM is never called.
        if remaining <= 0:
            log.warning(
                "Budget exhausted (%d/%d calls). Forcing stop.", call_num, max_calls
            )
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text="[BUDGET EXHAUSTED] Stopping — no LLM calls remaining."
                        )
                    ],
                )
            )

        if remaining <= _BUDGET_WARNING_THRESHOLD and llm_request.contents:
            budget_part = types.Part(
                text=f"[BUDGET] {remaining} LLM calls remaining. "
                "Finish the current step now and return."
            )
            # Merge into last user content to avoid breaking role alternation.
            last = llm_request.contents[-1]
            if last.role == "user" and last.parts:
                last.parts.append(budget_part)
            else:
                llm_request.contents.append(
                    types.Content(role="user", parts=[budget_part])
                )

        # ── Loop detection ─────────────────────────────────────────
        # Only track tool calls that returned errors. Successful repeated calls
        # (e.g. find_ui_elements with different queries, interact_with_element
        # for each form field) are normal and should NOT trigger this.
        if llm_request.contents:
            for c in llm_request.contents[-3:]:
                for p in c.parts or []:
                    fr = getattr(p, "function_response", None)
                    if fr and fr.response and isinstance(fr.response, dict):
                        if fr.response.get("status") == "error":
                            _failed_calls.append(fr.name)
            # Trim so we don't leak memory.
            del _failed_calls[: max(0, len(_failed_calls) - _LOOP_WINDOW * 2)]

            window = _failed_calls[-_LOOP_WINDOW:]
            if len(window) >= _LOOP_THRESHOLD:
                from collections import Counter

                top_call, top_count = Counter(window).most_common(1)[0]
                if top_count >= _LOOP_THRESHOLD:
                    log.warning(
                        "Loop detected: %s failed %d times in recent window. Injecting recovery hint.",
                        top_call,
                        top_count,
                    )
                    hint = types.Part(
                        text=(
                            f"[LOOP DETECTED] '{top_call}' has failed {top_count} times. "
                            "This approach is not working. STOP retrying and either: "
                            "(1) try a completely different approach, or "
                            "(2) call request_human to ask for help. "
                            "Do NOT call the same tool with the same arguments again."
                        )
                    )
                    last = llm_request.contents[-1]
                    if last.role == "user" and last.parts:
                        last.parts.append(hint)
                    else:
                        llm_request.contents.append(
                            types.Content(role="user", parts=[hint])
                        )
                    _failed_calls.clear()

        # ── Truncate oversized function responses ──────────────────
        # Large accessibility tree / element dumps can cause Gemini
        # to reject the request with INVALID_ARGUMENT.
        _MAX_RESP_CHARS = 8000
        for c in llm_request.contents or []:
            for p in c.parts or []:
                if (
                    hasattr(p, "function_response")
                    and p.function_response
                    and p.function_response.response
                ):
                    resp_str = str(p.function_response.response)
                    if len(resp_str) > _MAX_RESP_CHARS:
                        truncated = resp_str[:_MAX_RESP_CHARS] + "\n...[TRUNCATED]"
                        p.function_response.response = {
                            "status": "success",
                            "note": "Response truncated to fit context window.",
                            "data": truncated,
                        }

        if not llm_request.contents:
            return None
        content = llm_request.contents[-1]
        if not content.parts:
            return None
        for part in content.parts:
            if (
                hasattr(part, "function_response")
                and part.function_response
                and part.function_response.name == "take_screenshot"
            ):
                response = part.function_response.response
                if response.get("status") == "success":
                    artifact = await callback_context.load_artifact("screenshot.jpg")
                    if (
                        artifact
                        and artifact.inline_data
                        and artifact.inline_data.data
                        and isinstance(artifact.inline_data.data, bytes)
                        and len(artifact.inline_data.data) > 100
                    ):
                        llm_request.contents.append(
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part(
                                        inline_data=types.Blob(
                                            mime_type="image/jpeg",
                                            data=artifact.inline_data.data,
                                        )
                                    ),
                                    types.Part(text="This is the current screenshot."),
                                ],
                            )
                        )
                    else:
                        log.warning(
                            "Skipping screenshot injection — artifact data missing or corrupt."
                        )
        return None

    return _inject_screenshot_callback


def capture_phase_instruction_before_agent_callback(
    callback_context: CallbackContext,
) -> None:
    # Planner invokes desktop_agent via AgentTool; phase text arrives as user_content.
    user_content = callback_context.user_content
    phase_text = ""
    if user_content and user_content.parts:
        phase_text = getattr(user_content.parts[0], "text", "") or ""
    callback_context.state["journal_phase_instruction"] = phase_text
    return None


_planner = BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=512))


def system_prompt_provider(context: ReadonlyContext) -> str:
    return SYSTEM_PROMPT


def parent_prompt_provider(context: ReadonlyContext) -> str:
    return PARENT_SYSTEM_PROMPT


def build_desktop_agent(
    desktop_model: str,
    extra_tools: Optional[list] = None,
    max_calls: int = 30,
    budget_counter: Optional[dict[str, int]] = None,
    output_schema: Optional[type] = None,
    output_key: Optional[str] = None,
) -> Agent:
    kwargs: dict[str, Any] = dict(
        model=make_lite_llm(desktop_model),
        name=DESKTOP_EXECUTOR_AGENT_NAME,
        description="""Handles all desktop UI automation: browser control, forms, dropdowns,
        file uploads, job applications (LinkedIn Easy Apply, Indeed).
        Delegate any phase that requires interacting with the screen or apps to this agent.
        This agent is responsible for all the desktop UI automation tasks.""",
        instruction=system_prompt_provider,
        before_model_callback=make_inject_screenshot_callback(
            max_calls=max_calls,
            budget_counter=budget_counter,
        ),
        before_agent_callback=capture_phase_instruction_before_agent_callback,
        tools=[
            list_active_windows,
            manage_window,
            click_first,
            type_into,
            find_ui_elements,
            fill_form_fields,
            find_ui_elements_hwnd,
            get_window_tree,
            get_window_tree_hwnd,
            interact_with_element,
            wait_for_element,
            scroll_page,
            get_form_fields,
            select_dropdown_option,
            select_option_by_label,
            clipboard_get,
            clipboard_set,
            list_directory,
            duckduckgo_search,
            move_file_approval,
            move_files_approval,
            create_directory_and_move_approval,
            write_file_approval,
            read_file,
            append_to_file_approval,
            read_pdf,
            read_csv,
            write_csv_approval,
            search_files,
            file_exists,
            get_file_info,
            copy_file_approval,
            delete_file,
            create_directory_approval,
            find_in_file,
            get_system_info,
            find_installed_apps,
            press_hotkey,
            navigate_to_url,
            launch_and_get_pid,
            take_screenshot,
            get_popuphost_menu_window,
            upload_file_approval,
            request_human,
            get_page_text,
            wait_for_text,
            run_shell_approval,
        ]
        + (extra_tools or []),
    )
    if output_schema is not None:
        kwargs["output_schema"] = output_schema
    if output_key is not None:
        kwargs["output_key"] = output_key
    return Agent(**kwargs)


def build_parent_agent(
    planner_model: str,
    desktop_agent: Agent,
    output_schema: Optional[type] = None,
    output_key: Optional[str] = None,
) -> Agent:
    desktop_tool = AgentTool(desktop_agent)
    kwargs: dict[str, Any] = dict(
        model=make_lite_llm(planner_model),
        name="planner",
        planner=_planner,
        description="""High-level planner that breaks goals into phases and delegates desktop automation to desktop_agent.
        This agent is responsible for breaking down the user's goal into clear phases and delegating the tasks to the desktop_agent tool.""",
        instruction=parent_prompt_provider,
        tools=[duckduckgo_search, desktop_tool],
    )
    if output_schema is not None:
        kwargs["output_schema"] = output_schema
    if output_key is not None:
        kwargs["output_key"] = output_key
    return Agent(**kwargs)


def build_agents(
    *,
    desktop_model: str = DEFAULT_DESKTOP_MODEL,
    planner_model: str = DEFAULT_PLANNER_MODEL,
    planner: bool = True,
    extra_tools: Optional[list] = None,
    output_schema: Optional[type] = None,
    max_calls: int = 30,
    budget_counter: Optional[dict[str, int]] = None,
    output_key: Optional[str] = None,
) -> tuple[Agent, Agent]:
    """Return (root_agent, desktop_agent) for the requested model strings."""
    desktop_agent = build_desktop_agent(
        desktop_model,
        extra_tools=extra_tools,
        max_calls=max_calls,
        budget_counter=budget_counter,
        output_schema=output_schema if not planner else None,
        output_key=output_key if not planner else None,
    )
    if planner:
        root_agent = build_parent_agent(
            planner_model,
            desktop_agent,
            output_schema=output_schema,
            output_key=output_key,
        )
    else:
        root_agent = desktop_agent
    return root_agent, desktop_agent
