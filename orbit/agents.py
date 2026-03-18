from google.adk.agents import Agent
from google.genai import types

from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.tools import AgentTool, LongRunningFunctionTool
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.lite_llm import LiteLlm

import os

from .prompts import SYSTEM_PROMPT, PARENT_SYSTEM_PROMPT
from ._tools.ui import (
    list_active_windows,
    manage_window,
    find_ui_elements,
    find_ui_elements_hwnd,
    get_window_tree,
    get_window_tree_hwnd,
    interact_with_element,
    wait_for_element,
    navigate_to_url,
    launch_and_get_pid,
    mouse_click,
    mouse_type,
    take_screenshot,
    scroll_page,
    get_form_fields,
    select_dropdown_option,
    select_option_by_label,
    get_popuphost_menu_window,
)
from ._tools.clipboard import (
    clipboard_get,
    clipboard_set,
)
from ._tools.filesystem import (
    get_system_info,
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
)
from ._tools.hotkey import press_hotkey

DEFAULT_DESKTOP_MODEL = "gemini-3-pro-preview"
DEFAULT_PLANNER_MODEL = "gemini-3-pro-preview"


def make_lite_llm(model: str) -> LiteLlm:
    """
    Create an ADK LiteLlm from the user-provided model string.

    ADK + LiteLLM typically expects provider-prefixed model strings (`provider/model-name`).
    To keep user experience simple, we normalize the common raw Gemini format
    `gemini-3-pro-preview` into `gemini/gemini-3-pro-preview` (provider prefix).
    For any already provider-prefixed model (contains `/`), we pass it through unchanged.
    """
    if "/" not in (model or "") and model.startswith("gemini-"):
        model = f"gemini/{model}"
    return LiteLlm(model)


async def inject_screenshot_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """
    Finds screenshot artifacts in tool responses and injects
    them as inline images so the model can actually see them.
    """
    for content in llm_request.contents:
        if not content.parts:
            continue
        for part in content.parts:
            if (
                hasattr(part, "function_response")
                and part.function_response
                and part.function_response.name == "take_screenshot"
            ):

                response = part.function_response.response
                if response.get("status") == "success":
                    # Load the artifact
                    artifact = await callback_context.load_artifact("screenshot.jpg")
                    if artifact and artifact.inline_data:
                        # Inject image into model context
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
                                    types.Part(
                                        text="This is the current screenshot. Use mouse_click(x, y) to interact with what you see."
                                    ),
                                ],
                            )
                        )
    return None


_model_args = types.ThinkingConfig(thinking_budget=-1)
_planner = BuiltInPlanner(thinking_config=_model_args)


def system_prompt_provider(context: ReadonlyContext) -> str:
    return SYSTEM_PROMPT


def parent_prompt_provider(context: ReadonlyContext) -> str:
    return PARENT_SYSTEM_PROMPT


def build_desktop_agent(desktop_model: str) -> Agent:
    return Agent(
        model=make_lite_llm(desktop_model),
        name="desktop_agent",
        description="""Handles all desktop UI automation: browser control, forms, dropdowns, 
        file uploads, job applications (LinkedIn Easy Apply, Indeed). 
        Delegate any phase that requires interacting with the screen or apps to this agent.
        This agent is responsible for all the desktop UI automation tasks.""",
        instruction=system_prompt_provider,
        before_model_callback=inject_screenshot_callback,
        tools=[
            list_active_windows,
            manage_window,
            find_ui_elements,
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
            LongRunningFunctionTool(move_file_approval),
            LongRunningFunctionTool(move_files_approval),
            LongRunningFunctionTool(create_directory_and_move_approval),
            LongRunningFunctionTool(write_file_approval),
            read_file,
            LongRunningFunctionTool(append_to_file_approval),
            read_pdf,
            read_csv,
            LongRunningFunctionTool(write_csv_approval),
            search_files,
            file_exists,
            get_file_info,
            LongRunningFunctionTool(copy_file_approval),
            LongRunningFunctionTool(delete_file),
            LongRunningFunctionTool(create_directory_approval),
            find_in_file,
            get_system_info,
            press_hotkey,
            navigate_to_url,
            launch_and_get_pid,
            get_popuphost_menu_window,
            LongRunningFunctionTool(upload_file_approval),
            LongRunningFunctionTool(request_human),
        ],
        planner=_planner,
    )


def build_parent_agent(planner_model: str, desktop_agent: Agent) -> Agent:
    return Agent(
        model=make_lite_llm(planner_model),
        name="planner",
        description="""High-level planner that breaks goals into phases and delegates desktop automation to desktop_agent.
        This agent is responsible for breaking down the user's goal into clear phases and delegating the tasks to the desktop_agent.""",
        instruction=parent_prompt_provider,
        tools=[],
        sub_agents=[desktop_agent],
    )


def build_agents(
    *,
    desktop_model: str = DEFAULT_DESKTOP_MODEL,
    planner_model: str = DEFAULT_PLANNER_MODEL,
) -> tuple[Agent, Agent]:
    """Return (parent_agent, desktop_agent) for the requested model strings."""
    desktop_agent = build_desktop_agent(desktop_model)
    parent_agent = build_parent_agent(planner_model, desktop_agent)
    return parent_agent, desktop_agent


parent_agent, desktop_agent = build_agents()
