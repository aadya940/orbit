"""BaseActionAgent — the extensibility primitive for Orbit verbs and domain agents."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from .runner import Agent, RunResult

log = logging.getLogger("orbit.action")


class BaseActionAgent(ABC):
    """Abstract base for all Orbit verbs and user-defined domain agents.

    Composes :class:`Agent` internally — does **not** inherit from it.
    Subclasses must override :meth:`task_prompt`. Optionally override
    :meth:`output_schema` to get typed output via ADK's ``output_schema``.

    Usage::

        class ReadResume(BaseActionAgent):
            def __init__(self, path: str, **kw):
                super().__init__(**kw)
                self.path = path

            def task_prompt(self) -> str:
                return f"Read the resume at {self.path}"

            def output_schema(self):
                return ResumeData  # Pydantic model
    """

    def __init__(
        self,
        *,
        session: Optional[Any] = None,
        llm: str = "gemini-3-pro-preview",
        max_steps: Optional[int] = 40,
        planner: bool = True,
        verbose: bool = False,
        log_file_path: Optional[str] = None,
        extra_info: Optional[str] = None,
        extra_tools: Optional[list] = None,
        human_in_the_loop: bool = True,
        output_schema: Optional[Type] = None,
        **kwargs,
    ):
        self._session = session
        self._owns_session = session is None
        self._llm = llm
        self._max_steps = max_steps
        self._planner = planner
        self._verbose = verbose
        self._log_file_path = log_file_path
        self._extra_info = extra_info
        self._extra_tools = extra_tools or []
        self._human_in_the_loop = human_in_the_loop
        self._output_schema = output_schema
        self._kwargs = kwargs

    @abstractmethod
    def task_prompt(self) -> str:
        """Return the natural language task string for the agent."""
        ...

    def output_schema(self) -> Optional[Type]:
        """Return the output schema Pydantic model, if any.

        Can be set via the ``output_schema=`` constructor kwarg, or overridden
        in subclasses to return a hardcoded model.
        """
        return self._output_schema

    async def run(self) -> RunResult:
        """Build an internal Agent, execute the task, and return the result."""
        agent = Agent(
            task=self.task_prompt(),
            llm=self._llm,
            max_steps=self._max_steps,
            planner=self._planner,
            verbose=self._verbose,
            log_file_path=self._log_file_path,
            session=self._session,
            output_schema=self.output_schema(),
            extra_info=self._extra_info,
            extra_tools=self._extra_tools,
            human_in_the_loop=self._human_in_the_loop,
            **self._kwargs,
        )
        return await agent.run()

    async def __aenter__(self):
        if self._owns_session:
            from .session import Session

            self._session = Session()
            await self._session.__aenter__()
        return self

    async def __aexit__(self, *exc):
        if self._owns_session and self._session:
            await self._session.__aexit__(*exc)
            self._session = None
            self._owns_session = False
