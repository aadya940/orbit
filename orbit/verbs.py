"""Built-in verb classes — low-level screen primitives for programmatic control."""

from typing import Optional, Type

from .action import BaseActionAgent
from .runner import RunResult


class Do(BaseActionAgent):
    """Perform an action on the desktop.

    Usage::

        await Do("click the submit button", session=s).run()
    """

    def __init__(self, task: str, **kw):
        super().__init__(**kw)
        self._task = task

    def task_prompt(self) -> str:
        return (
            f"ACTION: {self._task}\n"
            "Perform this action on the desktop. "
            "Once the action is clearly completed, stop immediately."
        )


class Read(BaseActionAgent):
    """Extract structured data from the screen.

    Pass ``schema`` (a Pydantic model) to get typed output via ADK
    ``output_schema``.

    Usage::

        result = await Read("job listings on page", schema=JobListing, session=s).run()
        # result.output → validated JobListing instance
    """

    def __init__(self, task: str, *, schema: Optional[Type] = None, **kw):
        super().__init__(**kw)
        self._task = task
        self._schema = schema

    def task_prompt(self) -> str:
        prompt = (
            f"OBSERVE: {self._task}\n"
            "Extract the requested information from the current screen state. "
            "Do NOT perform any actions that change application state — "
            "only read, observe, and report back. "
            "Use read_pdf / read_file / read_csv for local files instead of opening them."
        )
        if self._schema is not None:
            prompt += (
                f"\nReturn the data matching this structure: {self._schema.__name__}."
            )
        return prompt

    def output_schema(self) -> Optional[Type]:
        return self._schema


class Check(BaseActionAgent):
    """Boolean query about screen state.

    Use ``.check()`` for a Python ``bool``, or ``.run()`` for a full
    :class:`RunResult`.

    Usage::

        if await Check("Place Order button is visible", session=s).check():
            await Do("click Place Order", session=s).run()
    """

    def __init__(self, condition: str, **kw):
        super().__init__(**kw)
        self._condition = condition

    def task_prompt(self) -> str:
        return (
            f"VERIFY: {self._condition}\n"
            "Check whether this condition is true on the current screen. "
            "Do NOT perform any actions — only observe. "
            "Respond with ONLY the word 'true' or 'false'."
        )

    async def check(self) -> bool:
        """Convenience: returns Python bool directly."""
        result = await self.run()
        return "true" in result.summary.lower()


class Navigate(BaseActionAgent):
    """Navigate to a URL or open an application.

    Usage::

        await Navigate("linkedin.com/jobs", session=s).run()
        await Navigate("Notepad", session=s).run()
    """

    def __init__(self, target: str, **kw):
        super().__init__(**kw)
        self._target = target

    def task_prompt(self) -> str:
        return (
            f"NAVIGATE: {self._target}\n"
            "Open this URL in the browser or launch this application. "
            "Wait until the target is loaded and ready, then stop immediately. "
            "Do not interact with any content — just get there."
        )


class Fill(BaseActionAgent):
    """Fill a form with provided data.

    Usage::

        await Fill("the job application form", data={"name": "Aadya", "email": "..."}, session=s).run()
    """

    def __init__(self, target: str, *, data: dict, **kw):
        super().__init__(**kw)
        self._target = target
        self._data = data

    def task_prompt(self) -> str:
        fields = "\n".join(f"  • {k}: {v}" for k, v in self._data.items())
        return (
            f"FILL: {self._target}\n"
            f"Enter these values into the form fields:\n{fields}\n"
            "Use fill_form_fields where possible for efficiency. "
            "Do NOT submit the form — only fill the fields."
        )
