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


def _default_drivers(browser: str = "chrome") -> Dict[str, Any]:
    """Build the built-in driver set for a session.

    Imports the optional drivers package lazily so that a build without
    it can still be used by passing drivers explicitly.

    Parameters
    ----------
    browser : str, optional
        Name of the browser the web driver should control. Default is
        ``"chrome"``.

    Returns
    -------
    Dict[str, Any]
        Mapping of driver name to driver instance.

    Raises
    ------
    ImportError
        If the default drivers package is not installed in this build.

    Examples
    --------
    >>> drivers = _default_drivers(browser="firefox")
    >>> sorted(drivers)  # doctest: +SKIP
    ['native', 'web']

    Notes
    -----
    The returned drivers are not started here. Drivers start lazily on
    first use so a native-only task never launches a browser and a
    web-only task never starts the accessibility daemon.
    """
    try:
        from .drivers import default_drivers
        return default_drivers(browser=browser)
    except ImportError as exc:
        raise ImportError(
            "Default Orbit drivers are not available in this build. "
            "Pass drivers explicitly: Session(drivers={'name': driver}). "
            f"(underlying error: {exc})"
        ) from exc


async def _call_if_present(obj: Any, *names: str) -> None:
    """Call the first method on ``obj`` whose name matches, awaiting if needed.

    Used to shut a driver down without caring whether its teardown hook
    is named ``stop`` or ``close``, or whether it is sync or async.

    Parameters
    ----------
    obj : Any
        Object to look up the method names on.
    *names : str
        Candidate method names, tried in order. The first one present is
        called and the search stops there.

    Returns
    -------
    None
        Nothing is returned; the call is made for its side effect.

    Examples
    --------
    >>> await _call_if_present(driver, "stop", "close")  # doctest: +SKIP

    Notes
    -----
    Only started backends are stopped, so callers filter the driver set
    before calling this helper.
    """
    for name in names:
        fn = getattr(obj, name, None)
        if fn is not None:
            res = fn()
            if hasattr(res, "__await__"):
                await res
            return


class Session:
    """An agent session: the public entry point to Orbit.

    A session owns a language model, a policy, a tool registry and a set
    of drivers, then exposes verb methods (:meth:`do`, :meth:`read`,
    :meth:`check`, :meth:`navigate`, :meth:`fill`) that each run one
    agent loop against them.

    Attributes
    ----------
    llm : LLM
        The model client used by every verb unless a verb is given its
        own ``llm``.
    policy : Policy
        Guardrails applied to actions the agent takes.
    max_steps : int
        Default cap on agent steps per run.
    _drivers : Optional[Dict[str, Any]]
        Driver instances by name, or ``None`` until defaults are built.
    _browser : str
        Browser name used when building default drivers.
    _tools : Dict[str, Tool]
        Tool registry the agent can call, by tool name.
    _runtime : Dict[str, Any]
        Runtime state (started backends, focused surface) shared across
        verbs for the lifetime of the session.

    Examples
    --------
    >>> async with orbit.session(llm="gemini/gemini-3.6-flash") as s:
    ...     result = await s.do("click the login button")

    Notes
    -----
    Drivers start lazily so a native-only task never launches a browser
    and a web-only task never starts the accessibility daemon. On exit
    only started backends are stopped. The session's LLM is injected
    into the vision rung, and runtime state (started backends, focused
    surface) is shared across verbs.
    """

    def __init__(
        self,
        llm: Union[str, LLM] = DEFAULT_MODEL,
        policy: Optional[Policy] = None,
        max_steps: int = 40,
        drivers: Optional[Dict[str, Any]] = None,
        tools: Optional[list] = None,
        include_default_tools: bool = True,
        browser: str = "chrome",
    ) -> None:
        """Create a session.

        Parameters
        ----------
        llm : Union[str, LLM], optional
            Model identifier to wrap in a ``LiteLLMClient``, or a ready
            ``LLM`` instance. Default is ``DEFAULT_MODEL``.
        policy : Optional[Policy], optional
            Guardrails for agent actions. Default is ``None``, which
            builds a default :class:`Policy`.
        max_steps : int, optional
            Default cap on agent steps per run. Default is 40.
        drivers : Optional[Dict[str, Any]], optional
            Explicit driver instances by name. Default is ``None``,
            which builds the default drivers on first use.
        tools : Optional[list], optional
            Extra tools to register, overriding defaults of the same
            name. Default is ``None``.
        include_default_tools : bool, optional
            Whether to include the standard tool set. Default is
            ``True``.
        browser : str, optional
            Browser used when building default drivers. Default is
            ``"chrome"``.

        Returns
        -------
        None
            Constructors return nothing.

        Examples
        --------
        >>> session = Session(llm="gemini/gemini-3.6-flash", max_steps=10)

        Notes
        -----
        Nothing is launched here. Drivers start lazily so a native-only
        task never launches a browser and a web-only task never starts
        the accessibility daemon, and runtime state (started backends,
        focused surface) is shared across verbs.
        """
        self.llm: LLM = LiteLLMClient(llm) if isinstance(llm, str) else llm
        self.policy = policy or Policy()
        self.max_steps = max_steps
        self._drivers = drivers
        self._browser = browser
        from .tools import build_registry
        self._tools = build_registry(tools, include_defaults=include_default_tools)
        # Shared across verbs: which backends are started + last surface
        # hint. Drivers start lazily (on first use) so a native-only task
        # never launches a browser, and a web-only task never starts the
        # accessibility daemon.
        self._runtime: Dict[str, Any] = {}

    async def __aenter__(self) -> "Session":
        """Enter the async context, preparing drivers.

        Builds the default drivers if none were supplied and injects the
        session's model into any driver that accepts one.

        Returns
        -------
        Session
            This same session, for use as the ``as`` target.

        Examples
        --------
        >>> async with orbit.session(llm="gemini/gemini-3.6-flash") as s:
        ...     result = await s.do("click the login button")

        Notes
        -----
        The session's LLM is injected into the vision rung so grounding
        uses the same model unless a driver already has one. Drivers are
        still not started here: they start lazily so a native-only task
        never launches a browser and a web-only task never starts the
        accessibility daemon.
        """
        if self._drivers is None:
            self._drivers = _default_drivers(browser=self._browser)
        # Vision grounds with the session's model unless it was given one.
        for driver in self._drivers.values():
            inject = getattr(driver, "set_llm", None)
            if inject is not None:
                inject(self.llm)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Exit the async context, stopping the drivers that ran.

        Parameters
        ----------
        *exc : Any
            Standard exception triple forwarded by the ``async with``
            statement. Ignored: exceptions are never suppressed.

        Returns
        -------
        None
            Nothing is returned; teardown happens as a side effect.

        Examples
        --------
        >>> async with orbit.session(llm="gemini/gemini-3.6-flash") as s:
        ...     await s.navigate("example.com")

        Notes
        -----
        Only started backends are stopped, which is what makes lazy
        driver start safe: a native-only task never launches a browser
        and a web-only task never starts the accessibility daemon, so
        neither is torn down.
        """
        # Only stop backends that were actually started.
        started = self._runtime.get("started", set())
        for name, driver in (self._drivers or {}).items():
            if name in started:
                await _call_if_present(driver, "stop", "close")

    # -- internals ---------------------------------------------------------
    def _world(self, max_steps: Optional[int]) -> World:
        """Build the :class:`World` a single run executes against.

        Parameters
        ----------
        max_steps : Optional[int]
            Per-run step cap. If ``None``, the session's ``max_steps``
            is used.

        Returns
        -------
        World
            World bound to this session's drivers, policy and runtime
            state.

        Examples
        --------
        >>> world = session._world(None)  # doctest: +SKIP

        Notes
        -----
        Default drivers are built here if they were not built on entry,
        keeping driver start lazy so a native-only task never launches a
        browser and a web-only task never starts the accessibility
        daemon. The same runtime mapping (started backends, focused
        surface) is passed to every world, so state is shared across
        verbs.
        """
        if self._drivers is None:
            self._drivers = _default_drivers(browser=self._browser)
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
        """Run one agent loop for an already-phrased task.

        Every verb funnels through here after prefixing the task with
        its intent.

        Parameters
        ----------
        task : str
            Fully phrased task text handed to the agent.
        llm : Optional[LLM], optional
            Model override for this run. Default is ``None``, which uses
            the session's model.
        max_steps : Optional[int], optional
            Step cap for this run. Default is ``None``, which uses the
            session's ``max_steps``.
        guidance : Optional[str], optional
            Extra instructions appended to the prompt. Default is
            ``None``.
        schema : Optional[Type[BaseModel]], optional
            Pydantic model the final output must validate against.
            Default is ``None``, meaning no structured output.
        timeout : Optional[float], optional
            Wall-clock limit in seconds. Default is ``None`` for no
            limit.

        Returns
        -------
        RunResult
            Status, output and trace for the run.

        Examples
        --------
        >>> run = await session._run("ACTION: click save")  # doctest: +SKIP

        Notes
        -----
        The world is rebuilt per run but shares the session's runtime
        state (started backends, focused surface) across verbs, and
        drivers still start lazily so a native-only task never launches
        a browser and a web-only task never starts the accessibility
        daemon.
        """
        return await loop.run(
            task=task,
            world=self._world(max_steps),
            llm=llm or self.llm,
            schema=schema,
            guidance=guidance,
            timeout=timeout,
            tools=self._tools,
        )

    # -- verbs -------------------------------------------------------------
    async def do(self, task: str, *, llm: Optional[LLM] = None,
                 max_steps: Optional[int] = None, guidance: Optional[str] = None,
                 timeout: Optional[float] = None) -> RunResult:
        """Perform an action on screen.

        The workhorse verb: the agent may click, type, scroll and use
        tools until the task is done or the step cap is reached.

        Parameters
        ----------
        task : str
            Plain-language description of what to do.
        llm : Optional[LLM], optional
            Model override for this run. Default is ``None``, which uses
            the session's model.
        max_steps : Optional[int], optional
            Step cap for this run. Default is ``None``, which uses the
            session's ``max_steps``.
        guidance : Optional[str], optional
            Extra instructions appended to the prompt. Default is
            ``None``.
        timeout : Optional[float], optional
            Wall-clock limit in seconds. Default is ``None`` for no
            limit.

        Returns
        -------
        RunResult
            Status, output and trace. Check ``status`` to see whether
            the action succeeded.

        Examples
        --------
        >>> async with orbit.session(llm="gemini/gemini-3.6-flash") as s:
        ...     result = await s.do("click the login button")

        Notes
        -----
        Whichever backend the task needs is started on first use, so a
        native-only task never launches a browser and a web-only task
        never starts the accessibility daemon. Started backends and the
        focused surface carry over to later verbs.
        """
        return await self._run(f"ACTION: {task}", llm=llm, max_steps=max_steps,
                               guidance=guidance, timeout=timeout)

    async def read(self, task: str, *, schema: Optional[Type[BaseModel]] = None,
                   llm: Optional[LLM] = None, max_steps: Optional[int] = None,
                   guidance: Optional[str] = None, timeout: Optional[float] = None) -> RunResult:
        """Observe the screen and extract information without changing it.

        The agent is told to observe only. With a ``schema`` the run must
        end in a value that validates against that model.

        Parameters
        ----------
        task : str
            Plain-language description of what to read.
        schema : Optional[Type[BaseModel]], optional
            Pydantic model the extracted data must validate against.
            Default is ``None``, which returns free text.
        llm : Optional[LLM], optional
            Model override for this run. Default is ``None``, which uses
            the session's model.
        max_steps : Optional[int], optional
            Step cap for this run. Default is ``None``, which uses the
            session's ``max_steps``.
        guidance : Optional[str], optional
            Extra instructions appended to the prompt. Default is
            ``None``.
        timeout : Optional[float], optional
            Wall-clock limit in seconds. Default is ``None`` for no
            limit.

        Returns
        -------
        RunResult
            Status, output and trace. When a ``schema`` is given, the
            output is a validated Pydantic instance of that model or the
            run failed. There is no half-valid middle state: if the
            status is not success, the output cannot be trusted and is
            typically ``None``.

        Examples
        --------
        >>> class Price(BaseModel):
        ...     total: float
        >>> async with orbit.session(llm="gemini/gemini-3.6-flash") as s:
        ...     run = await s.read("the cart total", schema=Price)
        ...     run.output.total
        49.99

        Notes
        -----
        Reading uses the same lazily started backends as the other
        verbs, so a native-only task never launches a browser and a
        web-only task never starts the accessibility daemon. Started
        backends and the focused surface are shared across verbs.
        """
        return await self._run(
            f"READ (observe only, do not change anything): {task}",
            llm=llm, max_steps=max_steps, guidance=guidance,
            schema=schema, timeout=timeout,
        )

    async def check(self, condition: str, *, llm: Optional[LLM] = None,
                    max_steps: Optional[int] = None, guidance: Optional[str] = None,
                    timeout: Optional[float] = None) -> bool:
        """Ask whether a condition currently holds on screen.

        Runs an observe-only agent loop constrained to a boolean schema
        and unwraps the answer, so callers can branch directly on the
        result instead of inspecting a run.

        Parameters
        ----------
        condition : str
            Plain-language statement to evaluate, phrased so that true
            or false is meaningful.
        llm : Optional[LLM], optional
            Model override for this run. Default is ``None``, which uses
            the session's model.
        max_steps : Optional[int], optional
            Step cap for this run. Default is ``None``, which uses the
            session's ``max_steps``.
        guidance : Optional[str], optional
            Extra instructions appended to the prompt. Default is
            ``None``.
        timeout : Optional[float], optional
            Wall-clock limit in seconds. Default is ``None`` for no
            limit.

        Returns
        -------
        bool
            A real ``bool``, not a run result and not a truthy object:
            ``True`` only when the run succeeded and the model judged
            the condition true. ``False`` when the model judged it
            false, and also ``False`` when the run failed (non-success
            status or missing output), so a failed check never reads as
            a passing one.

        Examples
        --------
        >>> async with orbit.session(llm="gemini/gemini-3.6-flash") as s:
        ...     if await s.check("the user is logged in"):
        ...         await s.do("open the settings page")

        Notes
        -----
        Because failure collapses to ``False``, treat a ``False`` result
        as "not confirmed" rather than "definitely false" when a run may
        have timed out. Backends still start lazily, so a native-only
        check never launches a browser and a web-only check never starts
        the accessibility daemon, and the focused surface is shared
        across verbs.
        """
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
        """Open a target and stop there.

        The agent is instructed to open the target and finish without
        further interaction, which makes this a cheap way to put the
        session on a known starting surface.

        Parameters
        ----------
        target : str
            What to open: a URL, a site name, or an application name.
        llm : Optional[LLM], optional
            Model override for this run. Default is ``None``, which uses
            the session's model.
        max_steps : Optional[int], optional
            Step cap for this run. Default is ``None``, which uses the
            session's ``max_steps``.
        guidance : Optional[str], optional
            Extra instructions appended to the prompt. Default is
            ``None``.
        timeout : Optional[float], optional
            Wall-clock limit in seconds. Default is ``None`` for no
            limit.

        Returns
        -------
        RunResult
            Status, output and trace for the navigation run.

        Examples
        --------
        >>> async with orbit.session(llm="gemini/gemini-3.6-flash") as s:
        ...     await s.navigate("https://example.com")
        ...     result = await s.do("click the login button")

        Notes
        -----
        This is usually the call that starts a backend, since drivers
        start lazily: a native-only task never launches a browser and a
        web-only task never starts the accessibility daemon. The surface
        it lands on is recorded in runtime state shared across verbs, so
        a following :meth:`do` continues where this left off.
        """
        return await self._run(
            f"NAVIGATE: open {target}, then finish. No further interaction.",
            llm=llm, max_steps=max_steps, guidance=guidance, timeout=timeout,
        )

    async def fill(self, form_name: str, data: Dict[str, Any], *, llm: Optional[LLM] = None,
                   max_steps: Optional[int] = None, guidance: Optional[str] = None,
                   timeout: Optional[float] = None) -> RunResult:
        """Fill a named form with the given field values.

        The values are serialised to JSON and handed to the agent, which
        matches them to fields on screen and then finishes. Submitting
        is not implied: ask for it with :meth:`do` or ``guidance``.

        Parameters
        ----------
        form_name : str
            Human name of the form to fill, as it appears on screen.
        data : Dict[str, Any]
            Field name to value mapping. Values that are not JSON
            serialisable are rendered with ``str``.
        llm : Optional[LLM], optional
            Model override for this run. Default is ``None``, which uses
            the session's model.
        max_steps : Optional[int], optional
            Step cap for this run. Default is ``None``, which uses the
            session's ``max_steps``.
        guidance : Optional[str], optional
            Extra instructions appended to the prompt. Default is
            ``None``.
        timeout : Optional[float], optional
            Wall-clock limit in seconds. Default is ``None`` for no
            limit.

        Returns
        -------
        RunResult
            Status, output and trace for the fill run.

        Examples
        --------
        >>> async with orbit.session(llm="gemini/gemini-3.6-flash") as s:
        ...     await s.fill("Sign up", {"email": "a@b.com", "name": "Ada"})
        ...     await s.do("click submit")

        Notes
        -----
        Field names are matched by the model, so they need to resemble
        the visible labels rather than the underlying markup. The form
        is filled on whichever surface the session is focused on, since
        runtime state (started backends, focused surface) is shared
        across verbs and drivers start lazily.
        """
        return await self._run(
            f"FILL the form {form_name!r} with these values, then finish:\n"
            f"{json.dumps(data, default=str, indent=2)}",
            llm=llm, max_steps=max_steps, guidance=guidance, timeout=timeout,
        )
