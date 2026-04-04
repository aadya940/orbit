"""Session owns the OculOS daemon and shared ADK session state."""

import asyncio
import logging
from typing import Optional

from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService

from .daemon import OculOSManager
from ._ui.toast import run_toast_ui

log = logging.getLogger("orbit.session")


class Session:
    """Async context manager that owns a single OculOS daemon and ADK session.

    When multiple verbs share a session, they share the same ADK conversation
    so the planner retains context across calls.

    Usage::

        async with Session() as s:
            await Do("open Notepad", session=s).run()
            await Do("type hello", session=s).run()  # planner knows Notepad is open
    """

    def __init__(self):
        self._daemon = OculOSManager()
        self._started = False
        self._session_service = InMemorySessionService()
        self._artifact_service = InMemoryArtifactService()
        self._adk_session = None

    async def __aenter__(self) -> "Session":
        await self._daemon.start()
        self._started = True
        return self

    async def __aexit__(self, *exc):
        try:
            self._daemon.stop()
        except Exception:
            log.debug("Daemon shutdown failed; continuing.", exc_info=True)
        self._started = False
        # Show a completion toast only when the session exits cleanly after running tasks.
        if exc and exc[0] is None and self._adk_session is not None:
            try:
                loop = asyncio.get_running_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        run_toast_ui,
                        "completion",
                        {
                            "description": (
                                "Orbit has finished all tasks. "
                                "You can use your screen now."
                            )
                        },
                    ),
                    timeout=5,
                )
            except Exception:
                log.debug(
                    "Completion toast failed; continuing shutdown.", exc_info=True
                )

    @property
    def started(self) -> bool:
        return self._started

    @property
    def session_service(self) -> InMemorySessionService:
        return self._session_service

    @property
    def artifact_service(self) -> InMemoryArtifactService:
        return self._artifact_service

    @property
    def adk_session(self):
        return self._adk_session

    @adk_session.setter
    def adk_session(self, value):
        self._adk_session = value


def session() -> Session:
    """Factory for ``async with orbit.session() as s:``"""
    return Session()
