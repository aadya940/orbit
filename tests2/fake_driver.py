"""Scripted fakes: FakeScreen + FakeDriver + FakeLLM."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from orbit2.llm import LLMReply
from orbit2.types import Action, Element, Observation, OrbitError

Behavior = Union[str, OrbitError]  # "ok" | "noop" | error to raise


class FakeScreen:
    """Shared observation sequence so multiple drivers see one 'world'."""

    def __init__(self, observations: List[Observation]) -> None:
        assert observations
        self.observations = observations
        self.idx = 0

    def current(self) -> Observation:
        return self.observations[min(self.idx, len(self.observations) - 1)]

    def advance(self) -> None:
        self.idx = min(self.idx + 1, len(self.observations) - 1)


class FakeDriver:
    """Driver scripted by a behavior map.

    behavior maps (kind_value, target) or kind_value to:
      "ok"   -> succeed and advance the screen to the next observation
      "noop" -> report success but do NOT advance (stale-tree click)
      OrbitError instance -> raise it
    Missing key defaults to "ok".
    """

    def __init__(
        self,
        observations_or_screen: Union[List[Observation], FakeScreen],
        name: str = "fake",
        behavior: Optional[Dict[Union[str, Tuple[str, Optional[str]]], Behavior]] = None,
    ) -> None:
        self.screen = (
            observations_or_screen
            if isinstance(observations_or_screen, FakeScreen)
            else FakeScreen(observations_or_screen)
        )
        self.name = name
        self.behavior = behavior or {}
        self.acted: List[Action] = []
        self.observe_count = 0

    def _lookup(self, action: Action) -> Behavior:
        for key in ((action.kind.value, action.target), action.kind.value):
            if key in self.behavior:
                return self.behavior[key]
        return "ok"

    async def observe(self) -> Observation:
        self.observe_count += 1
        return self.screen.current()

    async def act(self, action: Action) -> Optional[Element]:
        self.acted.append(action)
        b = self._lookup(action)
        if isinstance(b, OrbitError):
            raise b
        if b == "ok":
            self.screen.advance()
        # "noop": succeed without advancing
        el = None
        if action.target:
            el = Element(role="button", name=action.target)
        return el

    async def screenshot(self) -> Optional[bytes]:
        return None


class FakeLLM:
    """LLM scripted from a list of LLMReply (last reply repeats)."""

    def __init__(self, replies: List[LLMReply]) -> None:
        self.replies = replies
        self.idx = 0
        self.calls: List[List[dict]] = []

    async def complete(self, messages: List[dict], tools: List[dict]) -> LLMReply:
        self.calls.append(list(messages))
        reply = self.replies[min(self.idx, len(self.replies) - 1)]
        self.idx += 1
        return reply
