import pytest

from conftest import make_obs
from fake_driver import FakeDriver, FakeScreen
from orbit.drivers.base import run_ladder
from orbit.types import Action, ActionKind, TargetNotFound, TargetUnresolvable

CLICK = Action(kind=ActionKind.CLICK, target="Save")


def two_state_screen():
    return FakeScreen([make_obs("Save"), make_obs("Save", "Done")])


async def test_first_driver_lands():
    screen = two_state_screen()
    d1 = FakeDriver(screen, name="tree")
    d2 = FakeDriver(screen, name="dom")
    out = await run_ladder([d1, d2], CLICK, observe_via=d1)
    assert out.result.landed
    assert out.result.strategy == "tree"
    assert out.result.attempts == 1
    assert not d2.acted


async def test_fallback_on_target_not_found():
    screen = two_state_screen()
    d1 = FakeDriver(screen, name="tree", behavior={"click": TargetNotFound("nope")})
    d2 = FakeDriver(screen, name="dom")
    out = await run_ladder([d1, d2], CLICK, observe_via=d2)
    assert out.result.strategy == "dom"
    assert out.result.attempts == 2
    assert len(out.errors) == 1


async def test_noop_escalates():
    screen = two_state_screen()
    d1 = FakeDriver(screen, name="tree", behavior={"click": "noop"})
    d2 = FakeDriver(screen, name="dom")
    out = await run_ladder([d1, d2], CLICK, observe_via=d1)
    assert out.result.strategy == "dom"
    assert out.result.attempts == 2
    assert any("no observable change" in str(e) for e in out.errors)


async def test_all_fail_raises_unresolvable():
    screen = two_state_screen()
    d1 = FakeDriver(screen, name="tree", behavior={"click": TargetNotFound("a")})
    d2 = FakeDriver(screen, name="dom", behavior={"click": "noop"})
    with pytest.raises(TargetUnresolvable) as ei:
        await run_ladder([d1, d2], CLICK, observe_via=d1)
    assert ei.value.context["attempts"] == 2


async def test_no_effect_expected_still_lands():
    screen = FakeScreen([make_obs("Save")])  # single state: nothing ever changes
    d1 = FakeDriver(screen, name="keyboard", behavior={"press": "noop"})
    action = Action(kind=ActionKind.PRESS, value="ctrl+c", expects_effect=False)
    out = await run_ladder([d1], action, observe_via=d1)
    assert out.result.landed
    assert not out.result.diff.changed
