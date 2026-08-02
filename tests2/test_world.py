import pytest

from conftest import make_obs
from fake_driver import FakeDriver, FakeScreen
from orbit2.types import Action, ActionKind, BudgetExhausted, Observation, Element
from orbit2.world import World


def simple_world(max_steps=5):
    screen = FakeScreen([make_obs("Save"), make_obs("Save", "Done")])
    d = FakeDriver(screen, name="dom")
    return World(drivers={"dom": d}, max_steps=max_steps), d


def test_budget_spend_and_exhaustion():
    world, _ = simple_world(max_steps=2)
    world.spend()
    assert world.remaining == 1
    world.spend()
    assert world.remaining == 0
    with pytest.raises(BudgetExhausted):
        world.spend()
    assert any(e["kind"] == "budget_exhausted" for e in world.journal.to_list())


async def test_act_journals():
    world, _ = simple_world()
    result = await world.act(Action(kind=ActionKind.CLICK, target="Save"))
    assert result.landed
    entries = [e for e in world.journal.to_list() if e["kind"] == "action"]
    assert len(entries) == 1
    assert entries[0]["target"] == "Save" and entries[0]["strategy"] == "dom"


async def test_probe_reorders_by_score():
    rich = make_obs("Save", "Cancel", "Email", "Name", "Submit")
    poor = Observation(surface="s", kind="browser",
                       elements=[Element(role="button", name="")])
    d_tree = FakeDriver([poor], name="tree")
    d_dom = FakeDriver([rich], name="dom")
    world = World(drivers={"tree": d_tree, "dom": d_dom}, primary="dom")
    order = await world.probe("s")
    assert order[0] == "dom"
    assert world.ladder_order == order
    assert "s" in world.probe_cache


def test_worlds_are_isolated():
    w1, _ = simple_world()
    w2, _ = simple_world()
    w1.spend()
    w1.journal.append("x")
    assert w2.used == 0
    assert w2.journal.to_list() == []
    assert w1.drivers is not w2.drivers
