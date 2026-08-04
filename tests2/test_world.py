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


class _BlindDriver(FakeDriver):
    """Perception-incapable driver (like keyboard/vision): observe raises."""
    async def observe(self):
        from orbit2.types import SurfaceUnreadable
        raise SurfaceUnreadable(f"{self.name} cannot perceive")


async def test_route_falls_through_unusable_driver():
    # dom (tried first) sees a blank/poor surface -> unusable; routing falls
    # through to the tree driver that actually sees content.
    poor = Observation(surface="browser", kind="browser",
                       elements=[Element(role="generic", name="")])
    rich = make_obs("7", "8", "×", "=", "C", "Clear")
    world = World(drivers={
        "dom": FakeDriver([poor], name="dom"),
        "tree": FakeDriver([rich], name="tree"),
        "keyboard": _BlindDriver([poor], name="keyboard"),
    })  # no primary -> auto_route
    world._started.update({"dom", "tree"})  # both running -> pure capability probe
    obs = await world.observe()
    assert world.primary == "tree"          # dom unusable -> tree
    assert obs.elements[0].name == "7"
    assert world.ladder_order[0] == "tree"  # ladder reordered behind it


def test_navigator_dispatch_by_capability():
    # Drivers own "can I open this?"; World asks, first yes wins.
    world = World(drivers={
        "dom": FakeDriver([make_obs("x")], name="dom"),
        "tree": FakeDriver([make_obs("y")], name="tree"),
    })
    from orbit2.drivers.base import is_web_target

    class _NavDom(FakeDriver):
        @staticmethod
        def can_navigate(t): return is_web_target(t)

    class _NavTree(FakeDriver):
        @staticmethod
        def can_navigate(t): return bool(t) and not is_web_target(t)

    world.drivers["dom"] = _NavDom([make_obs("x")], name="dom")
    world.drivers["tree"] = _NavTree([make_obs("y")], name="tree")
    assert world._pick_navigator("https://example.com") == "dom"
    assert world._pick_navigator("gnome-calculator") == "tree"
    assert world._pick_navigator("localhost:3000") == "dom"


async def test_explicit_primary_disables_auto_route():
    rich = make_obs("A", "B", "C", "D", "E")
    world = World(drivers={
        "dom": FakeDriver([make_obs("only")], name="dom"),
        "tree": FakeDriver([rich], name="tree"),
    }, primary="dom")
    assert world.auto_route is False
    await world.observe()
    assert world.primary == "dom"           # respected, never re-routed


async def test_navigate_then_route():
    # Before opening anything, observe yields an empty surface (no probing).
    # A navigate opens the app and defers routing to the next observe.
    window = make_obs("File", "Edit", "View", "Save", "Open")
    world = World(drivers={
        "tree": FakeDriver(FakeScreen([window, window]), name="tree"),
    })
    obs = await world.observe()
    assert obs.kind == "none"               # nothing open yet, no probe-launch
    await world.act(Action(kind=ActionKind.NAVIGATE, value="app:editor"))
    assert "tree" in world._started          # app launched via the tree driver
    assert world._routed is False           # routing deferred to next observe
    obs = await world.observe()
    assert world._routed is True            # now routed against the real app
    assert world.primary == "tree"


def test_worlds_are_isolated():
    w1, _ = simple_world()
    w2, _ = simple_world()
    w1.spend()
    w1.journal.append("x")
    assert w2.used == 0
    assert w2.journal.to_list() == []
    assert w1.drivers is not w2.drivers
