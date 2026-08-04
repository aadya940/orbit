"""VisionDriver: pure parsing/geometry (no screen, no model needed)."""

import pytest

from orbit.drivers.vision import (
    VisionDriver,
    box_to_bounds,
    elements_from_grounding,
    parse_grounding,
)
from orbit.types import Action, ActionKind, Source, SurfaceUnreadable, TargetNotFound


# -- parse_grounding ---------------------------------------------------------

def test_parses_plain_json():
    els, title, _ = parse_grounding('{"elements": [{"role": "button", "name": "OK"}], "title": "Dialog"}')
    assert len(els) == 1 and els[0]["name"] == "OK"
    assert title == "Dialog"


def test_parses_fenced_json():
    raw = 'Here is what I see:\n```json\n{"elements": [{"role": "link", "name": "Home"}]}\n```'
    els, _, _ = parse_grounding(raw)
    assert len(els) == 1 and els[0]["role"] == "link"


def test_parses_json_with_leading_prose():
    raw = 'I can see a login screen. {"elements": [{"role": "textbox", "name": "Email"}]}'
    els, _, _ = parse_grounding(raw)
    assert len(els) == 1 and els[0]["name"] == "Email"


@pytest.mark.parametrize("raw", ["", "no json here", "{broken", '{"elements": "notalist"}'])
def test_bad_grounding_yields_nothing(raw):
    els, title, _ = parse_grounding(raw)
    assert els == [] and title == ""


# -- box_to_bounds -----------------------------------------------------------

def test_box_scales_to_pixels():
    # [ymin, xmin, ymax, xmax] normalized 0-1000 on a 1000x500 screen
    b = box_to_bounds([100, 200, 300, 600], 1000, 500)
    assert b is not None
    assert b.x == pytest.approx(200.0)     # 200/1000 * 1000
    assert b.y == pytest.approx(50.0)      # 100/1000 * 500
    assert b.width == pytest.approx(400.0)
    assert b.height == pytest.approx(100.0)
    assert b.center == pytest.approx((400.0, 100.0))


@pytest.mark.parametrize("box", [None, [1, 2, 3], "nope", [1, 2, "x", 4], [500, 500, 100, 100]])
def test_bad_boxes_rejected(box):
    b = box_to_bounds(box, 800, 600)
    assert b is None or b.width < 0 or b.height < 0


# -- elements_from_grounding -------------------------------------------------

def test_builds_elements_with_vision_provenance():
    raw = [
        {"role": "button", "name": "Save", "box": [0, 0, 100, 100], "enabled": True},
        {"role": "button", "name": "Greyed", "box": [200, 0, 300, 100], "enabled": False},
    ]
    els = elements_from_grounding(raw, 1000, 1000)
    assert len(els) == 2
    assert els[0].provenance == frozenset({Source.VISION})
    assert els[0].confidence == 0.5          # single source -> suspect
    assert els[1].enabled is False
    # ref carries click coordinates
    assert els[0].ref["cx"] == pytest.approx(50.0)


def test_elements_without_geometry_dropped():
    raw = [
        {"role": "button", "name": "Good", "box": [0, 0, 50, 50]},
        {"role": "button", "name": "NoBox"},
        {"role": "button", "name": "BadBox", "box": [1, 2]},
    ]
    els = elements_from_grounding(raw, 500, 500)
    assert [e.name for e in els] == ["Good"]


def test_offscreen_elements_dropped():
    # center outside the surface rect -> coordinate insanity, dropped
    raw = [{"role": "button", "name": "Offscreen", "box": [1800, 1800, 1900, 1900]}]
    assert elements_from_grounding(raw, 800, 600) == []


# -- driver behavior ---------------------------------------------------------

async def test_observe_without_model_raises_surface_unreadable():
    d = VisionDriver()
    with pytest.raises((SurfaceUnreadable, TargetNotFound)):
        await d.observe()


async def test_ref_out_of_range_is_typed_error():
    d = VisionDriver()
    d._last_elements = elements_from_grounding(
        [{"role": "button", "name": "A", "box": [0, 0, 10, 10]}], 100, 100
    )
    with pytest.raises(TargetNotFound):
        d._resolve(Action(kind=ActionKind.CLICK, ref=99))


def test_ref_resolves_to_shown_element():
    d = VisionDriver()
    d._last_elements = elements_from_grounding([
        {"role": "button", "name": "First", "box": [0, 0, 10, 10]},
        {"role": "button", "name": "Second", "box": [20, 20, 30, 30]},
    ], 100, 100)
    el = d._resolve(Action(kind=ActionKind.CLICK, ref=1))
    assert el.name == "Second"


def test_vision_is_surface_agnostic():
    # Must ride along as a fallback rung on any surface.
    assert VisionDriver.surface is None
