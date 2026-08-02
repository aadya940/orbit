"""Tests for orbit2.drivers: lazy-import discipline, protocol conformance,
and the pure fuzzy target matcher."""

from __future__ import annotations

import subprocess
import sys

import pytest

from orbit2.drivers.matching import (
    best_match,
    parse_description,
    rank_matches,
    score_element,
    suggestions,
)
from orbit2.types import Element


# ---------------------------------------------------------------------------
# Lazy-import discipline
# ---------------------------------------------------------------------------

_LAZY_IMPORT_SCRIPT = r"""
import sys

BLOCKED = ("playwright", "patchright", "requests", "pyautogui", "PIL")

class _Blocker:
    def find_module(self, name, path=None):
        return self if name.split(".")[0] in BLOCKED else None
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked optional dep: {name}")
        return None

for m in list(sys.modules):
    if m.split(".")[0] in BLOCKED:
        del sys.modules[m]
sys.meta_path.insert(0, _Blocker())

import orbit2.drivers as d
drv = d.default_drivers()
assert set(drv) == {"dom", "tree", "keyboard", "vision"}, drv
# Accessing lazy exports must also work without optional deps installed.
for cls_name in ("DomDriver", "AccessibilityDriver", "KeyboardDriver", "VisionDriver"):
    getattr(d, cls_name)
print("OK")
"""


def test_import_without_optional_deps():
    """orbit2.drivers imports and instantiates with zero optional deps."""
    proc = subprocess.run(
        [sys.executable, "-c", _LAZY_IMPORT_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_default_drivers_returns_four():
    from orbit2.drivers import default_drivers

    drivers = default_drivers()
    assert set(drivers) == {"dom", "tree", "keyboard", "vision"}
    for name, driver in drivers.items():
        assert driver.name == name


def test_drivers_satisfy_protocol():
    from orbit2.drivers import Driver, default_drivers

    for driver in default_drivers().values():
        assert isinstance(driver, Driver), driver


# ---------------------------------------------------------------------------
# Fuzzy matcher (pure, no I/O)
# ---------------------------------------------------------------------------

def _el(name: str, role: str = "generic", enabled: bool = True, value=None) -> Element:
    return Element(role=role, name=name, enabled=enabled, value=value)


def test_exact_name_beats_partial():
    els = [_el("Login now"), _el("Login"), _el("Log")]
    ranked = rank_matches(els, "Login")
    assert ranked[0].element.name == "Login"


def test_exact_match_case_insensitive():
    els = [_el("SUBMIT"), _el("Submit application")]
    assert best_match(els, "submit").name == "SUBMIT"


def test_partial_containment_matches():
    els = [_el("Easy Apply to this job")]
    assert best_match(els, "Easy Apply") is not None


def test_role_hint_prefers_matching_role():
    els = [_el("Login", role="link"), _el("Login", role="button")]
    ranked = rank_matches(els, "the login button")
    assert ranked[0].element.role == "button"


def test_role_hint_field_prefers_textbox():
    els = [_el("Email", role="button"), _el("Email", role="textbox")]
    assert best_match(els, "Email field").role == "textbox"


def test_role_hint_tree_style_roles():
    # OS accessibility trees use roles like "Edit" and "Push Button".
    els = [_el("Search", role="push button"), _el("Search", role="edit")]
    assert best_match(els, "search input").role == "edit"


def test_no_match_returns_empty():
    els = [_el("Cancel"), _el("Home"), _el("Settings")]
    assert rank_matches(els, "purchase warranty") == []
    assert best_match(els, "purchase warranty") is None


def test_empty_description_no_match():
    assert rank_matches([_el("Anything")], "") == []


def test_empty_elements_no_match():
    assert rank_matches([], "Login") == []


def test_stopwords_and_quotes_stripped():
    els = [_el("Email address")]
    assert best_match(els, '"the Email address"') is not None
    name, hint = parse_description("the login button")
    assert name == "login"
    assert hint is not None and "button" in hint


def test_token_overlap_partial_words():
    els = [_el("Submit your application now")]
    m = best_match(els, "submit application")
    assert m is not None


def test_disabled_penalized_over_enabled():
    els = [_el("Next", enabled=False), _el("Next", enabled=True)]
    ranked = rank_matches(els, "Next")
    assert ranked[0].element.enabled is True


def test_value_match_scores():
    els = [_el("", role="textbox", value="alex@example.com")]
    assert score_element(els[0], "alex@example.com") > 0


def test_ranked_order_descending():
    els = [_el("Apply"), _el("Apply now"), _el("Applying tips article")]
    ranked = rank_matches(els, "Apply")
    scores = [m.score for m in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0].element.name == "Apply"


def test_suggestions_dedupe_and_limit():
    els = [_el("A"), _el("A"), _el("B"), _el(""), _el("C"), _el("D"), _el("E"), _el("F")]
    s = suggestions(els)
    assert s == ["A", "B", "C", "D", "E"]


def test_pure_role_query_matches_role_only():
    els = [_el("OK", role="button"), _el("Story", role="article")]
    m = best_match(els, "the button")
    assert m is not None and m.role == "button"
