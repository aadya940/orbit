"""Pure fuzzy target matching shared by all drivers.

Given a list of :class:`~orbit2.types.Element` and a natural-language
target description ("the login button", "Email address field"), return
ranked matches. No I/O — fully unit-testable.

Heuristics (transplanted from the v1 smart_dom_tools / ui.py matchers):
- exact name match beats partial containment beats token overlap
- role words embedded in the description ("button", "link", "field", ...)
  act as a role hint: matching roles get a bonus, and the role word is
  stripped from the name comparison
- disabled elements are slightly penalized (still returned — the caller
  decides whether to raise TargetObstructed)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..types import Element

# Words in a description that hint at the element's role. Values are the
# element roles (lower-cased) they should boost, covering both DOM roles
# and OS accessibility roles (OculOS/UIA/AT-SPI naming).
_ROLE_HINTS: Dict[str, Tuple[str, ...]] = {
    "button": ("button", "push button", "pushbutton"),
    "link": ("link", "a", "hyperlink"),
    "tab": ("tab", "page tab", "tabitem"),
    "checkbox": ("checkbox", "check box"),
    "radio": ("radio", "radio button", "radiobutton"),
    "field": ("textbox", "edit", "text", "input", "textarea", "searchbox"),
    "input": ("textbox", "edit", "text", "input", "textarea", "searchbox"),
    "textbox": ("textbox", "edit", "text", "input", "textarea", "searchbox"),
    "dropdown": ("combobox", "select", "listbox", "combo box"),
    "combobox": ("combobox", "select", "listbox", "combo box"),
    "select": ("combobox", "select", "listbox", "combo box"),
    "menu": ("menu", "menuitem", "menu item"),
    "option": ("option", "listitem", "menuitem"),
    "dialog": ("dialog", "alertdialog", "window"),
    "heading": ("heading", "h1", "h2", "h3"),
    "slider": ("slider", "range"),
}

_STOPWORDS = frozenset({"the", "a", "an", "on", "in", "of", "for", "to", "with"})


@dataclass
class Match:
    element: Element
    score: float


def _normalize(text: str) -> str:
    """Lower-case, strip wrapping quotes/backticks and collapse whitespace."""
    t = text.strip().strip('"').strip("'").strip("`").strip()
    return re.sub(r"\s+", " ", t).lower()


def parse_description(description: str) -> Tuple[str, Optional[Tuple[str, ...]]]:
    """Split a description into (name text, role hint roles).

    "the login button" -> ("login", ("button", ...))
    "Email address"    -> ("email address", None)
    """
    words = [w for w in _normalize(description).split(" ") if w]
    hint: Optional[Tuple[str, ...]] = None
    kept: List[str] = []
    for w in words:
        if w in _ROLE_HINTS and hint is None:
            hint = _ROLE_HINTS[w]
            continue
        if w in _STOPWORDS:
            continue
        kept.append(w)
    return " ".join(kept), hint


def _token_overlap(a: str, b: str) -> float:
    ta = {w for w in a.split(" ") if w and w not in _STOPWORDS}
    tb = {w for w in b.split(" ") if w and w not in _STOPWORDS}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


# Generic UI-symbol vocabulary: controls labeled with a bare glyph that
# users (and models) describe in words. Desktop-wide, not app-specific.
_WORD_TO_SYMBOL = {
    "equals": ("=",),
    "plus": ("+",), "add": ("+",),
    "minus": ("−", "-"), "subtract": ("−", "-"),
    # NB: deliberately no ASCII "x" alias — calculators have an algebra
    # variable button named 'x' (verified live on GNOME Calculator).
    "multiply": ("×", "*"), "times": ("×", "*"),
    "divide": ("÷", "/"),
    "percent": ("%",),
    "close": ("✕",),
    "search": ("🔍",),
    "settings": ("⚙", "⚙️"),
    "menu": ("☰",),
}


def score_element(element: Element, description: str) -> float:
    """Score one element against a description. 0.0 = no match."""
    needle, role_hint = parse_description(description)
    if not needle and not role_hint:
        return 0.0

    name = _normalize(element.name or "")
    value = _normalize(element.value or "")
    role = _normalize(element.role or "")
    hint = _normalize(element.hint or "")

    raw_name = (element.name or "").strip()
    score = 0.0
    if needle:
        if name == needle:
            score = 1.0
        elif raw_name and raw_name in _WORD_TO_SYMBOL.get(needle, ()):
            # Word-described symbol keys: "equals" -> '=', "multiply" -> '×'.
            score = 1.0
        elif name and needle in name:
            score = 0.7
        # name-in-needle only for real words: 'c' ⊂ 'calculate result'
        # must NOT match the C button (verified live on a calculator).
        elif len(name) >= 3 and name in needle:
            score = 0.7
        elif hint and (needle in hint or _token_overlap(hint, needle) >= 0.5):
            # Symbol-labeled controls ('×') carry their meaning in the
            # hint ("Multiply [*]") — often the only word-based signal.
            score = 0.65
        elif value and needle in value:
            score = 0.5
        else:
            score = 0.6 * _token_overlap(name, needle)
    elif role_hint:
        # Pure role query like "the button" — weak base score.
        score = 0.2

    if score <= 0.0:
        return 0.0

    if role_hint:
        if any(role == r or r in role for r in role_hint):
            score += 0.15
        else:
            score -= 0.15

    if not element.enabled:
        score -= 0.05

    return max(0.0, min(score, 1.15))


def rank_matches(
    elements: Sequence[Element],
    description: str,
    min_score: float = 0.3,
) -> List[Match]:
    """Rank elements against a description, best first.

    Returns only matches scoring >= min_score; empty list if nothing
    plausibly matches.
    """
    scored = [Match(el, score_element(el, description)) for el in elements]
    scored = [m for m in scored if m.score >= min_score]
    scored.sort(key=lambda m: m.score, reverse=True)
    return scored


def best_match(
    elements: Sequence[Element],
    description: str,
    min_score: float = 0.3,
) -> Optional[Element]:
    matches = rank_matches(elements, description, min_score=min_score)
    return matches[0].element if matches else None


def suggestions(elements: Sequence[Element], limit: int = 5) -> List[str]:
    """Closest-label suggestions for TargetNotFound context."""
    names = []
    for el in elements:
        if el.name and el.name.strip() and el.name not in names:
            names.append(el.name.strip()[:60])
    return names[:limit]
