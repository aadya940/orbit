"""Pure fuzzy target matching shared by all drivers.

Given a list of :class:`~orbit.types.Element` and a natural-language
target description ("the login button", "Email address field"), return
ranked matches. There is no I/O here, so the module is fully unit
testable.

Notes
-----
Heuristics transplanted from the v1 ``smart_dom_tools`` and ``ui.py``
matchers:

* An exact name match beats partial containment, which beats token
  overlap.
* Role words embedded in the description ("button", "link", "field")
  act as a role hint. Matching roles get a bonus and the role word is
  stripped from the name comparison.
* Disabled elements are slightly penalized but still returned. The
  caller decides whether to raise TargetObstructed, since the matcher
  should not conflate "not found" with "found but unavailable".
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
    """One scored candidate element.

    Attributes
    ----------
    element : Element
        The candidate element.
    score : float
        Its match score against the description, higher is better.
    """

    element: Element
    score: float


def _normalize(text: str) -> str:
    """Lower-case, strip wrapping quotes and backticks, collapse whitespace.

    Parameters
    ----------
    text : str
        Raw text to normalize.

    Returns
    -------
    str
        The normalized comparison form.
    """
    t = text.strip().strip('"').strip("'").strip("`").strip()
    return re.sub(r"\s+", " ", t).lower()


def parse_description(description: str) -> Tuple[str, Optional[Tuple[str, ...]]]:
    """Split a description into its name text and any role hint.

    Parameters
    ----------
    description : str
        Natural-language target description.

    Returns
    -------
    Tuple[str, Optional[Tuple[str, ...]]]
        The remaining name text, and the tuple of roles the description
        hinted at, or None if no role word was present.

    Notes
    -----
    Only the first role word encountered is consumed as a hint. Later
    ones stay in the name text, since a description like "open menu
    button" should still compare against the word "menu". Stopwords are
    dropped entirely.

    Examples
    --------
    "the login button" becomes ("login", ("button", ...)) and
    "Email address" becomes ("email address", None).
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
    """Measure shared-word overlap between two normalized strings.

    Parameters
    ----------
    a : str
        First normalized string.
    b : str
        Second normalized string.

    Returns
    -------
    float
        Count of shared non-stopword tokens divided by the larger token
        set size, in the range 0.0 to 1.0. Returns 0.0 if either side
        has no usable tokens.

    Notes
    -----
    Dividing by the larger set rather than the union penalizes a short
    query that happens to be a subset of a very long label, which keeps
    generic labels from absorbing every query.
    """
    ta = {w for w in a.split(" ") if w and w not in _STOPWORDS}
    tb = {w for w in b.split(" ") if w and w not in _STOPWORDS}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def score_element(element: Element, description: str) -> float:
    """Score one element against a description.

    Parameters
    ----------
    element : Element
        The candidate element to score.
    description : str
        Natural-language target description.

    Returns
    -------
    float
        Match strength, where 0.0 means no match. Scores are clamped to
        a maximum of 1.15 so that a role bonus can push a perfect name
        match above other exact matches without unbounded growth.

    Notes
    -----
    A quoted span in the description is treated as the element's literal
    name and wins before any fuzzy heuristic. The model quotes what it
    sees in the observation, often echoing the render format verbatim,
    so this is the strongest available signal and it also covers symbol
    names that normalization would otherwise strip.
    """
    needle, role_hint = parse_description(description)
    if not needle and not role_hint and not description.strip():
        return 0.0

    name = _normalize(element.name or "")
    value = _normalize(element.value or "")
    role = _normalize(element.role or "")
    hint = _normalize(element.hint or "")

    # The model quotes what it sees in the observation, often echoing the
    # render format verbatim: `button '7' (Clear Display [Escape])`.
    # A quoted span is therefore the strongest signal (the element's
    # literal name) and must win before any fuzzy heuristics. This also
    # covers symbol names ('×', '=') that normalization would strip.
    raw_name = (element.name or "").strip()
    quoted = re.search(r"['\"]([^'\"]+)['\"]", description)
    raw_needle = (quoted.group(1) if quoted else description).strip().strip("'\"")
    score = 0.0
    if needle or raw_needle:
        if raw_name and raw_name == raw_needle:
            score = 1.0
        elif raw_name and raw_name.lower() == raw_needle.lower():
            score = 0.95
        elif name and name == needle:
            score = 1.0
        elif name and needle in name:
            score = 0.7
        # name-in-needle only for real words: 'c' ⊂ 'calculate result'
        # must NOT match the C button (verified live on a calculator).
        elif len(name) >= 3 and name in needle:
            score = 0.7
        elif hint and (needle in hint or _token_overlap(hint, needle) >= 0.5):
            # Symbol-labeled controls ('×') carry their meaning in the
            # hint ("Multiply [*]"), often the only word-based signal.
            score = 0.65
        elif value and needle in value:
            score = 0.5
        else:
            score = 0.6 * _token_overlap(name, needle)
    elif role_hint:
        # Pure role query like "the button": weak base score.
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

    Parameters
    ----------
    elements : Sequence[Element]
        Candidate elements to rank.
    description : str
        Natural-language target description.
    min_score : float, optional
        Minimum score an element must reach to be returned. Default is
        0.3.

    Returns
    -------
    List[Match]
        Matches scoring at or above ``min_score``, sorted by descending
        score. Empty if nothing plausibly matches.
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
    """Return the single best matching element, if any clears the threshold.

    Parameters
    ----------
    elements : Sequence[Element]
        Candidate elements to search.
    description : str
        Natural-language target description.
    min_score : float, optional
        Minimum score the winner must reach. Default is 0.3.

    Returns
    -------
    Element or None
        The highest scoring element, or None if nothing plausibly
        matches.
    """
    matches = rank_matches(elements, description, min_score=min_score)
    return matches[0].element if matches else None


def suggestions(elements: Sequence[Element], limit: int = 5) -> List[str]:
    """Collect nearby element labels to attach to a TargetNotFound error.

    Parameters
    ----------
    elements : Sequence[Element]
        Elements available on the surface when resolution failed.
    limit : int, optional
        Maximum number of suggestions to return. Default is 5.

    Returns
    -------
    List[str]
        Distinct non-blank element names, each truncated to 60
        characters, in observation order.

    Notes
    -----
    Suggestions are what let a failed action produce an actionable error
    rather than a bare "not found", so the caller can retarget without
    another full observation.
    """
    names = []
    for el in elements:
        if el.name and el.name.strip() and el.name not in names:
            names.append(el.name.strip()[:60])
    return names[:limit]
