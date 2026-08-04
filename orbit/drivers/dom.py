"""DomDriver, Playwright and Patchright backed DOM perception and action.

Notes
-----
Transplanted from orbit v1:

* Shadow DOM piercing element scan, click, fill and select routing, from
  ``orbit/smart_dom_tools.py``.
* Diagnosis heuristics (not_found, invisible, disabled, covered,
  off_screen) from ``orbit/_tools/perception.py``, converted here into
  typed raises.
* Browser lifecycle with the ephemeral profile pattern and CDP attach,
  from ``orbit/_tools/browser.py``.

All heavy imports (patchright, playwright) are lazy, so importing this
module works with zero optional dependencies installed.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import shutil
import uuid
from typing import Any, Dict, List, Optional

from ..types import (
    Action,
    ActionKind,
    Bounds,
    Element,
    Observation,
    Source,
    SurfaceUnreadable,
    TargetNotFound,
    TargetObstructed,
)
from .base import is_web_target
from .matching import parse_description

log = logging.getLogger("orbit.drivers.dom")

_TMP_PROFILE_PREFIX = "/tmp/orbit-chrome-profile-"
_TMP_PROFILE_GLOB = "/tmp/orbit-chrome-profile-*"
# Frame scanning bounds: an ad-heavy page can carry dozens of tracking
# iframes, so cap both how many are scanned and how much each contributes.
_MAX_FRAMES = 12
_MAX_FRAME_ELEMENTS = 40

_PROFILE_FILES = ("Login Data", "Login Data-journal", "Cookies", "Web Data", "Web Data-journal")

# Only three rendering engines exist behind every mainstream browser, and
# Playwright exposes exactly those. Everything that isn't Firefox or
# WebKit is a Chromium fork (Edge, Brave, Opera, Vivaldi, Arc...), so the
# engine is derived, not enumerated, and the binary comes from the OS's
# own PATH lookup rather than a table of guessed install locations.
_FIREFOX_NAMES = {"firefox", "ff", "mozilla"}
_WEBKIT_NAMES = {"webkit", "safari", "epiphany"}

# Firefox rejects Chrome's flags; it needs prefs instead. Accessibility is
# force-enabled the same way, just through a different door.
_FIREFOX_PREFS = {
    "accessibility.force_disabled": 0,
    "devtools.accessibility.enabled": True,
}


def engine_for(browser: str) -> str:
    """Derive the Playwright engine that drives a named browser.

    Parameters
    ----------
    browser : str
        Browser name, for example ``"chrome"``, ``"firefox"`` or
        ``"brave"``.

    Returns
    -------
    str
        One of ``"chromium"``, ``"firefox"`` or ``"webkit"``.

    Notes
    -----
    Only three rendering engines exist behind every mainstream browser,
    and Playwright exposes exactly those. Anything that is not Firefox
    or WebKit is a Chromium fork (Edge, Brave, Opera, Vivaldi, Arc), so
    the engine is derived rather than enumerated and new forks need no
    table entry.
    """
    b = (browser or "chrome").strip().lower()
    if b in _FIREFOX_NAMES:
        return "firefox"
    if b in _WEBKIT_NAMES:
        return "webkit"
    return "chromium"


def find_browser_binary(browser: str) -> Optional[str]:
    """Locate a browser executable on PATH.

    Parameters
    ----------
    browser : str
        Browser name to look for.

    Returns
    -------
    str or None
        Absolute path to the executable, or None if no candidate name
        is on PATH.

    Notes
    -----
    The OS is asked rather than guessing from a table of install
    locations, and the common packaging suffixes for the same browser
    are tried in turn, since distributions disagree on whether the
    binary is called ``chrome``, ``chrome-browser`` or
    ``google-chrome-stable``.
    """
    import shutil

    b = (browser or "").strip().lower()
    if not b:
        return None
    candidates = [b, f"{b}-browser", f"{b}-stable", f"{b}-browser-stable"]
    if b == "chrome":
        candidates = ["google-chrome", "google-chrome-stable", *candidates]
    elif b == "edge":
        candidates = ["microsoft-edge", "microsoft-edge-stable", *candidates]
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    return None


_CHROME_FLAGS = [
    "--force-renderer-accessibility",
    "--enable-accessibility",
    "--disable-gpu",
    "--no-sandbox",
    "--disable-dev-shm-usage",  # Docker /dev/shm is 64MB; Chrome OOM-crashes without this
    "--window-size=1280,768",
]


def _async_playwright(engine: str = "chromium"):
    """Import the driver library appropriate for one engine.

    Parameters
    ----------
    engine : str, optional
        The Playwright engine that will be launched. Default is
        ``"chromium"``.

    Returns
    -------
    Any
        The ``async_playwright`` factory from the selected library.

    Raises
    ------
    SurfaceUnreadable
        If no suitable library is installed, with the exact install
        command in the message.

    Notes
    -----
    Patchright only stealth patches Chromium, so Chromium prefers
    Patchright when it is installed and falls back to stock Playwright
    otherwise.

    Firefox and WebKit must use stock Playwright. Patchright's CDP based
    patches break on those engines, with page evaluation failing on a
    cryptic ``_client`` error, so Patchright is refused here rather than
    letting a run fail deep inside an action.
    """
    # Chromium may use either library. Firefox/WebKit must use stock
    # Playwright: Patchright's CDP-based patches break on them (page
    # evaluation fails with a cryptic '_client' error), so we refuse it
    # here rather than failing deep inside a run.
    order = ["patchright", "playwright"] if engine == "chromium" else ["playwright"]
    for module in order:
        try:
            mod = __import__(f"{module}.async_api", fromlist=["async_playwright"])
            return mod.async_playwright
        except ImportError:
            continue
    raise SurfaceUnreadable(
        f"The {engine!r} engine needs stock Playwright. "
        f"Run: pip install playwright && playwright install {engine}"
    )


# ---------------------------------------------------------------------------
# Shared JS library: shadow-DOM piercing helpers (transplanted verbatim
# heuristics from smart_dom_tools._JS_SHADOW_LIB)
# ---------------------------------------------------------------------------

_JS_SHADOW_LIB = r"""
const ShadowLib = (() => {
    function walk(root, visitor) {
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, null, false);
        let node;
        while ((node = walker.nextNode())) {
            visitor(node);
            if (node.shadowRoot) walk(node.shadowRoot, visitor);
        }
    }

    function query(selector, root) {
        root = root || document;
        try { const el = root.querySelector(selector); if (el) return el; } catch (_) {}
        const hosts = root.querySelectorAll ? [...root.querySelectorAll('*')] : [];
        for (const host of hosts) {
            if (host.shadowRoot) {
                const found = query(selector, host.shadowRoot);
                if (found) return found;
            }
        }
        return null;
    }

    function queryAll(selector, root) {
        root = root || document;
        const results = [];
        try { results.push(...root.querySelectorAll(selector)); } catch (_) {}
        const hosts = root.querySelectorAll ? [...root.querySelectorAll('*')] : [];
        for (const host of hosts) {
            if (host.shadowRoot) results.push(...queryAll(selector, host.shadowRoot));
        }
        return results;
    }

    function findByText(text, tags, exact) {
        tags = tags || ['button','a','span','div','label','p','li'];
        exact = exact !== undefined ? exact : false;
        const needle = text.toLowerCase();
        const results = [];
        walk(document, function(node) {
            if (!tags.includes(node.tagName.toLowerCase())) return;
            const t = (node.innerText || node.textContent || '').trim();
            if (exact ? t === text : t.toLowerCase().includes(needle)) results.push(node);
        });
        return results;
    }

    function findInputByLabel(labelText) {
        const needle = labelText.toLowerCase();
        const labels = queryAll('label');
        for (const label of labels) {
            if (!(label.textContent || '').toLowerCase().includes(needle)) continue;
            if (label.htmlFor) {
                const el = query('#' + CSS.escape(label.htmlFor));
                if (el) return el;
            }
            const inner = label.querySelector('input,select,textarea');
            if (inner) return inner;
            const sib = label.nextElementSibling;
            if (sib) {
                const t = sib.tagName.toLowerCase();
                if (['input','select','textarea'].includes(t)) return sib;
                const child = sib.querySelector('input,select,textarea');
                if (child) return child;
            }
        }
        const ariaMatches = queryAll('input[aria-label],select[aria-label],textarea[aria-label]');
        for (const el of ariaMatches) {
            if ((el.getAttribute('aria-label') || '').toLowerCase().includes(needle)) return el;
        }
        const allInputs = queryAll('input,select,textarea');
        for (const el of allInputs) {
            const ids = (el.getAttribute('aria-labelledby') || '').split(' ').filter(Boolean);
            for (const id of ids) {
                const labelEl = document.getElementById(id);
                if (labelEl && (labelEl.textContent || '').toLowerCase().includes(needle)) return el;
            }
        }
        const placeholderMatches = queryAll('input[placeholder],textarea[placeholder]');
        for (const el of placeholderMatches) {
            if ((el.placeholder || '').toLowerCase().includes(needle)) return el;
        }
        return null;
    }

    function activeModal() {
        const sels = [
            '[role="dialog"]:not([aria-hidden="true"])',
            '[role="alertdialog"]:not([aria-hidden="true"])',
        ];
        for (const s of sels) {
            const m = document.querySelector(s);
            if (m && m.getBoundingClientRect().width > 0) return m;
        }
        let found = null;
        walk(document, function(node) {
            if (found) return;
            const role = node.getAttribute('role');
            if ((role === 'dialog' || role === 'alertdialog') &&
                node.getAttribute('aria-hidden') !== 'true') {
                const r = node.getBoundingClientRect();
                if (r.width > 0) found = node;
            }
        });
        return found;
    }

    function modalCount() {
        let n = 0;
        walk(document, function(node) {
            const role = node.getAttribute && node.getAttribute('role');
            if ((role === 'dialog' || role === 'alertdialog') &&
                node.getAttribute('aria-hidden') !== 'true') {
                const r = node.getBoundingClientRect();
                const s = getComputedStyle(node);
                if (r.width > 0 && s.display !== 'none' && s.visibility !== 'hidden' &&
                    parseFloat(s.opacity) >= 0.1) n++;
            }
        });
        return n;
    }

    function inViewport(rect) {
        return rect.top >= 0 && rect.left >= 0 &&
               rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
               rect.right  <= (window.innerWidth  || document.documentElement.clientWidth);
    }

    function roleOf(el) {
        const explicit = el.getAttribute('role');
        if (explicit) return explicit;
        const tag = el.tagName.toLowerCase();
        if (tag === 'a') return 'link';
        if (tag === 'button') return 'button';
        if (tag === 'select') return 'combobox';
        if (tag === 'textarea') return 'textbox';
        if (tag === 'input') {
            const t = (el.type || 'text').toLowerCase();
            if (t === 'checkbox') return 'checkbox';
            if (t === 'radio') return 'radio';
            if (t === 'submit' || t === 'button') return 'button';
            if (t === 'file') return 'file';
            return 'textbox';
        }
        if (el.getAttribute('contenteditable') === 'true') return 'textbox';
        return tag;
    }

    function describe(el) {
        if (!el) return null;
        const rect = el.getBoundingClientRect();
        const style = getComputedStyle(el);
        return {
            tag:        el.tagName.toLowerCase(),
            role:       roleOf(el),
            id:         el.id || null,
            name:       (el.getAttribute('aria-label') || el.innerText ||
                         el.getAttribute('placeholder') || el.textContent || '').trim().slice(0, 120),
            type:       el.type || null,
            value:      el.value !== undefined ? String(el.value).slice(0, 80) : null,
            disabled:   !!(el.disabled || el.getAttribute('aria-disabled') === 'true'),
            focused:    document.activeElement === el ||
                        (el.getRootNode() instanceof ShadowRoot &&
                         el.getRootNode().activeElement === el),
            rect: {
                x:  rect.x,  y:  rect.y, w: rect.width, h: rect.height,
                cx: Math.round(rect.x + rect.width  / 2),
                cy: Math.round(rect.y + rect.height / 2),
            },
            inShadow:   !document.contains(el),
            inViewport: inViewport(rect),
            visible:    rect.width > 0 && rect.height > 0 &&
                        style.display !== 'none' && style.visibility !== 'hidden' &&
                        parseFloat(style.opacity) > 0,
        };
    }

    // Why can't this element be acted on right now? null = actionable.
    function diagnose(el) {
        const d = describe(el);
        if (!d.visible) return { reason: 'invisible', element: d };
        if (d.disabled) return { reason: 'disabled', element: d };
        if (!d.inViewport) return { reason: 'off_screen', element: d };
        const top = document.elementFromPoint(d.rect.cx, d.rect.cy);
        if (top && top !== el && !el.contains(top) && !top.contains(el) &&
            !(el.getRootNode() instanceof ShadowRoot)) {
            return { reason: 'covered', element: d, coveredBy: describe(top) };
        }
        return null;
    }

    // React/Vue/Svelte-compatible fill via native prototype setter.
    function reactFill(el, value) {
        const tag = el.tagName;
        if (tag === 'SELECT') {
            el.value = value;
        } else {
            const proto = tag === 'TEXTAREA'
                ? window.HTMLTextAreaElement.prototype
                : window.HTMLInputElement.prototype;
            const desc = Object.getOwnPropertyDescriptor(proto, 'value');
            if (desc && desc.set) {
                desc.set.call(el, '');
                desc.set.call(el, value);
            } else {
                el.value = value;
            }
        }
        el.dispatchEvent(new Event('input',  { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
    }

    function mouseClick(el) {
        const r = el.getBoundingClientRect();
        el.dispatchEvent(new MouseEvent('click', {
            bubbles: true, cancelable: true, view: window,
            clientX: r.x + r.width / 2, clientY: r.y + r.height / 2,
        }));
    }

    // Resolve a click/select target by text / aria-label / role hint.
    // Priority + scoring transplanted from dom_smart_click: exact text >
    // modal-scoped > in-viewport > smallest area.
    function resolveTarget(text, roleHint) {
        let el = null;
        const needle = (text || '').toLowerCase();

        if (needle) {
            const all = queryAll('[aria-label]');
            el = all.find(function(e) {
                const v = (e.getAttribute('aria-label') || '').toLowerCase();
                return v === needle;
            }) || null;
        }

        if (!el && roleHint && needle) {
            const candidates = [];
            for (const r of roleHint) {
                candidates.push(...queryAll('[role="' + r + '"]'));
                if (r === 'button') candidates.push(...queryAll('button'));
                if (r === 'link') candidates.push(...queryAll('a[href]'));
                if (r === 'combobox') candidates.push(...queryAll('select'));
                if (r === 'checkbox') candidates.push(...queryAll('input[type="checkbox"]'));
                if (r === 'radio') candidates.push(...queryAll('input[type="radio"]'));
            }
            const scored = candidates.map(function(e) {
                const t = (e.innerText || e.textContent ||
                           e.getAttribute('aria-label') || e.value || '').trim();
                const tl = t.toLowerCase();
                if (!tl.includes(needle)) return null;
                const r = e.getBoundingClientRect();
                if (r.width === 0 && r.height === 0) return null;
                return { el: e, exact: tl === needle, vp: inViewport(r) };
            }).filter(Boolean);
            scored.sort(function(a, b) {
                if (a.exact !== b.exact) return a.exact ? -1 : 1;
                if (a.vp   !== b.vp)   return a.vp   ? -1 : 1;
                return 0;
            });
            el = scored.length ? scored[0].el : null;
        }

        if (!el && needle) {
            const partial = queryAll('[aria-label]').find(function(e) {
                return (e.getAttribute('aria-label') || '').toLowerCase().includes(needle);
            });
            if (partial) el = partial;
        }

        if (!el && text) {
            const candidates = findByText(text,
                ['button','a','span','div','li','p','label','input'], false);
            const modal = activeModal();
            const scored = candidates.map(function(e) {
                const r = e.getBoundingClientRect();
                if (r.width === 0 && r.height === 0) return null;
                const t = (e.innerText || e.textContent || '').trim();
                return {
                    el: e,
                    exactM: t === text,
                    vp: inViewport(r),
                    modal: modal ? modal.contains(e) : false,
                    area: r.width * r.height,
                };
            }).filter(Boolean);
            scored.sort(function(a, b) {
                if (a.exactM !== b.exactM) return a.exactM ? -1 : 1;
                if (a.modal  !== b.modal)  return a.modal  ? -1 : 1;
                if (a.vp     !== b.vp)     return a.vp     ? -1 : 1;
                return a.area - b.area;
            });
            el = scored.length ? scored[0].el : null;
        }

        return el;
    }

    // Closest-label suggestions for not_found diagnosis
    function closestLabels(text) {
        const all = queryAll('button,a[href],[role="button"],[role="link"],input,select,textarea');
        const labels = all.map(function(e) {
            return (e.innerText || e.getAttribute('aria-label') ||
                    e.getAttribute('placeholder') || e.textContent || '').trim();
        }).filter(function(t) { return t && t.length < 60; });
        const uniq = [...new Set(labels)];
        const needle = (text || '').toLowerCase();
        let closest = uniq.filter(function(t) {
            return needle && t.toLowerCase().includes(needle.slice(0, 4));
        }).slice(0, 5);
        if (closest.length === 0) closest = uniq.slice(0, 8);
        return closest;
    }

    return {
        walk, query, queryAll, findByText, findInputByLabel, activeModal,
        modalCount, inViewport, describe, diagnose, reactFill, mouseClick,
        resolveTarget, closestLabels, roleOf,
    };
})();
"""


# Full-page interactive-element inventory (transplanted from dom_scan).
_SCAN_JS = _JS_SHADOW_LIB + r"""
(maxEl) => {
    const SELECTORS = [
        'button', 'a[href]',
        'input:not([type="hidden"])',
        'select', 'textarea',
        '[role="button"]', '[role="link"]',
        '[role="tab"]', '[role="checkbox"]', '[role="radio"]',
        '[role="combobox"]', '[role="option"]', '[role="menuitem"]',
        '[contenteditable="true"]',
    ];
    const results = [];
    const seen = new Set();

    function scanRoot(scanFrom) {
        for (const sel of SELECTORS) {
            let els;
            try { els = [...scanFrom.querySelectorAll(sel)]; } catch(_) { continue; }
            for (const el of els) {
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 && rect.height === 0) continue;
                const key = el.tagName + '|' + Math.round(rect.left) + '|' + Math.round(rect.top);
                if (seen.has(key)) continue;
                seen.add(key);
                results.push(ShadowLib.describe(el));
                if (results.length >= maxEl) return;
            }
        }
        ShadowLib.walk(scanFrom, function(node) {
            if (!node.shadowRoot || results.length >= maxEl) return;
            scanRoot(node.shadowRoot);
        });
    }
    scanRoot(document);

    const body = document.body || document.documentElement;
    const text = (body.innerText || body.textContent || '').replace(/\s+/g, ' ').trim();

    return {
        elements: results,
        modal_count: ShadowLib.modalCount(),
        text: text.slice(0, 20000),
    };
}
"""


# Resolve target and click atomically (adapted from dom_smart_click).
_CLICK_JS = _JS_SHADOW_LIB + r"""
({ text, roleHint }) => {
    const el = ShadowLib.resolveTarget(text, roleHint);
    if (!el) return { found: false, closest: ShadowLib.closestLabels(text) };

    el.scrollIntoView({ block: 'nearest', behavior: 'instant' });
    const diag = ShadowLib.diagnose(el);
    if (diag && diag.reason !== 'off_screen') {
        // off_screen is fixed by the scrollIntoView above; the diagnose ran
        // before layout settled, so ignore it. Everything else blocks.
        return { found: true, blocked: diag };
    }

    const desc = ShadowLib.describe(el);
    ShadowLib.mouseClick(el);   // more trusted than el.click()
    return { found: true, clicked: desc };
}
"""


# Resolve field and fill atomically (adapted from dom_smart_fill, keeps the
# SELECT auto-routing and the LinkedIn "SELECT next to INPUT" heuristic).
_FILL_JS = _JS_SHADOW_LIB + r"""
({ label, value }) => {
    let el = ShadowLib.findInputByLabel(label);
    if (!el) {
        // fall back to text resolution (e.g. contenteditable near label text)
        el = ShadowLib.resolveTarget(label, ['textbox']);
        if (el && !['input','textarea','select'].includes(el.tagName.toLowerCase()) &&
            el.getAttribute('contenteditable') !== 'true') el = null;
    }
    if (!el) return { found: false, closest: ShadowLib.closestLabels(label) };

    // SELECT matched by label but value isn't one of its options: prefer a
    // non-SELECT input also matching the label (phone-country-code case).
    if (el.tagName === 'SELECT') {
        const needle = value.toLowerCase();
        const optMatch = [...el.options].find(function(o) {
            return (o.text || '').toLowerCase().includes(needle) ||
                   (o.value || '').toLowerCase().includes(needle);
        });
        if (!optMatch) {
            const needleL = label.toLowerCase();
            for (const cand of ShadowLib.queryAll('input,textarea')) {
                const al = (cand.getAttribute('aria-label') || '').toLowerCase();
                const ph = (cand.placeholder || '').toLowerCase();
                if (al.includes(needleL) || ph.includes(needleL)) { el = cand; break; }
                if (cand.id) {
                    const lbl = document.querySelector('label[for="' + CSS.escape(cand.id) + '"]');
                    if (lbl && (lbl.textContent || '').toLowerCase().includes(needleL)) {
                        el = cand; break;
                    }
                }
            }
        }
    }

    el.scrollIntoView({ block: 'center', behavior: 'instant' });
    const diag = ShadowLib.diagnose(el);
    if (diag && diag.reason !== 'off_screen' && diag.reason !== 'covered') {
        // 'covered' is common and harmless for fills (label overlays);
        // invisible/disabled genuinely block.
        return { found: true, blocked: diag };
    }
    el.focus();

    if (el.tagName === 'SELECT') {
        const needle = value.toLowerCase();
        const opt = [...el.options].find(function(o) {
            return (o.text || '').toLowerCase().includes(needle) ||
                   (o.value || '').toLowerCase().includes(needle);
        });
        if (opt) {
            el.value = opt.value;
            el.dispatchEvent(new Event('change', { bubbles: true }));
            el.dispatchEvent(new Event('input',  { bubbles: true }));
            return { found: true, routed: 'select', filled: ShadowLib.describe(el) };
        }
        // Pre-filled locked select (e.g. account email): accept existing value.
        if (el.value && el.value !== '' && el.value !== 'Select an option') {
            return { found: true, routed: 'select', preFilled: true,
                     filled: ShadowLib.describe(el) };
        }
        return { found: true, selectError: true,
                 available: [...el.options].map(function(o) { return o.text; }) };
    }

    if (el.getAttribute('contenteditable') === 'true') {
        const range = document.createRange();
        range.selectNodeContents(el);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        el.textContent = value;
        el.dispatchEvent(new InputEvent('input', { bubbles: true, data: value }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
    } else {
        ShadowLib.reactFill(el, value);
    }
    const actual = el.value !== undefined ? el.value : el.textContent;
    return {
        found: true, routed: 'fill', filled: ShadowLib.describe(el),
        verified: actual === value || String(actual).includes(value),
    };
}
"""


# Custom (ARIA) dropdown option pick, used after clicking the trigger.
_PICK_OPTION_JS = _JS_SHADOW_LIB + r"""
(optionText) => {
    const needle = optionText.toLowerCase();
    const candidates = ShadowLib.queryAll(
        '[role="option"],[role="listbox"] li,[role="menuitem"],option'
    );
    const visible = candidates.filter(function(c) {
        const r = c.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
    });
    const target = visible.find(function(c) {
        return (c.innerText || c.textContent || '').trim().toLowerCase().includes(needle);
    });
    if (!target) return {
        found: false,
        available: visible.map(function(c) { return (c.innerText || c.textContent || '').trim(); }),
    };
    target.scrollIntoView({ block: 'nearest' });
    ShadowLib.mouseClick(target);
    return { found: true, selected: (target.innerText || target.textContent || '').trim() };
}
"""


def _element_from_desc(desc: Dict[str, Any]) -> Element:
    """Convert one JS element description into an Element.

    Parameters
    ----------
    desc : Dict[str, Any]
        The description produced by ``ShadowLib.describe`` in the page.

    Returns
    -------
    Element
        The converted element, carrying DOM provenance and a ref holding
        the tag, id and center point.

    Notes
    -----
    The center point is kept in the ref so that a later action can click
    exactly the element that was described, without re-resolving it from
    its name.
    """
    rect = desc.get("rect") or {}
    return Element(
        role=desc.get("role") or desc.get("tag") or "generic",
        name=desc.get("name") or desc.get("text") or "",
        bounds=Bounds(
            x=float(rect.get("x", 0)), y=float(rect.get("y", 0)),
            width=float(rect.get("w", 0)), height=float(rect.get("h", 0)),
        ) if rect else None,
        provenance=frozenset({Source.DOM}),
        ref={"tag": desc.get("tag"), "id": desc.get("id"),
             "cx": rect.get("cx"), "cy": rect.get("cy")},
        enabled=not desc.get("disabled", False),
        focused=bool(desc.get("focused", False)),
        value=desc.get("value") or None,
    )


class DomDriver:
    """Driver over a Playwright or Patchright managed browser page.

    Attributes
    ----------
    name : str
        Driver id, ``"dom"``.
    surface : str
        ``"web"``, so the ladder never uses this driver as a rung for a
        native task, which would type into a background browser page.
    """

    name = "dom"
    surface = "web"

    def __init__(
        self,
        *,
        browser: str = "chrome",
        persistent_profile: Optional[str] = None,
        executable_path: Optional[str] = None,
        headless: bool = False,
    ) -> None:
        """Configure the driver without launching a browser.

        Parameters
        ----------
        browser : str, optional
            Browser name, used to derive the engine and to look up a
            system binary. Default is ``"chrome"``.
        persistent_profile : str, optional
            Path to a profile directory whose credentials should be
            copied in at start and synced back on clean stop. Default is
            None, meaning a throwaway profile.
        executable_path : str, optional
            Explicit browser binary to launch. Default is None, which
            triggers a PATH lookup for Chromium engines only.
        headless : bool, optional
            Whether to launch without a visible window. Default is
            False.

        Notes
        -----
        Chromium family browsers speak standard CDP, so any installed
        build works and a system binary beats the often absent bundled
        one. Firefox and WebKit are driven through a patched protocol
        and work ONLY with the driver's own build: a system Firefox
        exits immediately. Those engines are therefore never pointed at
        a system binary, which is why the PATH lookup is gated on the
        engine being Chromium.
        """
        self._persistent_profile = persistent_profile
        self._browser = (browser or "chrome").strip().lower()
        self._engine = engine_for(self._browser)
        # Chromium-family browsers speak standard CDP, so any installed
        # build works and a system binary beats the (often absent) bundled
        # one. Firefox/WebKit are driven through a patched protocol and
        # ONLY work with the driver's own build: a system Firefox exits
        # immediately. So we never point those at a system binary.
        if executable_path is None and self._engine == "chromium":
            executable_path = find_browser_binary(self._browser)
        self._executable_path = executable_path
        self._headless = headless
        self._playwright: Any = None
        self._context: Any = None
        self._page: Any = None
        self._tmp_profile: Optional[str] = None
        self._cdp_connected = False
        self._lock = asyncio.Lock()
        self._last_elements: List[Element] = []

    # -- lifecycle ----------------------------------------------------------

    async def start(self, *, purge_stale: bool = True) -> None:
        """Launch the browser using the ephemeral profile pattern.

        Parameters
        ----------
        purge_stale : bool, optional
            Whether to delete leftover temp profiles from aborted runs
            before starting. Default is True; the crash recovery path
            passes False so it does not disturb a concurrent session.

        Returns
        -------
        None

        Raises
        ------
        SurfaceUnreadable
            If no suitable driver library is installed for the engine.

        Notes
        -----
        The persistent profile, when configured, is copied to a temp
        directory so the source is never locked or corrupted by a live
        browser, and credentials sync back on a clean stop.

        Engine specific setup differs because Firefox rejects Chrome's
        command line flags outright and takes preferences instead.
        Accessibility is force enabled either way, just through a
        different door.

        The temp profile is removed if launch fails, so a failed start
        leaves nothing behind. The call is idempotent once running.
        """
        async with self._lock:
            if self._playwright is not None:
                return  # idempotent: already running
            if purge_stale:
                for stale in glob.glob(_TMP_PROFILE_GLOB):
                    shutil.rmtree(stale, ignore_errors=True)
            tmp_profile = self._prepare_tmp_profile()
            factory = _async_playwright(self._engine)
            try:
                playwright = await factory().start()
                kwargs: Dict[str, Any] = dict(
                    user_data_dir=tmp_profile,
                    headless=self._headless,
                    ignore_https_errors=True,
                    no_viewport=True,
                )
                # Engine-specific setup: Chromium takes flags, Firefox takes
                # prefs and rejects Chrome's flags outright.
                if self._engine == "chromium":
                    kwargs["args"] = _CHROME_FLAGS
                    kwargs["ignore_default_args"] = ["--enable-automation"]
                elif self._engine == "firefox":
                    kwargs["firefox_user_prefs"] = _FIREFOX_PREFS
                if self._executable_path:
                    kwargs["executable_path"] = self._executable_path
                engine = getattr(playwright, self._engine)
                context = await engine.launch_persistent_context(**kwargs)
            except Exception:
                shutil.rmtree(tmp_profile, ignore_errors=True)
                raise
            self._playwright = playwright
            self._context = context
            self._tmp_profile = tmp_profile
            log.info("DomDriver started (profile: %s)", tmp_profile)

    @classmethod
    async def connect_cdp(cls, port: int) -> "DomDriver":
        """Attach to a running Chromium or Electron app over CDP.

        Parameters
        ----------
        port : int
            Local debugging port the target is listening on. Common
            defaults are 9222 for Chrome and 9229 for Electron.

        Returns
        -------
        DomDriver
            A driver bound to the existing browser's first context.

        Raises
        ------
        SurfaceUnreadable
            If the target exposes no browser contexts.

        Notes
        -----
        The target must have been launched with
        ``--remote-debugging-port``. No profile sync happens on stop for
        CDP attached instances, because the profile belongs to the app
        we attached to and is not ours to write back.

        Playwright is stopped again if attaching fails, so a failed
        attach does not leak a driver process.
        """
        driver = cls()
        factory = _async_playwright('chromium')
        driver._playwright = await factory().start()
        try:
            browser = await driver._playwright.chromium.connect_over_cdp(
                f"http://localhost:{port}"
            )
            contexts = browser.contexts
            if not contexts:
                raise SurfaceUnreadable(
                    f"No browser contexts found on CDP port {port}. "
                    "Ensure the target app is running and the port is correct."
                )
            driver._context = contexts[0]
            pages = driver._context.pages
            driver._page = pages[0] if pages else await driver._context.new_page()
        except Exception:
            try:
                await driver._playwright.stop()
            except Exception:
                pass
            driver._playwright = None
            raise
        driver._cdp_connected = True
        return driver

    async def stop(self) -> None:
        """Close the browser, sync the profile back, and release resources.

        Returns
        -------
        None

        Notes
        -----
        Errors from closing are logged rather than raised, so teardown
        always completes and never masks the real outcome of a run. The
        profile is synced back only for browsers this driver launched,
        never for CDP attached ones, and the temp directory is removed
        either way.
        """
        async with self._lock:
            if self._context is not None:
                try:
                    await self._context.close()
                except Exception as exc:
                    log.warning("error closing browser context: %s", exc)
                self._context = None
                self._page = None
            if self._tmp_profile is not None:
                if not self._cdp_connected:
                    self._sync_profile_back(self._tmp_profile)
                shutil.rmtree(self._tmp_profile, ignore_errors=True)
                self._tmp_profile = None
            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                except Exception as exc:
                    log.warning("error stopping playwright: %s", exc)
                self._playwright = None

    async def __aenter__(self) -> "DomDriver":
        """Start the browser on entering an async context.

        Returns
        -------
        DomDriver
            This driver, started and ready.
        """
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Stop the browser on leaving an async context.

        Parameters
        ----------
        *exc : Any
            Standard exception triple, ignored: teardown is
            unconditional.

        Returns
        -------
        None
        """
        await self.stop()

    def _prepare_tmp_profile(self) -> str:
        """Create a throwaway profile directory, seeded from the persistent one.

        Returns
        -------
        str
            Path to the newly created temp profile directory.

        Notes
        -----
        Only the credential bearing files are copied, not the whole
        profile, which keeps startup fast and avoids dragging along
        caches or lock files from the source profile.
        """
        tmp = f"{_TMP_PROFILE_PREFIX}{uuid.uuid4().hex}"
        os.makedirs(os.path.join(tmp, "Default"), exist_ok=True)
        src_root = self._persistent_profile
        if src_root and os.path.exists(src_root):
            for f_name in _PROFILE_FILES:
                src = os.path.join(src_root, "Default", f_name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(tmp, "Default", f_name))
        return tmp

    def _sync_profile_back(self, tmp_profile: str) -> None:
        """Copy credential files from the temp profile back to the source.

        Parameters
        ----------
        tmp_profile : str
            Path to the temp profile being torn down.

        Returns
        -------
        None

        Notes
        -----
        This is how a login performed during a run survives to the next
        one. Failures are logged rather than raised, since losing a
        session refresh must not fail an otherwise successful run. The
        copy is a no-op when no persistent profile was configured.
        """
        dst_root = self._persistent_profile
        if not dst_root or not os.path.exists(tmp_profile):
            return
        try:
            os.makedirs(os.path.join(dst_root, "Default"), exist_ok=True)
            for f_name in _PROFILE_FILES:
                src = os.path.join(tmp_profile, "Default", f_name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(dst_root, "Default", f_name))
        except Exception as exc:
            log.error("profile sync failed: %s", exc)

    async def _require_page(self) -> Any:
        """Return a live page, recovering from a dead browser if necessary.

        Returns
        -------
        Any
            A Playwright page that is open and brought to the front.

        Raises
        ------
        SurfaceUnreadable
            If the browser cannot be started at all.

        Notes
        -----
        A page whose handle is still valid is reused. Otherwise a live
        page is picked from the context, preferring one that has
        actually navigated somewhere over a blank tab.

        If the browser or context died underneath us, from a crash or an
        external close, the driver relaunches rather than cascading the
        failure into every subsequent action. Stale profile purging is
        skipped on that path so a concurrent session is left alone.
        """
        if self._playwright is None:
            await self.start()
        if self._page is not None and not self._page.is_closed():
            return self._page
        self._page = None
        try:
            pages = [p for p in self._context.pages if not p.is_closed()]
            valid = [p for p in pages if p.url != "about:blank"]
            if valid:
                self._page = valid[0]
            elif pages:
                self._page = pages[0]
            else:
                self._page = await self._context.new_page()
        except Exception:
            # Browser/context died underneath us (crash, external close).
            # Recover with a full relaunch instead of cascading the failure.
            self._playwright = None
            self._context = None
            self._page = None
            await self.start(purge_stale=False)
            self._page = await self._context.new_page()
        try:
            await self._page.bring_to_front()
        except Exception:
            pass
        return self._page

    # -- Driver protocol ----------------------------------------------------

    async def observe(self) -> Observation:
        """Scan the live page into an observation.

        Returns
        -------
        Observation
            A browser observation carrying interactive elements from the
            main frame and every readable child frame, the page text,
            title, url and modal count.

        Raises
        ------
        SurfaceUnreadable
            If the page cannot be scanned even after one retry.

        Notes
        -----
        A tab closed or navigated mid observation drops the stale handle
        and retries once against whatever page is now live, which turns
        a common race into a non-event.

        The flattened element list is cached so a later act can address
        an element by ref: the model points at the index it was shown
        rather than re-describing the element in words, which removes
        any chance of the description resolving to a different element.
        """
        for attempt in (0, 1):
            page = await self._require_page()
            try:
                raw = await page.evaluate(_SCAN_JS, 150)
                title = await page.title()
                url = page.url
                break
            except Exception as exc:
                # Tab closed/navigated mid-observation: drop the handle and
                # retry once against whatever page is now live.
                if attempt == 0 and ("closed" in str(exc).lower() or page.is_closed()):
                    self._page = None
                    continue
                raise SurfaceUnreadable(f"DOM scan failed: {exc}") from exc

        elements = [_element_from_desc(d) for d in raw.get("elements", []) if d]
        text = raw.get("text", "")

        # Iframes are separate documents, so the main-frame scan cannot see
        # into them: embedded checkouts, payment fields, editors and consent
        # dialogs would all be invisible. Scan each child frame and translate
        # its coordinates into page space so clicks still land.
        frame_elements, frame_text = await self._scan_frames(page)
        elements += frame_elements
        if frame_text:
            text = f"{text}\n{frame_text}" if text else frame_text

        # Cache for ref addressing: the model points at the index it was
        # shown rather than re-describing the element in words.
        self._last_elements = elements
        focused_key = next((e.key for e in elements if e.focused), None)
        return Observation(
            surface="browser:main",
            kind="browser",
            title=title,
            url=url,
            elements=elements,
            text=text,
            modal_count=int(raw.get("modal_count", 0)),
            focused_key=focused_key,
        )

    async def _scan_frames(self, page: Any) -> tuple:
        """Scan child frames, returning page space elements and their text.

        Parameters
        ----------
        page : Any
            The Playwright page whose child frames should be scanned.

        Returns
        -------
        tuple
            A pair of the page space elements and the joined frame text.

        Notes
        -----
        Iframes are separate documents, so the main frame scan cannot
        see into them. Embedded checkouts, payment fields, editors and
        consent dialogs would all be invisible without scanning them
        explicitly.

        Each frame reports geometry relative to its own document, so
        every rect is offset by the frame's position within the page.
        Without that translation a click computed from an iframe element
        would land in the wrong place on screen.

        Frames that cannot be read, whether detached mid scan or still
        blank, are skipped rather than failing the whole observation.
        Both the frame count and the elements taken from each are capped
        because an ad heavy page can carry dozens of tracking iframes.
        """
        elements: List[Element] = []
        texts: List[str] = []
        try:
            frames = [f for f in page.frames if f is not page.main_frame]
        except Exception:
            return elements, ""

        for frame in frames[:_MAX_FRAMES]:
            try:
                handle = await frame.frame_element()
                box = await handle.bounding_box()
                if not box or box["width"] < 1 or box["height"] < 1:
                    continue  # hidden or zero-sized frame: nothing to see
                raw = await frame.evaluate(_SCAN_JS, _MAX_FRAME_ELEMENTS)
            except Exception:
                continue

            dx, dy = float(box["x"]), float(box["y"])
            for desc in raw.get("elements", []) or []:
                if not desc:
                    continue
                rect = desc.get("rect")
                if rect:
                    for key, delta in (("x", dx), ("cx", dx), ("y", dy), ("cy", dy)):
                        if rect.get(key) is not None:
                            rect[key] = float(rect[key]) + delta
                element = _element_from_desc(desc)
                element.ref["frame_url"] = frame.url
                elements.append(element)
            if raw.get("text"):
                texts.append(raw["text"])
        return elements, "\n".join(texts)

    @staticmethod
    def can_navigate(target: Optional[str]) -> bool:
        """Report whether this driver claims a navigate target.

        Parameters
        ----------
        target : str or None
            The navigate target to classify.

        Returns
        -------
        bool
            True for web targets: URLs and bare hosts.

        Notes
        -----
        The complement of this predicate is claimed by the
        accessibility driver, and both route by the same shared
        :func:`is_web_target` definition.
        """
        return is_web_target(target)

    def pointer(self) -> "_PagePointer":
        """Build a surface local input adapter for this page.

        Returns
        -------
        _PagePointer
            An adapter driving the page's own mouse and keyboard.

        Notes
        -----
        This is handed to the vision rung so that grounded page
        coordinates are clicked in page space rather than OS global
        screen space. Vision grounds on the page bitmap, so acting
        through page input keeps both halves in the same coordinate
        system and works even without a reachable display.
        """
        return _PagePointer(self)

    async def act(self, action: Action) -> Optional[Element]:
        """Resolve a target on the live page and act on it atomically.

        Parameters
        ----------
        action : Action
            The action to perform.

        Returns
        -------
        Element or None
            The element acted on, or None for NAVIGATE, PRESS and
            untargeted SCROLL.

        Raises
        ------
        TargetNotFound
            If a required value is missing, the target cannot be
            resolved, or the action kind is unsupported.
        TargetObstructed
            If the element is present but invisible, disabled or
            covered.
        SurfaceUnreadable
            If no live page can be obtained.

        Notes
        -----
        A bare target string is given a scheme when navigating, so that
        a bare host resolves rather than being treated as a relative
        path.
        """
        page = await self._require_page()

        if action.kind is ActionKind.NAVIGATE:
            if not action.value:
                raise TargetNotFound("navigate requires a url in action.value")
            url = action.value if "://" in action.value else f"https://{action.value}"
            await page.goto(url, wait_until="domcontentloaded")
            return None

        if action.kind is ActionKind.PRESS:
            if not action.value:
                raise TargetNotFound("press requires a key chord in action.value")
            await page.keyboard.press(self._normalize_chord(action.value))
            return None

        if action.kind is ActionKind.SCROLL:
            return await self._scroll(page, action)

        if action.kind is ActionKind.CLICK:
            return await self._click(page, action)

        if action.kind is ActionKind.FILL:
            return await self._fill(page, action)

        if action.kind is ActionKind.SELECT:
            return await self._select(page, action)

        raise TargetNotFound(f"unsupported action kind: {action.kind}")

    async def screenshot(self) -> Optional[bytes]:
        """Capture the visible page area as PNG bytes.

        Returns
        -------
        bytes or None
            PNG bytes, or None if the page cannot be captured.

        Notes
        -----
        The capture is viewport only, not full page, because it is used
        for vision grounding and must correspond to what is actually on
        screen for coordinates to be meaningful. Errors are reported as
        None, matching the protocol's treatment of an absent screenshot.
        """
        try:
            page = await self._require_page()
            return await page.screenshot(type="png", full_page=False)
        except Exception:
            return None

    # -- action internals ---------------------------------------------------

    @staticmethod
    def _normalize_chord(chord: str) -> str:
        """Translate a colloquial key chord into Playwright's key names.

        Parameters
        ----------
        chord : str
            Key chord with parts separated by plus signs, for example
            ``"ctrl+a"``.

        Returns
        -------
        str
            The chord with known aliases replaced, for example
            ``"Control+a"``. Unrecognised parts, including single
            character keys, pass through unchanged.
        """
        replacements = {"ctrl": "Control", "cmd": "Meta", "win": "Meta",
                        "alt": "Alt", "shift": "Shift", "enter": "Enter",
                        "esc": "Escape", "escape": "Escape", "tab": "Tab",
                        "space": "Space", "backspace": "Backspace",
                        "delete": "Delete", "up": "ArrowUp", "down": "ArrowDown",
                        "left": "ArrowLeft", "right": "ArrowRight",
                        "pageup": "PageUp", "pagedown": "PageDown",
                        "home": "Home", "end": "End"}
        parts = [p.strip() for p in chord.split("+")]
        return "+".join(replacements.get(p.lower(), p) for p in parts)

    async def _scroll(self, page: Any, action: Action) -> Optional[Element]:
        """Scroll an element into view, or scroll the page in a direction.

        Parameters
        ----------
        page : Any
            The live Playwright page.
        action : Action
            The scroll action. A target scrolls that element into view;
            otherwise the value names a direction.

        Returns
        -------
        Element or None
            The element scrolled into view, or None for a directional
            page scroll.

        Raises
        ------
        TargetNotFound
            If a target was named but no element matches it.
        """
        if action.target:
            found = await page.evaluate(
                _JS_SHADOW_LIB + r"""
                (text) => {
                    const el = ShadowLib.resolveTarget(text, null);
                    if (!el) return null;
                    el.scrollIntoView({ block: 'center', behavior: 'instant' });
                    return ShadowLib.describe(el);
                }""",
                action.target,
            )
            if not found:
                raise TargetNotFound(
                    f"scroll target {action.target!r} not found", target=action.target
                )
            return _element_from_desc(found)
        direction = (action.value or "down").lower()
        dy = {"down": 600, "up": -600}.get(direction, 600)
        dx = {"left": -600, "right": 600}.get(direction, 0)
        if direction in ("left", "right"):
            dy = 0
        await page.mouse.wheel(dx, dy)
        return None

    def _raise_blocked(self, action: Action, blocked: Dict[str, Any]) -> None:
        """Convert a JS diagnosis into a typed obstruction error.

        Parameters
        ----------
        action : Action
            The action that was blocked, used for its kind and target in
            the message.
        blocked : Dict[str, Any]
            The diagnosis from ``ShadowLib.diagnose``, carrying a reason
            and the element, plus the covering element when relevant.

        Returns
        -------
        None
            Never returns normally.

        Raises
        ------
        TargetObstructed
            Always, carrying the reason and element details so a caller
            can tell "covered by a banner" from "genuinely disabled".
        """
        raise TargetObstructed(
            f"{action.kind.value} target {action.target!r} is {blocked['reason']}",
            reason=blocked["reason"],
            target=action.target,
            element=blocked.get("element"),
            covered_by=blocked.get("coveredBy"),
        )

    def _by_ref(self, ref: Optional[int]) -> Optional[Element]:
        """Look up the element the model pointed at by observation index.

        Parameters
        ----------
        ref : int or None
            Index into the last rendered observation, or None when the
            action carried no ref.

        Returns
        -------
        Element or None
            The referenced element, or None when ``ref`` is None.

        Raises
        ------
        TargetNotFound
            If the ref is out of range, meaning the caller is working
            from a stale view and should observe again.

        Notes
        -----
        Addressing by ref rather than re-matching by name is what keeps
        acting deterministic: the index names exactly the element that
        was rendered, even when several controls share a label.
        """
        if ref is None:
            return None
        if 0 <= ref < len(self._last_elements):
            return self._last_elements[ref]
        raise TargetNotFound(
            f"ref {ref} is out of range (observation had "
            f"{len(self._last_elements)} elements), observe again",
            target=str(ref),
        )

    async def _click(self, page: Any, action: Action) -> Element:
        """Resolve a click target and click it atomically.

        Parameters
        ----------
        page : Any
            The live Playwright page.
        action : Action
            The click action, carrying a ref or a target description.

        Returns
        -------
        Element
            The element that was clicked.

        Raises
        ------
        TargetNotFound
            If neither a ref nor a target was supplied, or nothing
            matches the description, with the closest labels attached.
        TargetObstructed
            If the element is disabled, invisible or covered.

        Notes
        -----
        A ref with known coordinates clicks directly, skipping
        resolution entirely. A ref whose coordinates are missing
        degrades to a name based resolution using the element's own
        name, which is still better than the original description.

        After the in page click, a real mouse click is fired at the same
        point as a second event, which satisfies ``isTrusted`` checks
        that reject synthetic events. It is best effort: the JS click
        has already fired, so a failure here is deliberately ignored.
        """
        el = self._by_ref(action.ref)
        if el is not None:
            cx, cy = el.ref.get("cx"), el.ref.get("cy")
            if cx is not None and cy is not None:
                if not el.enabled:
                    raise TargetObstructed(
                        f"element {el.name!r} is disabled", reason="disabled",
                    )
                await page.mouse.click(float(cx), float(cy))
                await asyncio.sleep(0.15)
                return el
            action = Action(kind=action.kind, target=el.name or action.target,
                            value=action.value)
        if not action.target:
            raise TargetNotFound("click requires a ref or target description")
        text, role_hint = parse_description(action.target)
        result = await page.evaluate(
            _CLICK_JS, {"text": text or action.target, "roleHint": list(role_hint or [])}
        )
        if not result.get("found"):
            raise TargetNotFound(
                f"no element matches {action.target!r}",
                target=action.target,
                suggestions=result.get("closest", []),
            )
        if result.get("blocked"):
            self._raise_blocked(action, result["blocked"])

        clicked = result["clicked"]
        cx, cy = clicked["rect"]["cx"], clicked["rect"]["cy"]
        # Real mouse click as a second event: handles isTrusted checks.
        try:
            await page.mouse.click(cx, cy)
        except Exception:
            pass  # JS click already fired; this is best-effort backup
        await asyncio.sleep(0.15)
        return _element_from_desc(clicked)

    async def _fill(self, page: Any, action: Action) -> Element:
        """Resolve a field and fill it atomically.

        Parameters
        ----------
        page : Any
            The live Playwright page.
        action : Action
            The fill action, carrying a ref or target plus the value.

        Returns
        -------
        Element
            The element that was filled.

        Raises
        ------
        TargetNotFound
            If no value was supplied, neither ref nor target was given,
            no input matches, or the field is a select whose options do
            not include the requested value.
        TargetObstructed
            If the field is invisible or disabled.

        Notes
        -----
        A ref with known coordinates clicks the field and types through
        real key events, selecting existing content first so the value
        is replaced rather than appended.

        The in page path routes native selects automatically and keeps
        the heuristic of preferring a non-select input that also matches
        the label, which handles the common phone country code layout
        where a select sits beside the real text input.
        """
        if action.value is None:
            raise TargetNotFound("fill requires a value")
        el = self._by_ref(action.ref)
        if el is not None:
            cx, cy = el.ref.get("cx"), el.ref.get("cy")
            if cx is not None and cy is not None:
                await page.mouse.click(float(cx), float(cy))
                await page.keyboard.press("Control+a")
                await page.keyboard.type(str(action.value))
                await asyncio.sleep(0.1)
                return el
            action = Action(kind=action.kind, target=el.name or action.target,
                            value=action.value)
        if not action.target:
            raise TargetNotFound("fill requires a ref or target description")
        label, _ = parse_description(action.target)
        result = await page.evaluate(
            _FILL_JS, {"label": label or action.target, "value": action.value}
        )
        if not result.get("found"):
            raise TargetNotFound(
                f"no input matches {action.target!r}",
                target=action.target,
                suggestions=result.get("closest", []),
            )
        if result.get("blocked"):
            self._raise_blocked(action, result["blocked"])
        if result.get("selectError"):
            raise TargetNotFound(
                f"field {action.target!r} is a <select> and option "
                f"{action.value!r} is not available",
                target=action.target,
                available=result.get("available", []),
            )
        return _element_from_desc(result["filled"])

    async def _select(self, page: Any, action: Action) -> Element:
        """Choose an option in a native or custom dropdown.

        Parameters
        ----------
        page : Any
            The live Playwright page.
        action : Action
            The select action, carrying the dropdown target and the
            option value.

        Returns
        -------
        Element
            The dropdown or option element that was selected.

        Raises
        ------
        TargetNotFound
            If target or value is missing, no dropdown matches, or the
            option never becomes visible, with the available options
            attached.

        Notes
        -----
        Three strategies are tried in order of reliability: the fill
        path, which handles a native select directly; clicking the
        trigger and picking a rendered ARIA option; and finally a
        keyboard type ahead, transplanted from ``dom_smart_select``, for
        virtualised lists that only render the matching option once
        typed into.
        """
        if not action.target or action.value is None:
            raise TargetNotFound("select requires target and value")
        # First try the fill path: it handles native <select> directly.
        label, _ = parse_description(action.target)
        result = await page.evaluate(
            _FILL_JS, {"label": label or action.target, "value": action.value}
        )
        if result.get("found") and result.get("routed") == "select" and result.get("filled"):
            return _element_from_desc(result["filled"])

        # Custom combobox: click the trigger, then pick the option.
        trigger = await page.evaluate(
            _JS_SHADOW_LIB + r"""
            (label) => {
                let el = ShadowLib.findInputByLabel(label) ||
                         ShadowLib.resolveTarget(label, ['combobox']);
                if (!el) {
                    const all = ShadowLib.queryAll('select,[role="combobox"],[role="listbox"]');
                    el = all.find(function(e) {
                        const r = e.getBoundingClientRect();
                        return r.width > 0 && r.height > 0;
                    }) || null;
                }
                if (!el) return null;
                el.scrollIntoView({ block: 'center', behavior: 'instant' });
                if (el.tagName.toLowerCase() !== 'select') el.click();
                return ShadowLib.describe(el);
            }""",
            label or action.target,
        )
        if not trigger:
            raise TargetNotFound(
                f"no dropdown matches {action.target!r}", target=action.target
            )
        await asyncio.sleep(0.5)  # wait for options to render
        picked = await page.evaluate(_PICK_OPTION_JS, action.value)
        if picked.get("found"):
            return _element_from_desc(trigger)

        # Keyboard type-ahead fallback (transplanted from dom_smart_select).
        cx, cy = trigger["rect"]["cx"], trigger["rect"]["cy"]
        try:
            await page.mouse.click(cx, cy)
            await asyncio.sleep(0.2)
            await page.keyboard.type(action.value[:3], delay=80)
            await asyncio.sleep(0.4)
            picked2 = await page.evaluate(_PICK_OPTION_JS, action.value)
            if picked2.get("found"):
                return _element_from_desc(trigger)
        except Exception:
            pass
        raise TargetNotFound(
            f"option {action.value!r} not visible after opening dropdown "
            f"{action.target!r}",
            target=action.target,
            available=picked.get("available", []),
        )


class _PagePointer:
    """Page space input adapter used by the vision rung on browser surfaces.

    Notes
    -----
    Vision grounds on the page bitmap, so its coordinates are in page
    space. Driving the page's own mouse and keyboard keeps acting in
    that same space, instead of translating to OS global screen
    coordinates and depending on window position and a reachable
    display.
    """

    def __init__(self, driver: "DomDriver") -> None:
        """Bind the adapter to a DOM driver.

        Parameters
        ----------
        driver : DomDriver
            The driver whose live page will receive the input.
        """
        self._driver = driver

    async def click(self, x: float, y: float) -> None:
        """Click a point in page space.

        Parameters
        ----------
        x : float
            Horizontal page coordinate.
        y : float
            Vertical page coordinate.

        Returns
        -------
        None

        Notes
        -----
        A short pause afterwards gives the page a moment to react before
        the caller observes, so the effect check sees the result rather
        than the state before it.
        """
        page = await self._driver._require_page()
        await page.mouse.click(float(x), float(y))
        await asyncio.sleep(0.15)

    async def type(self, text: str) -> None:
        """Type text at the current focus, replacing existing content.

        Parameters
        ----------
        text : str
            The text to type.

        Returns
        -------
        None

        Notes
        -----
        Existing content is selected first so the value is replaced
        rather than appended to whatever the field already held.
        """
        page = await self._driver._require_page()
        await page.keyboard.press("Control+a")
        await page.keyboard.type(text)

    async def press(self, chord: str) -> None:
        """Send a key chord to the page.

        Parameters
        ----------
        chord : str
            Key chord in colloquial form, for example ``"ctrl+s"``,
            normalized to Playwright's key names before sending.

        Returns
        -------
        None
        """
        page = await self._driver._require_page()
        await page.keyboard.press(self._driver._normalize_chord(chord))

    async def scroll(self, x: float, y: float, direction: str) -> None:
        """Scroll the page at a point in page space.

        Parameters
        ----------
        x : float
            Horizontal page coordinate to scroll at.
        y : float
            Vertical page coordinate to scroll at.
        direction : str
            Either ``"up"`` or anything else, which scrolls down.

        Returns
        -------
        None

        Notes
        -----
        The pointer is moved first so the wheel event lands on the
        intended scrollable container rather than wherever the pointer
        happened to be.
        """
        page = await self._driver._require_page()
        await page.mouse.move(float(x), float(y))
        await page.mouse.wheel(0, -400 if direction == "up" else 400)
