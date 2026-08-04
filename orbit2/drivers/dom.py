"""DomDriver — Playwright/Patchright-backed DOM perception and action.

Transplanted from orbit v1:
- shadow-DOM-piercing element scan / click / fill / select routing
  (orbit/smart_dom_tools.py)
- diagnosis heuristics: not_found / invisible / disabled / covered /
  off_screen (orbit/_tools/perception.py) — converted to typed raises
- browser lifecycle with ephemeral-profile pattern and CDP attach
  (orbit/_tools/browser.py)

All heavy imports (patchright/playwright) are lazy so importing this
module works with zero optional deps installed.
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

log = logging.getLogger("orbit2.drivers.dom")

_TMP_PROFILE_PREFIX = "/tmp/orbit2-chrome-profile-"
_TMP_PROFILE_GLOB = "/tmp/orbit2-chrome-profile-*"
_PROFILE_FILES = ("Login Data", "Login Data-journal", "Cookies", "Web Data", "Web Data-journal")

_CHROME_FLAGS = [
    "--force-renderer-accessibility",
    "--enable-accessibility",
    "--disable-gpu",
    "--no-sandbox",
    "--disable-dev-shm-usage",  # Docker /dev/shm is 64MB; Chrome OOM-crashes without this
    "--window-size=1280,768",
]


def _async_playwright():
    """Lazy import: prefer patchright (stealth-patched), fall back to playwright."""
    try:
        from patchright.async_api import async_playwright  # type: ignore
    except ImportError:
        try:
            from playwright.async_api import async_playwright  # type: ignore
        except ImportError as exc:
            raise SurfaceUnreadable(
                "Neither patchright nor playwright is installed. "
                "Run: pip install patchright && patchright install chromium"
            ) from exc
    return async_playwright


# ---------------------------------------------------------------------------
# Shared JS library — shadow-DOM piercing helpers (transplanted verbatim
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
    """Driver over a Playwright/Patchright-managed Chromium page."""

    name = "dom"
    surface = "web"

    def __init__(
        self,
        *,
        persistent_profile: Optional[str] = None,
        executable_path: Optional[str] = None,
        headless: bool = False,
    ) -> None:
        self._persistent_profile = persistent_profile
        if executable_path is None:
            # Prefer a system Chrome over patchright's bundled Chromium,
            # which is typically not downloaded.
            for candidate in ("/usr/bin/google-chrome", "/usr/bin/chromium",
                              "/usr/bin/chromium-browser"):
                if os.path.exists(candidate):
                    executable_path = candidate
                    break
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
        """Launch Chromium with the ephemeral-profile pattern.

        The persistent profile (if configured) is copied to a temp dir so the
        source is never locked/corrupted by a live browser; credentials sync
        back on clean stop(). Stale temp dirs from aborted runs are purged.
        """
        async with self._lock:
            if self._playwright is not None:
                return  # idempotent: already running
            if purge_stale:
                for stale in glob.glob(_TMP_PROFILE_GLOB):
                    shutil.rmtree(stale, ignore_errors=True)
            tmp_profile = self._prepare_tmp_profile()
            factory = _async_playwright()
            try:
                playwright = await factory().start()
                kwargs: Dict[str, Any] = dict(
                    user_data_dir=tmp_profile,
                    headless=self._headless,
                    args=_CHROME_FLAGS,
                    ignore_default_args=["--enable-automation"],
                    ignore_https_errors=True,
                    no_viewport=True,
                )
                if self._executable_path:
                    kwargs["executable_path"] = self._executable_path
                context = await playwright.chromium.launch_persistent_context(**kwargs)
            except Exception:
                shutil.rmtree(tmp_profile, ignore_errors=True)
                raise
            self._playwright = playwright
            self._context = context
            self._tmp_profile = tmp_profile
            log.info("DomDriver started (profile: %s)", tmp_profile)

    @classmethod
    async def connect_cdp(cls, port: int) -> "DomDriver":
        """Attach to a running Chromium/Electron app on localhost:<port>.

        The target must have been launched with --remote-debugging-port.
        Common defaults: 9222 (Chrome), 9229 (Electron). No profile sync
        happens on stop() for CDP-attached instances.
        """
        driver = cls()
        factory = _async_playwright()
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
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    def _prepare_tmp_profile(self) -> str:
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
            text=raw.get("text", ""),
            modal_count=int(raw.get("modal_count", 0)),
            focused_key=focused_key,
        )

    @staticmethod
    def can_navigate(target: Optional[str]) -> bool:
        """This driver opens web targets (URLs / bare hosts)."""
        return is_web_target(target)

    async def act(self, action: Action) -> Optional[Element]:
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
        try:
            page = await self._require_page()
            return await page.screenshot(type="png", full_page=False)
        except Exception:
            return None

    # -- action internals ---------------------------------------------------

    @staticmethod
    def _normalize_chord(chord: str) -> str:
        """"ctrl+a" -> "Control+a"; single keys pass through."""
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
        raise TargetObstructed(
            f"{action.kind.value} target {action.target!r} is {blocked['reason']}",
            reason=blocked["reason"],
            target=action.target,
            element=blocked.get("element"),
            covered_by=blocked.get("coveredBy"),
        )

    def _by_ref(self, ref: Optional[int]) -> Optional[Element]:
        """Element the model pointed at, from the last rendered observation."""
        if ref is None:
            return None
        if 0 <= ref < len(self._last_elements):
            return self._last_elements[ref]
        raise TargetNotFound(
            f"ref {ref} is out of range (observation had "
            f"{len(self._last_elements)} elements) — observe again",
            target=str(ref),
        )

    async def _click(self, page: Any, action: Action) -> Element:
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
        # Real mouse click as a second event — handles isTrusted checks.
        try:
            await page.mouse.click(cx, cy)
        except Exception:
            pass  # JS click already fired; this is best-effort backup
        await asyncio.sleep(0.15)
        return _element_from_desc(clicked)

    async def _fill(self, page: Any, action: Action) -> Element:
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
        if not action.target or action.value is None:
            raise TargetNotFound("select requires target and value")
        # First try the fill path — it handles native <select> directly.
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
