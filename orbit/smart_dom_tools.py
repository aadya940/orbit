"""
smart_dom_tools.py — General-purpose shadow-DOM-aware browser interaction tools.

Drop this file anywhere and import the 8 functions into orbit's agents.py.
All tools pierce shadow roots automatically. Modal detection is auto-scoped
where it makes sense (dom_scan), but can be overridden.

Tools:
  dom_scan          — full-page element inventory (shadow + iframes)
  dom_smart_click   — click by selector / text / aria-label / role+text
  dom_smart_fill    — fill input by selector / label / placeholder
  dom_smart_select  — select dropdown option (native <select> + ARIA combobox)
  dom_smart_upload  — upload file (finds hidden inputs in shadow roots)
  dom_inspect       — deep-inspect an element (shadow root, z-index, interceptor)
  dom_await_element — poll until element appears (shadow-aware)
  dom_click_at      — click at exact viewport coordinates (x, y)
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from orbit._tools.browser import global_browser

log = logging.getLogger("orbit.smart_dom_tools")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED JS LIBRARY  — injected into every evaluate call
# Provides shadow-piercing query/walk helpers and React-aware fill.
# ─────────────────────────────────────────────────────────────────────────────

_JS_SHADOW_LIB = r"""
const ShadowLib = (() => {
    // Walk every element including shadow roots, calling visitor(node)
    function walk(root, visitor) {
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, null, false);
        let node;
        while ((node = walker.nextNode())) {
            visitor(node);
            if (node.shadowRoot) walk(node.shadowRoot, visitor);
        }
    }

    // querySelector that pierces shadow roots; returns first match
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

    // querySelectorAll that pierces shadow roots; returns all matches
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

    // Find elements by visible text, piercing shadow roots
    function findByText(text, tags, exact) {
        tags = tags || ['button','a','span','div','label','p','li'];
        exact = exact || false;
        const needle = text.toLowerCase();
        const results = [];
        walk(document, function(node) {
            if (!tags.includes(node.tagName.toLowerCase())) return;
            const t = (node.innerText || node.textContent || '').trim();
            if (exact ? t === text : t.toLowerCase().includes(needle)) results.push(node);
        });
        return results;
    }

    // Find input associated with a label, piercing shadow roots
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
        return null;
    }

    // Describe an element for return to Python
    function describe(el) {
        if (!el) return null;
        const rect = el.getBoundingClientRect();
        return {
            tag: el.tagName.toLowerCase(),
            id: el.id || null,
            text: (el.innerText || el.value || el.getAttribute('aria-label') ||
                   el.getAttribute('placeholder') || el.textContent || '').trim().slice(0, 120),
            ariaLabel: el.getAttribute('aria-label'),
            role: el.getAttribute('role'),
            type: el.type || null,
            placeholder: el.placeholder || null,
            value: el.value !== undefined ? String(el.value).slice(0, 80) : null,
            rect: { x: Math.round(rect.x), y: Math.round(rect.y),
                    w: Math.round(rect.width), h: Math.round(rect.height),
                    cx: Math.round(rect.x + rect.width / 2),
                    cy: Math.round(rect.y + rect.height / 2) },
            inShadow: !document.contains(el),
            visible: rect.width > 0 && rect.height > 0,
        };
    }

    // Fire React/Vue/Svelte-compatible input events
    function reactFill(el, value) {
        const tag = el.tagName;
        if (tag === 'SELECT') {
            // Native setter is type-checked to HTMLInputElement; use el.value directly
            el.value = value;
        } else {
            const proto = tag === 'TEXTAREA'
                ? window.HTMLTextAreaElement.prototype
                : window.HTMLInputElement.prototype;
            const desc = Object.getOwnPropertyDescriptor(proto, 'value');
            if (desc && desc.set) {
                desc.set.call(el, value);
            } else {
                el.value = value;
            }
        }
        el.dispatchEvent(new Event('input',  { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
    }

    // Detect the topmost open modal/dialog on the page
    function activeModal() {
        const sels = [
            '[role="dialog"]:not([aria-hidden="true"])',
            '[role="alertdialog"]:not([aria-hidden="true"])',
        ];
        for (const s of sels) {
            const m = document.querySelector(s);
            if (m && m.getBoundingClientRect().width > 0) return m;
        }
        return null;
    }

    return { walk, query, queryAll, findByText, findInputByLabel, describe, reactFill, activeModal };
})();
"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

async def _page_and_frame(frame_url: Optional[str] = None):
    """Return (page, frame) resolving optional iframe by URL fragment."""
    await global_browser.ensure_active_page()
    page = global_browser.active_page
    if not page:
        return None, None
    frame = global_browser.active_frame_or_page
    if frame_url:
        for f in page.frames:
            if frame_url in f.url:
                frame = f
                break
    return page, frame


# ─────────────────────────────────────────────────────────────────────────────
# 1. dom_scan
# ─────────────────────────────────────────────────────────────────────────────

async def dom_scan(
    scope: str = "auto",
    include_frames: bool = True,
    max_elements: int = 150,
) -> Dict[str, Any]:
    """Return a structured inventory of every interactive element on the page,
    including those inside shadow roots and (optionally) iframes.

    scope:
      "auto"  — if a modal/dialog is open, scope to it; otherwise full page
      "page"  — always scan the full page regardless of open modals
      "modal" — scope to the open modal/dialog; error if none is open
      A CSS selector — scope to that element

    include_frames: also scan sub-frames (iframes)
    max_elements: cap on returned elements (default 150)

    Each element includes: tag, text, ariaLabel, role, type, placeholder,
    value, rect (with cx/cy center coords), inShadow, inFrame, suggestedTool.
    When a modal is active, response includes modal_active=True.

    Call this whenever you're unsure what's available, or standard selectors
    return nothing — it reveals the full reachable DOM including shadow roots.
    """
    page, frame = await _page_and_frame()
    if not page:
        return {"status": "error", "message": "Browser is not active."}

    js = _JS_SHADOW_LIB + r"""
    ([scopeArg, maxEl]) => {
        // Resolve scope root
        let root = document;
        let modalActive = false;

        if (scopeArg === 'auto') {
            const m = ShadowLib.activeModal();
            if (m) { root = m; modalActive = true; }
        } else if (scopeArg === 'modal') {
            const m = ShadowLib.activeModal();
            if (!m) return { status: 'error', message: 'No open modal found.' };
            root = m; modalActive = true;
        } else if (scopeArg !== 'page') {
            const el = ShadowLib.query(scopeArg);
            if (!el) return { status: 'error', message: `Scope element not found: ${scopeArg}` };
            root = el;
        }

        const SELECTORS = [
            'button:not([disabled])', 'a[href]',
            'input:not([type="hidden"]):not([disabled])',
            'select:not([disabled])', 'textarea:not([disabled])',
            '[role="button"]:not([disabled])', '[role="link"]',
            '[role="tab"]', '[role="checkbox"]', '[role="radio"]',
            '[role="combobox"]', '[role="option"]', '[role="menuitem"]',
            '[contenteditable="true"]',
        ];

        const results = [];
        const seen = new Set();

        function scanRoot(scanFrom, frameUrl) {
            for (const sel of SELECTORS) {
                let els;
                try { els = [...scanFrom.querySelectorAll(sel)]; } catch(_) { continue; }
                for (const el of els) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 && rect.height === 0) continue;
                    const key = el.tagName + '|' + Math.round(rect.left) + '|' + Math.round(rect.top);
                    if (seen.has(key)) continue;
                    seen.add(key);

                    const desc = ShadowLib.describe(el);
                    desc.inFrame = frameUrl || null;
                    desc.inShadow = !document.contains(el);

                    // Suggest the right tool
                    const tag = el.tagName.toLowerCase();
                    const t = el.type || '';
                    if (t === 'file') {
                        desc.suggestedTool = 'dom_smart_upload';
                    } else if (tag === 'input' || tag === 'textarea' ||
                               el.getAttribute('contenteditable') === 'true') {
                        desc.suggestedTool = 'dom_smart_fill';
                    } else if (tag === 'select' || el.getAttribute('role') === 'combobox' ||
                               el.getAttribute('role') === 'listbox') {
                        desc.suggestedTool = 'dom_smart_select';
                    } else {
                        desc.suggestedTool = 'dom_smart_click';
                    }

                    results.push(desc);
                    if (results.length >= maxEl) return;
                }
            }
            // Recurse into shadow roots
            ShadowLib.walk(scanFrom, function(node) {
                if (!node.shadowRoot || results.length >= maxEl) return;
                scanRoot(node.shadowRoot, frameUrl);
            });
        }

        scanRoot(root, null);
        return { elements: results, modalActive };
    }
    """

    try:
        raw = await frame.evaluate(js, [scope, max_elements])
        if isinstance(raw, dict) and raw.get("status") == "error":
            return {"status": "error", "message": raw["message"]}

        elements = raw.get("elements", []) if isinstance(raw, dict) else raw
        modal_active = raw.get("modalActive", False) if isinstance(raw, dict) else False

        # Optionally scan iframes
        if include_frames and scope in ("auto", "page"):
            for f in page.frames[1:]:
                try:
                    sub = await f.evaluate(js, [scope, max_elements - len(elements)])
                    sub_els = sub.get("elements", []) if isinstance(sub, dict) else sub
                    for el in sub_els:
                        el["inFrame"] = f.url
                    elements.extend(sub_els)
                    if len(elements) >= max_elements:
                        break
                except Exception:
                    pass

        out = {
            "status": "success",
            "url": page.url,
            "count": len(elements),
            "elements": elements,
        }
        if modal_active:
            out["modal_active"] = True
            out["note"] = (
                "Modal is open — results scoped to dialog. "
                "Use scope='page' to override and scan the full page."
            )
        return out
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. dom_smart_click
# ─────────────────────────────────────────────────────────────────────────────

async def dom_smart_click(
    selector: Optional[str] = None,
    text: Optional[str] = None,
    aria_label: Optional[str] = None,
    role: Optional[str] = None,
    exact: bool = False,
    frame_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Click an element using any combination of selector, visible text,
    aria-label, or ARIA role — automatically piercing shadow roots.

    Priority: selector > aria_label > role+text > text.
    Provide at least one argument.

    selector:   CSS selector (pierces shadow DOM)
    text:       Visible text content of the element
    aria_label: aria-label attribute value (partial match)
    role:       ARIA role (e.g. "button", "tab", "menuitem")
    exact:      If True, text/aria_label must match exactly (default: partial)
    frame_url:  URL fragment of an iframe to search inside

    On failure, the error message tells you to call dom_scan to see what's
    available. On success, returns the element description with coords.

    Examples:
      dom_smart_click(text="Easy Apply")
      dom_smart_click(aria_label="Submit application")
      dom_smart_click(selector="#submit-btn")
      dom_smart_click(role="tab", text="Experience")
    """
    page, frame = await _page_and_frame(frame_url)
    if not page:
        return {"status": "error", "message": "Browser is not active."}

    js = _JS_SHADOW_LIB + r"""
    ({ selector, text, ariaLabel, role, exact }) => {
        let el = null;

        if (selector) {
            el = ShadowLib.query(selector);
        }
        if (!el && ariaLabel) {
            const needle = ariaLabel.toLowerCase();
            el = ShadowLib.query(`[aria-label="${ariaLabel}"]`);
            if (!el) {
                const all = ShadowLib.queryAll('[aria-label]');
                el = all.find(function(e) {
                    const v = (e.getAttribute('aria-label') || '').toLowerCase();
                    return exact ? v === needle : v.includes(needle);
                }) || null;
            }
        }
        if (!el && role && text) {
            const needle = text.toLowerCase();
            const candidates = ShadowLib.queryAll(`[role="${role}"]`);
            el = candidates.find(function(e) {
                const t = (e.innerText || e.textContent || '').trim().toLowerCase();
                return exact ? t === needle : t.includes(needle);
            }) || null;
        }
        if (!el && text) {
            const candidates = ShadowLib.findByText(text,
                ['button','a','span','div','li','p','label','input'], exact);
            // Filter to visible elements, then pick the one with the smallest area
            // to prefer specific buttons/links over large container divs that happen
            // to contain the same text somewhere in their subtree.
            const visible = candidates.filter(function(e) {
                const r = e.getBoundingClientRect();
                return r.width > 0 && r.height > 0;
            });
            visible.sort(function(a, b) {
                const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
                return (ra.width * ra.height) - (rb.width * rb.height);
            });
            el = visible[0] || null;
        }

        if (!el) return { found: false };
        const desc = ShadowLib.describe(el);
        el.click();
        return { found: true, clicked: desc };
    }
    """

    try:
        result = await frame.evaluate(js, {
            "selector": selector,
            "text": text,
            "ariaLabel": aria_label,
            "role": role,
            "exact": exact,
        })
        if not result.get("found"):
            return {
                "status": "error",
                "message": (
                    f"Element not found (selector={selector!r}, text={text!r}, "
                    f"aria_label={aria_label!r}, role={role!r}). "
                    "Call dom_scan to see what's available on the page."
                ),
            }
        return {"status": "success", "clicked": result["clicked"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 3. dom_smart_fill
# ─────────────────────────────────────────────────────────────────────────────

async def dom_smart_fill(
    value: str,
    selector: Optional[str] = None,
    label: Optional[str] = None,
    placeholder: Optional[str] = None,
    frame_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Fill an input, textarea, or contenteditable element.

    Finds the target using any of: CSS selector, label text, or placeholder.
    Fires native input+change events so React/Vue/Svelte state updates correctly.
    Automatically pierces shadow roots.

    selector:    CSS selector (pierces shadow DOM)
    label:       Visible label text near the field (e.g. "First name", "Email")
    placeholder: Placeholder attribute text
    value:       Text to type into the field
    frame_url:   URL fragment of an iframe to search inside

    Provide at least one of selector, label, or placeholder.

    Examples:
      dom_smart_fill(value="Alex Rivera", label="Full name")
      dom_smart_fill(value="alex@example.com", placeholder="Email address")
      dom_smart_fill(value="4152847391", selector="#phone-input")
    """
    page, frame = await _page_and_frame(frame_url)
    if not page:
        return {"status": "error", "message": "Browser is not active."}

    js = _JS_SHADOW_LIB + r"""
    ({ selector, label, placeholder, value }) => {
        let el = null;

        if (selector) {
            el = ShadowLib.query(selector);
        }
        if (!el && label) {
            el = ShadowLib.findInputByLabel(label);
        }
        if (!el && placeholder) {
            const needle = placeholder.toLowerCase();
            el = ShadowLib.query(`[placeholder="${placeholder}"]`);
            if (!el) {
                const all = ShadowLib.queryAll('[placeholder]');
                el = all.find(function(e) {
                    return (e.placeholder || '').toLowerCase().includes(needle);
                }) || null;
            }
        }

        if (!el) return { found: false };

        el.focus();
        el.scrollIntoView({ block: 'center' });

        if (el.getAttribute('contenteditable') === 'true') {
            // contenteditable: select all and replace
            const range = document.createRange();
            range.selectNodeContents(el);
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
            el.textContent = value;
            el.dispatchEvent(new InputEvent('input', { bubbles: true, data: value }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        } else {
            // input/textarea: use React-compatible native setter
            ShadowLib.reactFill(el, value);
        }

        return { found: true, filled: ShadowLib.describe(el) };
    }
    """

    try:
        result = await frame.evaluate(js, {
            "selector": selector,
            "label": label,
            "placeholder": placeholder,
            "value": value,
        })
        if not result.get("found"):
            return {
                "status": "error",
                "message": (
                    f"Input not found (selector={selector!r}, label={label!r}, "
                    f"placeholder={placeholder!r}). "
                    "Call dom_scan to see available inputs."
                ),
            }
        return {"status": "success", "filled": result["filled"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. dom_smart_select
# ─────────────────────────────────────────────────────────────────────────────

async def dom_smart_select(
    option_text: str,
    selector: Optional[str] = None,
    label: Optional[str] = None,
    frame_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Select an option from a dropdown — handles native <select> and ARIA
    comboboxes/listboxes. Automatically pierces shadow roots.

    selector:    CSS selector for the <select> or combobox trigger
    label:       Label text to find the field (used if selector not given)
    option_text: Visible text of the option to select (case-insensitive, partial)

    On failure, returns available options so you can correct the option text.

    Examples:
      dom_smart_select(option_text="United States", label="Country")
      dom_smart_select(option_text="Full-time", selector="[role='combobox']")
      dom_smart_select(option_text="2", label="Years of experience")
    """
    page, frame = await _page_and_frame(frame_url)
    if not page:
        return {"status": "error", "message": "Browser is not active."}

    # Step 1: find the trigger element and click it to open the dropdown
    find_js = _JS_SHADOW_LIB + r"""
    ({ selector, label }) => {
        let el = null;
        if (selector) el = ShadowLib.query(selector);
        if (!el && label) el = ShadowLib.findInputByLabel(label);
        if (!el) {
            // Fallback: first visible select or combobox
            const all = ShadowLib.queryAll('select,[role="combobox"],[role="listbox"]');
            el = all.find(function(e) {
                const r = e.getBoundingClientRect();
                return r.width > 0 && r.height > 0;
            }) || null;
        }
        if (!el) return null;

        const isNative = el.tagName.toLowerCase() === 'select';
        const desc = ShadowLib.describe(el);

        if (!isNative) {
            // Click to open custom dropdown
            el.click();
        }

        return { isNative, desc, inShadow: !document.contains(el) };
    }
    """

    try:
        trigger = await frame.evaluate(find_js, {"selector": selector, "label": label})
        if not trigger:
            return {
                "status": "error",
                "message": (
                    f"Dropdown not found (selector={selector!r}, label={label!r}). "
                    "Call dom_scan to see available dropdowns."
                ),
            }

        if trigger["isNative"]:
            # Native <select>: use JS to set value directly (works even in shadow DOM)
            pick_js = _JS_SHADOW_LIB + r"""
            ({ selector, label, optionText }) => {
                let el = null;
                if (selector) el = ShadowLib.query(selector);
                if (!el && label) el = ShadowLib.findInputByLabel(label);
                if (!el) {
                    const all = ShadowLib.queryAll('select');
                    el = all[0] || null;
                }
                if (!el) return { found: false, available: [] };

                const needle = optionText.toLowerCase();
                const options = [...el.options];
                const opt = options.find(function(o) {
                    return (o.text || '').toLowerCase().includes(needle);
                });
                if (!opt) return { found: false, available: options.map(function(o) { return o.text; }) };

                el.value = opt.value;
                el.dispatchEvent(new Event('change', { bubbles: true }));
                el.dispatchEvent(new Event('input',  { bubbles: true }));
                return { found: true, selected: opt.text };
            }
            """
            result = await frame.evaluate(pick_js, {
                "selector": selector, "label": label, "optionText": option_text
            })
            if not result.get("found"):
                return {
                    "status": "error",
                    "message": (
                        f"Option '{option_text}' not found in <select>. "
                        f"Available: {result.get('available', [])}"
                    ),
                }
            return {"status": "success", "selected": result["selected"], "field": trigger["desc"]}

        # Custom combobox: wait for options to render, then click the matching one
        await asyncio.sleep(0.4)

        pick_js = _JS_SHADOW_LIB + r"""
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
            if (!target) return { found: false, available: visible.map(function(c) { return (c.innerText || c.textContent || '').trim(); }) };
            target.click();
            return { found: true, selected: (target.innerText || target.textContent || '').trim() };
        }
        """
        result = await frame.evaluate(pick_js, option_text)
        if not result.get("found"):
            return {
                "status": "error",
                "message": (
                    f"Option '{option_text}' not visible after opening dropdown. "
                    f"Available: {result.get('available', [])}"
                ),
            }
        return {"status": "success", "selected": result["selected"], "field": trigger["desc"]}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 5. dom_smart_upload
# ─────────────────────────────────────────────────────────────────────────────

async def dom_smart_upload(
    path: str,
    selector: Optional[str] = None,
    label: Optional[str] = None,
    frame_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload a file to a file input — finds it even inside shadow roots or
    when it's hidden/invisible (common with custom upload UIs).

    path:     Absolute path to the file (e.g. '/workspace/RESUME.pdf')
    selector: CSS selector for <input type="file"> (optional)
    label:    Label text near the upload area (optional)

    If neither selector nor label is given, finds the first file input
    on the page including inside shadow roots.

    Examples:
      dom_smart_upload(path="/workspace/RESUME.pdf")
      dom_smart_upload(path="/workspace/RESUME.pdf", label="Resume")
      dom_smart_upload(path="/workspace/doc.pdf", selector="#cv-upload")
    """
    page, frame = await _page_and_frame(frame_url)
    if not page:
        return {"status": "error", "message": "Browser is not active."}

    # Surface the file input so Playwright can interact with it
    expose_js = _JS_SHADOW_LIB + r"""
    ({ selector, label }) => {
        let el = null;

        if (selector) el = ShadowLib.query(selector);

        if (!el && label) {
            const needle = label.toLowerCase();
            const labelEl = ShadowLib.queryAll('label').find(function(l) {
                return (l.textContent || '').toLowerCase().includes(needle);
            });
            if (labelEl) {
                el = labelEl.querySelector('input[type="file"]');
                if (!el) {
                    const sib = labelEl.nextElementSibling;
                    if (sib) el = sib.querySelector('input[type="file"]') || null;
                }
            }
        }

        if (!el) {
            el = ShadowLib.query('input[type="file"]');
        }

        if (!el) return null;

        // Make it reachable by Playwright (remove hidden/invisible styling)
        el.style.cssText = 'display:block!important;visibility:visible!important;'
            + 'opacity:1!important;position:fixed!important;top:0;left:0;'
            + 'width:100px;height:100px;z-index:999999;';

        // Always overwrite — existing IDs may contain special chars like '(' that break CSS selectors
        el.id = '__orbit_upload_' + Date.now() + '__';
        return { id: el.id, desc: ShadowLib.describe(el) };
    }
    """

    try:
        info = await frame.evaluate(expose_js, {"selector": selector, "label": label})
        if not info:
            return {
                "status": "error",
                "message": (
                    "No file input found. "
                    "Call dom_scan to confirm a file input exists on the page."
                ),
            }
        await frame.set_input_files(f"#{info['id']}", path)
        return {"status": "success", "uploaded": path, "input": info["desc"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 6. dom_inspect
# ─────────────────────────────────────────────────────────────────────────────

async def dom_inspect(
    selector: Optional[str] = None,
    text: Optional[str] = None,
    include_children: bool = True,
    frame_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Deep-inspect a specific element — returns attributes, computed styles,
    shadow root info, children, and what element (if any) is intercepting clicks.

    Use this to diagnose why a click or fill is failing:
      - Is the element inside a shadow root?  (inShadow: true)
      - Is something blocking clicks?         (interceptedBy field)
      - Is it actually visible?               (style.display, opacity, zIndex)
      - Does it have a shadow root itself?    (hasShadowRoot, shadowChildren)

    selector: CSS selector (pierces shadow DOM)
    text:     Visible text to locate the element (if no selector)
    include_children: Include first-level children and shadow children

    Examples:
      dom_inspect(text="Easy Apply")
      dom_inspect(selector="#submit-button")
      dom_inspect(selector=".modal-container")
    """
    page, frame = await _page_and_frame(frame_url)
    if not page:
        return {"status": "error", "message": "Browser is not active."}

    js = _JS_SHADOW_LIB + r"""
    ({ selector, text, includeChildren }) => {
        let el = null;
        if (selector) el = ShadowLib.query(selector);
        if (!el && text) {
            const candidates = ShadowLib.findByText(text);
            el = candidates[0] || null;
        }
        if (!el) return null;

        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        const cx = rect.x + rect.width / 2;
        const cy = rect.y + rect.height / 2;
        const topEl = document.elementFromPoint(cx, cy);

        const attrs = {};
        for (const a of el.attributes) attrs[a.name] = a.value;

        const result = {
            tag: el.tagName.toLowerCase(),
            id: el.id || null,
            attrs,
            text: (el.innerText || '').trim().slice(0, 300),
            rect: { x: Math.round(rect.x), y: Math.round(rect.y),
                    w: Math.round(rect.width), h: Math.round(rect.height),
                    cx: Math.round(cx), cy: Math.round(cy) },
            visible: rect.width > 0 && rect.height > 0,
            inShadow: !document.contains(el),
            hasShadowRoot: !!el.shadowRoot,
            shadowRootChildCount: el.shadowRoot ? el.shadowRoot.childElementCount : 0,
            style: {
                display:       style.display,
                visibility:    style.visibility,
                opacity:       style.opacity,
                zIndex:        style.zIndex,
                pointerEvents: style.pointerEvents,
                overflow:      style.overflow,
            },
            interceptedBy: (topEl && topEl !== el) ? {
                tag: topEl.tagName.toLowerCase(),
                id: topEl.id || null,
                class: (topEl.className || '').toString().slice(0, 80),
                rect: (function() {
                    const r = topEl.getBoundingClientRect();
                    return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) };
                })(),
            } : null,
        };

        if (includeChildren) {
            result.children = [...el.children].slice(0, 10).map(function(c) {
                return {
                    tag: c.tagName.toLowerCase(),
                    id: c.id || null,
                    text: (c.innerText || '').trim().slice(0, 80),
                    class: (c.className || '').toString().slice(0, 60),
                };
            });
            if (el.shadowRoot) {
                result.shadowChildren = [...el.shadowRoot.children].slice(0, 10).map(function(c) {
                    return {
                        tag: c.tagName.toLowerCase(),
                        id: c.id || null,
                        text: (c.innerText || '').trim().slice(0, 80),
                        class: (c.className || '').toString().slice(0, 60),
                    };
                });
            }
        }

        return result;
    }
    """

    try:
        result = await frame.evaluate(js, {
            "selector": selector,
            "text": text,
            "includeChildren": include_children,
        })
        if not result:
            return {
                "status": "error",
                "message": f"Element not found (selector={selector!r}, text={text!r}).",
            }
        return {"status": "success", "element": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 7. dom_await_element
# ─────────────────────────────────────────────────────────────────────────────

async def dom_await_element(
    selector: Optional[str] = None,
    text: Optional[str] = None,
    timeout_ms: int = 8000,
    visible: bool = True,
    frame_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Wait for an element to appear in the DOM (including shadow roots),
    polling until it becomes visible or timeout_ms is reached.

    Use after triggering an action that opens a modal, navigates, or loads
    dynamic content — before trying to interact with the new content.

    selector:   CSS selector to wait for
    text:       Visible text to wait for (if no selector)
    timeout_ms: Max wait time in milliseconds (default 8000)
    visible:    If True (default), also require the element to be visible
                (non-zero bounding box); set False to accept hidden elements

    Returns the element description when found, including rect with center coords.

    Examples:
      dom_await_element(text="Application submitted")
      dom_await_element(selector='[role="dialog"]')
      dom_await_element(text="Next", timeout_ms=5000)
    """
    page, frame = await _page_and_frame(frame_url)
    if not page:
        return {"status": "error", "message": "Browser is not active."}

    js = _JS_SHADOW_LIB + r"""
    ({ selector, text, requireVisible }) => {
        let el = null;
        if (selector) el = ShadowLib.query(selector);
        if (!el && text) {
            const candidates = ShadowLib.findByText(text);
            el = candidates.find(function(e) {
                if (!requireVisible) return true;
                const r = e.getBoundingClientRect();
                return r.width > 0 && r.height > 0;
            }) || null;
        }
        if (!el) return null;
        if (requireVisible) {
            const r = el.getBoundingClientRect();
            if (r.width === 0 && r.height === 0) return null;
        }
        return ShadowLib.describe(el);
    }
    """

    interval_ms = 200
    elapsed = 0
    while elapsed < timeout_ms:
        try:
            result = await frame.evaluate(js, {
                "selector": selector,
                "text": text,
                "requireVisible": visible,
            })
            if result:
                return {"status": "success", "found": result, "elapsed_ms": elapsed}
        except Exception:
            pass
        await asyncio.sleep(interval_ms / 1000)
        elapsed += interval_ms

    return {
        "status": "error",
        "message": (
            f"Element not found after {timeout_ms}ms "
            f"(selector={selector!r}, text={text!r}). "
            "The action may not have triggered, or the element is in an iframe. "
            "Try dom_scan to see what's currently on the page."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. dom_click_at
# ─────────────────────────────────────────────────────────────────────────────

async def dom_click_at(x: int, y: int) -> Dict[str, Any]:
    """Click at exact viewport coordinates using a real mouse event.

    Use the cx/cy values from dom_scan or dom_inspect results.
    Useful when el.click() isn't enough (sites checking event.isTrusted)
    or when you want to click at a specific point in a canvas/overlay.

    x: Horizontal pixel coordinate from left edge of viewport
    y: Vertical pixel coordinate from top edge of viewport

    Example:
      # Get coords from dom_scan, then click:
      dom_click_at(x=element['rect']['cx'], y=element['rect']['cy'])
    """
    await global_browser.ensure_active_page()
    page = global_browser.active_page
    if not page:
        return {"status": "error", "message": "Browser is not active."}
    try:
        await page.mouse.click(x, y)
        await asyncio.sleep(0.2)
        return {"status": "success", "clicked_at": {"x": x, "y": y}}
    except Exception as e:
        return {"status": "error", "message": str(e)}
