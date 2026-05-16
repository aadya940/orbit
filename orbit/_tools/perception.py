"""
perception.py — Stateful page perception layer for orbit's DOM tools.

This module is the brain behind smart_dom_tools. Instead of every action
returning {"status": "success"}, actions now:

  1. Snapshot the page BEFORE acting (URL, title, text hash, modals, toasts)
  2. Execute the action
  3. Snapshot AFTER, compute a diff
  4. On failure, run an automated diagnosis (does element exist? visible?
     covered? off-screen? closest matches?)

The Python side keeps a PageState per browser name so each tool call sees
the consequences of the previous one — no more amnesiac round-trips.

Public surface (called from smart_dom_tools.py):
  snapshot(frame)                — take a StateSnapshot via JS
  diff(before, after)            — return list of changed keys + payload
  diagnose(frame, criteria)      — explain why an action failed
  understand(frame)              — 1-paragraph semantic page summary
  get_state(name) / set_state    — PageState registry
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

log = logging.getLogger("orbit.perception")


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StateSnapshot:
    """A point-in-time fingerprint of the page."""
    url: str = ""
    title: str = ""
    text_hash: str = ""           # sha1 of body innerText (trimmed)
    text_len: int = 0
    modal_count: int = 0
    modal_titles: List[str] = field(default_factory=list)
    toasts: List[str] = field(default_factory=list)
    form_fields_total: int = 0
    form_fields_filled: int = 0
    interactive_count: int = 0
    primary_action: Optional[str] = None   # label of first prominent button

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PageState:
    """Persistent state for one browser tab, keyed by browser name."""
    snapshot: Optional[StateSnapshot] = None
    last_action: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = field(default_factory=list)   # last N actions

    def record(self, action: Dict[str, Any]) -> None:
        self.last_action = action
        self.history.append(action)
        if len(self.history) > 20:
            self.history.pop(0)


_states: Dict[str, PageState] = {}


def get_state(name: str = "main") -> PageState:
    if name not in _states:
        _states[name] = PageState()
    return _states[name]


def reset_state(name: str = "main") -> None:
    _states.pop(name, None)


# ─────────────────────────────────────────────────────────────────────────────
# JS helpers — injected into evaluate calls
# ─────────────────────────────────────────────────────────────────────────────

# Snapshot JS: walks the page (without shadow DOM piercing for speed; we
# only need a fingerprint). Returns the raw fields we hash on the Python side.
_SNAPSHOT_JS = r"""
() => {
    // Active modal — simple, no shadow pierce (fast path for snapshots)
    function activeModals() {
        const sels = ['[role="dialog"]:not([aria-hidden="true"])',
                      '[role="alertdialog"]:not([aria-hidden="true"])'];
        const out = [];
        for (const s of sels) {
            for (const el of document.querySelectorAll(s)) {
                const r = el.getBoundingClientRect();
                if (r.width === 0 || r.height === 0) continue;
                // Filter phantom modals: must be visibly rendered (offsetParent set,
                // not display:none, not opacity:0). Catches hidden "save draft"
                // dialogs LinkedIn keeps in the DOM but doesn't show.
                if (el.offsetParent === null) continue;
                const style = getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden') continue;
                if (parseFloat(style.opacity) < 0.1) continue;
                const titleEl = el.querySelector('h1,h2,h3,[role="heading"]');
                out.push((titleEl && titleEl.textContent || '').trim().slice(0, 80));
            }
        }
        return out;
    }

    // Toast / alert / error messages currently visible
    function toasts() {
        const sels = ['[role="alert"]', '[role="status"]',
                      '.toast', '.notification', '.snackbar',
                      '[class*="error-message"]', '[class*="ErrorMessage"]'];
        const out = [];
        for (const s of sels) {
            for (const el of document.querySelectorAll(s)) {
                const r = el.getBoundingClientRect();
                if (r.width === 0 || r.height === 0) continue;
                const t = (el.innerText || el.textContent || '').trim();
                if (t && t.length < 300) out.push(t.slice(0, 200));
            }
        }
        return [...new Set(out)].slice(0, 5);
    }

    // Count interactive elements + form completion status
    function formStats() {
        const inputs = document.querySelectorAll(
            'input:not([type="hidden"]):not([disabled]),select:not([disabled]),textarea:not([disabled])'
        );
        let filled = 0;
        for (const el of inputs) {
            const v = el.value;
            if (el.type === 'checkbox' || el.type === 'radio') {
                if (el.checked) filled++;
            } else if (v && String(v).trim().length > 0) {
                filled++;
            }
        }
        const interactive = document.querySelectorAll(
            'button,a[href],input,select,textarea,[role="button"],[role="link"],[role="tab"],[role="menuitem"]'
        ).length;
        return { total: inputs.length, filled, interactive };
    }

    // Primary action: the most prominent enabled button in the lower half / modal
    function primaryAction() {
        const modal = document.querySelector('[role="dialog"]:not([aria-hidden="true"])');
        const root = modal || document;
        const buttons = [...root.querySelectorAll('button:not([disabled]),[role="button"]:not([aria-disabled="true"])')];
        let best = null, bestScore = -Infinity;
        for (const b of buttons) {
            const r = b.getBoundingClientRect();
            if (r.width === 0 || r.height === 0) continue;
            const txt = (b.innerText || b.textContent || '').trim();
            if (!txt || txt.length > 40) continue;
            // Score: lower-positioned + larger + has positive verb
            const verbBonus = /\b(submit|continue|next|apply|send|save|confirm|done|review)\b/i.test(txt) ? 50 : 0;
            const score = (r.top + r.height) * 0.1 + r.width * 0.2 + verbBonus;
            if (score > bestScore) { bestScore = score; best = txt; }
        }
        return best;
    }

    const body = document.body || document.documentElement;
    const text = (body.innerText || body.textContent || '').replace(/\s+/g, ' ').trim();
    const stats = formStats();

    return {
        url: location.href,
        title: document.title || '',
        text: text.slice(0, 50000),    // capped for hashing; full body is too big
        text_len: text.length,
        modal_titles: activeModals(),
        toasts: toasts(),
        form_fields_total: stats.total,
        form_fields_filled: stats.filled,
        interactive_count: stats.interactive,
        primary_action: primaryAction(),
    };
}
"""

# Diagnose JS: given criteria (selector, text, ariaLabel, role), figure out
# WHY the action failed. Returns a structured reason + suggestions.
_DIAGNOSE_JS = r"""
({ selector, text, ariaLabel, role }) => {
    function visible(el) {
        const r = el.getBoundingClientRect();
        const s = getComputedStyle(el);
        return r.width > 0 && r.height > 0 &&
               s.display !== 'none' && s.visibility !== 'hidden' &&
               parseFloat(s.opacity) > 0;
    }

    function inViewport(el) {
        const r = el.getBoundingClientRect();
        return r.top >= 0 && r.left >= 0 &&
               r.bottom <= (window.innerHeight || 0) &&
               r.right  <= (window.innerWidth  || 0);
    }

    function describe(el) {
        if (!el) return null;
        const r = el.getBoundingClientRect();
        return {
            tag: el.tagName.toLowerCase(),
            text: (el.innerText || el.textContent || '').trim().slice(0, 80),
            ariaLabel: el.getAttribute('aria-label'),
            disabled: el.disabled || el.getAttribute('aria-disabled') === 'true',
            rect: { x: Math.round(r.x), y: Math.round(r.y),
                    w: Math.round(r.width), h: Math.round(r.height) },
            visible: visible(el),
            inViewport: inViewport(el),
        };
    }

    // Try to resolve the criteria to an element (same priority as dom_smart_click)
    let el = null;
    if (selector) {
        try { el = document.querySelector(selector); } catch(_) {}
    }
    if (!el && ariaLabel) {
        const needle = ariaLabel.toLowerCase();
        for (const e of document.querySelectorAll('[aria-label]')) {
            if ((e.getAttribute('aria-label') || '').toLowerCase().includes(needle)) {
                el = e; break;
            }
        }
    }
    if (!el && role && text) {
        const needle = text.toLowerCase();
        for (const e of document.querySelectorAll('[role="' + role + '"]')) {
            if (((e.innerText || e.textContent || '').toLowerCase()).includes(needle)) {
                el = e; break;
            }
        }
    }
    if (!el && text) {
        const needle = text.toLowerCase();
        const tags = ['button','a','span','div','li','label','input','[role="button"]'];
        const all = [...document.querySelectorAll(tags.join(','))];
        el = all.find(e => ((e.innerText || e.textContent || '').toLowerCase()).includes(needle)) || null;
    }

    if (!el) {
        // Find closest text matches to suggest
        const all = [...document.querySelectorAll('button,a,[role="button"],[role="link"]')];
        const labels = all.map(e => (e.innerText || e.textContent || '').trim())
                          .filter(t => t && t.length < 60);
        const needle = (text || ariaLabel || '').toLowerCase();
        let closest = [];
        if (needle) {
            closest = [...new Set(labels)]
                .filter(t => t.toLowerCase().includes(needle.slice(0, 4)))
                .slice(0, 5);
            if (closest.length === 0) {
                // Fall back to first N enabled button labels — gives the agent a clue
                closest = [...new Set(labels)].slice(0, 8);
            }
        }
        return {
            reason: 'not_found',
            message: 'No element matches the given criteria.',
            closest_labels: closest,
            suggestion: 'Call dom_scan() and pick a matching label from the results.',
        };
    }

    const d = describe(el);

    if (!d.visible) {
        const s = getComputedStyle(el);
        let why = 'invisible';
        if (s.display === 'none') why = 'display:none';
        else if (s.visibility === 'hidden') why = 'visibility:hidden';
        else if (parseFloat(s.opacity) === 0) why = 'opacity:0';
        else if (d.rect.w === 0 || d.rect.h === 0) why = 'zero-size';
        return {
            reason: 'invisible',
            message: 'Element exists but is hidden (' + why + ').',
            element: d,
            suggestion: 'Wait for it to render (dom_await_element) or check if a parent is collapsed.',
        };
    }

    if (d.disabled) {
        return {
            reason: 'disabled',
            message: 'Element exists and is visible, but is disabled.',
            element: d,
            suggestion: 'A required field is likely empty — fill remaining fields first.',
        };
    }

    if (!d.inViewport) {
        return {
            reason: 'off_screen',
            message: 'Element is rendered but outside the viewport.',
            element: d,
            suggestion: 'It will be scrolled into view automatically. If still failing, an overlay may be intercepting.',
        };
    }

    // Check if covered
    const r = el.getBoundingClientRect();
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;
    const top = document.elementFromPoint(cx, cy);
    if (top && top !== el && !el.contains(top) && !top.contains(el)) {
        const td = describe(top);
        return {
            reason: 'covered',
            message: 'Element is covered by another element at its center point.',
            element: d,
            covered_by: td,
            suggestion: 'Dismiss the overlay (often a cookie banner or modal backdrop) before retrying.',
        };
    }

    return {
        reason: 'unknown',
        message: 'Element appears clickable but action did not succeed.',
        element: d,
        suggestion: 'The click may have fired but had no effect. Try dom_understand() to see current page state.',
    };
}
"""

# Understand JS: high-level semantic summary
_UNDERSTAND_JS = r"""
() => {
    function visibleText(el) {
        return (el.innerText || el.textContent || '').trim();
    }

    // Detect page type heuristically
    function pageType() {
        const url = location.href;
        const t = document.title.toLowerCase();
        const hasForm = document.querySelectorAll('form input:not([type="hidden"])').length > 2;
        const modal = document.querySelector('[role="dialog"]:not([aria-hidden="true"])');
        if (modal) return 'modal_open';
        if (/login|sign[- ]?in/.test(t) || document.querySelector('input[type="password"]')) return 'login';
        if (/signup|sign[- ]?up|register/.test(t)) return 'signup';
        if (/checkout|payment|cart/.test(t)) return 'checkout';
        if (hasForm) return 'form';
        const articleLike = document.querySelector('article,[role="article"]') ||
                            document.querySelectorAll('p').length > 8;
        if (articleLike) return 'article';
        const listLike = document.querySelectorAll('[role="list"],ul li a,ol li a').length > 15;
        if (listLike) return 'list';
        return 'unknown';
    }

    // Fields with their fill state
    function fieldSummary() {
        const inputs = [...document.querySelectorAll(
            'input:not([type="hidden"]):not([disabled]),select:not([disabled]),textarea:not([disabled])'
        )];
        return inputs.slice(0, 30).map(el => {
            // Find a label
            let lbl = null;
            if (el.id) {
                const lblEl = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
                if (lblEl) lbl = visibleText(lblEl);
            }
            lbl = lbl || el.getAttribute('aria-label') || el.placeholder ||
                  (el.name ? el.name : null) || el.type || 'unlabelled';
            const v = el.value;
            const filled = (el.type === 'checkbox' || el.type === 'radio')
                ? el.checked
                : (v && String(v).trim().length > 0);
            return {
                label: lbl.slice(0, 50),
                tag: el.tagName.toLowerCase(),
                type: el.type || null,
                required: el.required || el.getAttribute('aria-required') === 'true',
                filled,
                value: filled && el.type !== 'password' ? String(v).slice(0, 40) : null,
            };
        });
    }

    // Primary + secondary actions
    function actions() {
        const modal = document.querySelector('[role="dialog"]:not([aria-hidden="true"])');
        const root = modal || document;
        const buttons = [...root.querySelectorAll('button,[role="button"]')]
            .map(b => {
                const r = b.getBoundingClientRect();
                const txt = visibleText(b);
                return {
                    label: txt.slice(0, 40),
                    disabled: b.disabled || b.getAttribute('aria-disabled') === 'true',
                    rect_y: r.top, rect_h: r.height,
                };
            })
            .filter(b => b.label && b.rect_h > 0 && b.label.length < 40);
        // Sort by vertical position (later = lower = more likely a submit)
        buttons.sort((a, b) => b.rect_y - a.rect_y);
        return buttons.slice(0, 8);
    }

    const modal = document.querySelector('[role="dialog"]:not([aria-hidden="true"])');
    const modalTitle = modal
        ? (modal.querySelector('h1,h2,h3,[role="heading"]')?.textContent || '').trim()
        : null;

    return {
        url: location.href,
        title: document.title,
        page_type: pageType(),
        modal_open: !!modal,
        modal_title: modalTitle,
        fields: fieldSummary(),
        actions: actions(),
    };
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Python-side API
# ─────────────────────────────────────────────────────────────────────────────

async def snapshot(frame) -> StateSnapshot:
    """Take a snapshot of the current page state."""
    try:
        raw = await frame.evaluate(_SNAPSHOT_JS)
    except Exception as e:
        log.debug("snapshot failed: %s", e)
        return StateSnapshot()

    text = raw.get("text", "")
    text_hash = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    return StateSnapshot(
        url=raw.get("url", ""),
        title=raw.get("title", ""),
        text_hash=text_hash,
        text_len=raw.get("text_len", 0),
        modal_count=len(raw.get("modal_titles", [])),
        modal_titles=raw.get("modal_titles", []),
        toasts=raw.get("toasts", []),
        form_fields_total=raw.get("form_fields_total", 0),
        form_fields_filled=raw.get("form_fields_filled", 0),
        interactive_count=raw.get("interactive_count", 0),
        primary_action=raw.get("primary_action"),
    )


def diff(before: StateSnapshot, after: StateSnapshot) -> Dict[str, Any]:
    """Compute a structured diff between two snapshots.

    Returns:
      {
        "changed": ["url", "modals", ...],   # list of changed keys
        "url_changed": True/False,
        "new_modal": "<title>" or None,
        "modal_closed": True/False,
        "new_toasts": [...],
        "text_changed": True/False,
        "fields_filled_delta": int,
      }
    """
    changed: List[str] = []

    if before.url != after.url:
        changed.append("url")
    if before.title != after.title:
        changed.append("title")
    if before.text_hash != after.text_hash:
        changed.append("text")
    if before.modal_count != after.modal_count:
        changed.append("modals")
    if before.form_fields_filled != after.form_fields_filled:
        changed.append("fields_filled")
    if set(before.toasts) != set(after.toasts):
        changed.append("toasts")

    new_modal = None
    modal_closed = False
    if after.modal_count > before.modal_count:
        # New modal — return its title
        new_titles = [t for t in after.modal_titles if t not in before.modal_titles]
        new_modal = new_titles[0] if new_titles else "(untitled modal)"
    elif after.modal_count < before.modal_count:
        modal_closed = True

    new_toasts = [t for t in after.toasts if t not in before.toasts]

    return {
        "changed": changed,
        "url_changed": before.url != after.url,
        "new_modal": new_modal,
        "modal_closed": modal_closed,
        "new_toasts": new_toasts,
        "text_changed": before.text_hash != after.text_hash,
        "fields_filled_delta": after.form_fields_filled - before.form_fields_filled,
        "primary_action": after.primary_action,
    }


async def diagnose(frame, criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Run a diagnostic pass to explain why an action targeting `criteria` failed.

    criteria: {selector, text, ariaLabel, role} — same keys as dom_smart_click args.

    Returns: {reason, message, element?, closest_labels?, covered_by?, suggestion}
    """
    try:
        return await frame.evaluate(_DIAGNOSE_JS, {
            "selector": criteria.get("selector"),
            "text": criteria.get("text"),
            "ariaLabel": criteria.get("aria_label") or criteria.get("ariaLabel"),
            "role": criteria.get("role"),
        })
    except Exception as e:
        return {"reason": "diagnose_failed", "message": str(e)}


async def understand(frame) -> Dict[str, Any]:
    """Return a semantic summary of the page."""
    try:
        return await frame.evaluate(_UNDERSTAND_JS)
    except Exception as e:
        return {"error": str(e)}


def summarize_understanding(u: Dict[str, Any]) -> str:
    """Compress an understand() dict into a short natural-language paragraph."""
    if "error" in u:
        return f"(could not analyze page: {u['error']})"

    parts: List[str] = []
    pt = u.get("page_type", "unknown")
    title = u.get("title", "").strip()

    if u.get("modal_open"):
        parts.append(f"Modal open: {u.get('modal_title') or '(untitled)'}")
    else:
        parts.append(f"Page type: {pt}" + (f" — {title}" if title else ""))

    fields = u.get("fields", [])
    if fields:
        filled = sum(1 for f in fields if f.get("filled"))
        required_unfilled = [f["label"] for f in fields
                             if f.get("required") and not f.get("filled")]
        parts.append(f"Fields: {filled}/{len(fields)} filled")
        if required_unfilled:
            parts.append("Required & empty: " + ", ".join(required_unfilled[:5]))

    actions = u.get("actions", [])
    if actions:
        enabled = [a["label"] for a in actions if not a.get("disabled")]
        disabled = [a["label"] for a in actions if a.get("disabled")]
        if enabled:
            parts.append("Available actions: " + ", ".join(enabled[:5]))
        if disabled:
            parts.append("Disabled: " + ", ".join(disabled[:3]))

    return ". ".join(parts) + "."
