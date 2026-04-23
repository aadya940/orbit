import asyncio
import glob
import logging
import os
import shutil
import uuid
from typing import Optional

try:
    from patchright.async_api import async_playwright, BrowserContext, Page

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

log = logging.getLogger("orbit.browser")

_PERSISTENT_PROFILE = "/root/.config/google-chrome"
_TMP_PROFILE_GLOB = "/tmp/chrome-profile-*"
_TMP_PROFILE_PREFIX = "/tmp/chrome-profile-"

_CHROME_FLAGS = [
    "--force-renderer-accessibility",
    "--enable-accessibility",
    "--disable-gpu",
    "--no-sandbox",
    "--disable-dev-shm-usage",  # Docker's /dev/shm is 64MB by default; Chrome will OOM-crash without this
    "--window-size=1280,768",  # explicit size; --start-maximized is unreliable without a real WM
    # "--disable-blink-features=AutomationControlled" removed — patchright patches this at driver level
]


class BrowserManagerError(RuntimeError):
    """Raised when BrowserManager encounters an unrecoverable error."""


class BrowserManager:
    """
    Manages a single persistent-profile Chromium browser context via Playwright.

    Lifecycle
    ---------
    - Call `await start()` (or use as an async context manager) before any page ops.
    - Call `await stop()` (or exit the context manager) to close the browser and sync
      the ephemeral profile back to persistent storage.

    Ephemeral profile pattern
    -------------------------
    The persistent Chrome profile on disk is *copied* to a temp dir on each start so
    that the source volume is never locked or corrupted by a running browser process.
    On clean shutdown the temp copy is synced back.  On unclean shutdown the stale
    temp dirs are removed on the next `start()`.
    """

    def __init__(self) -> None:
        self._playwright = None
        self._browser_context: Optional["BrowserContext"] = None
        self._active_page: Optional["Page"] = None
        self.active_frame = None
        self._tmp_profile: Optional[str] = None
        self._lock = asyncio.Lock()
        self._cdp_connected: bool = False  # True when attached via CDP (skip profile sync)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self, *, purge_stale: bool = True) -> None:
        """Start the browser.  Safe to call only once; raises if called again.

        purge_stale: When True (default), removes leftover ephemeral profiles
            from previously aborted runs.  Pass False when launching a second
            browser alongside one that is already running, to avoid nuking the
            live profile of the first instance.
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise BrowserManagerError(
                "Patchright is not installed. Run: pip install patchright && patchright install chromium"
            )

        async with self._lock:
            if self._playwright is not None:
                raise BrowserManagerError(
                    "BrowserManager.start() called while already running."
                )

            if purge_stale:
                self._purge_stale_tmp_profiles()
            tmp_profile = self._prepare_tmp_profile()

            try:
                playwright = await async_playwright().start()
                context = await playwright.chromium.launch_persistent_context(
                    user_data_dir=tmp_profile,
                    executable_path="/usr/bin/google-chrome",
                    headless=False,
                    args=_CHROME_FLAGS,
                    ignore_default_args=["--enable-automation"],
                    ignore_https_errors=True,
                    no_viewport=True,
                )
                # No add_init_script needed — patchright patches navigator.webdriver at driver level
            except Exception:
                # Clean up the temp profile so we don't leak it
                shutil.rmtree(tmp_profile, ignore_errors=True)
                raise

            self._playwright = playwright
            self._browser_context = context
            self._tmp_profile = tmp_profile
            log.info("BrowserManager started (profile: %s)", tmp_profile)

    async def stop(self) -> None:
        """Close the browser and sync the profile back to persistent storage."""
        async with self._lock:
            await self._shutdown()

    async def ensure_active_page(self) -> "Page":
        """
        Return the active page, creating one if needed.
        Starts the browser automatically if it has not been started yet.
        """
        async with self._lock:
            if self._playwright is None:
                # Release lock before calling start() which also acquires it
                pass

        # Start outside the lock so start()'s own lock acquisition works
        if self._playwright is None:
            await self.start()

        async with self._lock:
            return await self._get_or_create_page()

    @property
    def active_page(self) -> Optional["Page"]:
        """The current active page, or None if not yet created."""
        return self._active_page

    @property
    def active_frame_or_page(self):
        """Returns the active frame (if set) or the active page."""
        return getattr(self, "active_frame", None) or self._active_page

    # ------------------------------------------------------------------
    # Async context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BrowserManager":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_or_create_page(self) -> "Page":
        """Must be called with self._lock held."""
        if self._active_page is not None and not self._active_page.is_closed():
            try:
                await self._active_page.bring_to_front()
            except:
                pass
            return self._active_page

        pages = self._browser_context.pages
        # Sometimes Playwright crashes if we immediately close pages just launched
        # It's safer to just skip about:blank tabs
        valid_pages = [p for p in pages if p.url != "about:blank"]

        if valid_pages:
            self._active_page = valid_pages[0]
        elif pages:
            self._active_page = pages[0]
            # Try to close any extra about:blank tabs left behind by Playwright initialization
            if len(pages) > 1:
                for extr_page in pages[1:]:
                    try:
                        await extr_page.close()
                    except:
                        pass
        else:
            self._active_page = await self._browser_context.new_page()

        try:
            await self._active_page.bring_to_front()
        except:
            pass
        return self._active_page

    async def _shutdown(self) -> None:
        """Core teardown logic.  Must be called with self._lock held."""
        close_error: Optional[BaseException] = None

        if self._browser_context is not None:
            try:
                await self._browser_context.close()
            except Exception as exc:
                log.warning("Error closing browser context: %s", exc)
                close_error = exc
            finally:
                self._browser_context = None
                self._active_page = None

        # Sync profile back only when we own the launched browser (not CDP-attached)
        if self._tmp_profile is not None:
            if not self._cdp_connected:
                self._sync_profile_to_persistent(self._tmp_profile)
            shutil.rmtree(self._tmp_profile, ignore_errors=True)
            self._tmp_profile = None

        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception as exc:
                log.warning("Error stopping Playwright: %s", exc)
            finally:
                self._playwright = None

        log.info("BrowserManager stopped.")

        if close_error is not None:
            raise BrowserManagerError(
                "Browser context did not close cleanly."
            ) from close_error

    # ------------------------------------------------------------------
    # Profile helpers (synchronous – called from async methods)
    # ------------------------------------------------------------------

    @staticmethod
    def _purge_stale_tmp_profiles() -> None:
        """Remove any leftover ephemeral profiles from previously aborted runs."""
        for stale in glob.glob(_TMP_PROFILE_GLOB):
            log.debug("Removing stale ephemeral profile: %s", stale)
            shutil.rmtree(stale, ignore_errors=True)

    @staticmethod
    def _prepare_tmp_profile() -> str:
        """
        Copy the persistent profile to a fresh temp dir (stripping lock files),
        or create an empty dir if no persistent profile exists yet.
        """
        tmp = f"{_TMP_PROFILE_PREFIX}{uuid.uuid4().hex}"

        os.makedirs(os.path.join(tmp, "Default"), exist_ok=True)
        if os.path.exists(_PERSISTENT_PROFILE):
            log.info(
                "Copying credentials from persistent profile → ephemeral storage (%s)",
                tmp,
            )
            for f_name in [
                "Login Data",
                "Login Data-journal",
                "Cookies",
                "Web Data",
                "Web Data-journal",
            ]:
                src = os.path.join(_PERSISTENT_PROFILE, "Default", f_name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(tmp, "Default", f_name))
        else:
            log.info("No persistent profile found; starting fresh (%s)", tmp)
        return tmp

    @staticmethod
    def _sync_profile_to_persistent(tmp_profile: str) -> None:
        """Sync ephemeral profile back to persistent storage."""
        if not os.path.exists(tmp_profile):
            log.warning("Ephemeral profile dir missing; skipping sync: %s", tmp_profile)
            return

        log.info(
            "Syncing ephemeral profile credentials → persistent storage (%s)", _PERSISTENT_PROFILE
        )

        try:
            os.makedirs(os.path.join(_PERSISTENT_PROFILE, "Default"), exist_ok=True)
            for f_name in [
                "Login Data",
                "Login Data-journal",
                "Cookies",
                "Web Data",
                "Web Data-journal",
            ]:
                src = os.path.join(tmp_profile, "Default", f_name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(_PERSISTENT_PROFILE, "Default", f_name))
        except Exception as exc:
            log.error("Profile sync failed: %s", exc)


# ---------------------------------------------------------------------------
# Named browser registry — supports multiple simultaneous Chromium surfaces
# ---------------------------------------------------------------------------

_registry: dict[str, "BrowserManager"] = {}
_active_name: str = "main"


def _active() -> "BrowserManager":
    """Return the currently active BrowserManager, creating a fresh one if needed."""
    if _active_name not in _registry:
        _registry[_active_name] = BrowserManager()
    return _registry[_active_name]


class _ActiveProxy:
    """Proxy that always delegates to the currently active BrowserManager.

    Backward-compatible shim — existing tools that do
    ``from .browser import global_browser`` continue to work unchanged.
    """

    def __getattr__(self, name: str):
        return getattr(_active(), name)

    def __setattr__(self, name: str, value):
        setattr(_active(), name, value)

    async def ensure_active_page(self):
        return await _active().ensure_active_page()


#: Global proxy — always delegates to the active entry in the browser registry.
global_browser = _ActiveProxy()


async def open_browser(name: str = "main") -> "BrowserManager":
    """Launch a new Chrome instance and register it under *name*.

    The newly launched browser immediately becomes the active target for all
    ``dom_*`` tools.  For the first/only browser you don't need to call this —
    ``dom_navigate`` auto-launches on first use.
    """
    global _active_name
    mgr = BrowserManager()
    # Only purge stale ephemeral profiles when no other browser instances are
    # currently running — otherwise we would delete a live profile.
    await mgr.start(purge_stale=not bool(_registry))
    _registry[name] = mgr
    _active_name = name
    log.info("Browser '%s' launched and set as active.", name)
    return mgr


async def connect_browser_cdp(port: int, name: str) -> "BrowserManager":
    """Attach to an externally-running Chromium app on ``localhost:<port>``.

    The target must have been launched with ``--remote-debugging-port=<port>``.
    Common defaults: 9222 (Chrome/Chromium), 9229 (many Electron apps).
    The connected browser is registered under *name* and becomes active.
    """
    global _active_name
    if not PLAYWRIGHT_AVAILABLE:
        raise BrowserManagerError(
            "Patchright is not installed. Run: pip install patchright && patchright install chromium"
        )
    mgr = BrowserManager()
    # Start the Playwright/Patchright engine without launching a browser process.
    mgr._playwright = await async_playwright().start()
    try:
        browser = await mgr._playwright.chromium.connect_over_cdp(f"http://localhost:{port}")
        contexts = browser.contexts
        if not contexts:
            raise BrowserManagerError(
                f"No browser contexts found on CDP port {port}. "
                "Ensure the target app is running and the port is correct."
            )
        mgr._browser_context = contexts[0]
        pages = mgr._browser_context.pages
        mgr._active_page = pages[0] if pages else await mgr._browser_context.new_page()
    except Exception:
        # Clean up the playwright engine so we don't leak it on connection failure
        try:
            await mgr._playwright.stop()
        except Exception:
            pass
        mgr._playwright = None
        raise
    mgr._cdp_connected = True  # skip profile sync on stop()
    _registry[name] = mgr
    _active_name = name
    log.info("CDP browser '%s' connected on port %d and set as active.", name, port)
    return mgr


def switch_active_browser(name: str) -> None:
    """Switch which registered browser all ``dom_*`` tools target.

    Raises ``KeyError`` if *name* has not been registered via
    ``open_browser`` or ``connect_browser_cdp``.
    """
    global _active_name
    if name not in _registry:
        raise KeyError(
            f"No browser registered under '{name}'. "
            "Call dom_open_browser or dom_connect_cdp first."
        )
    _active_name = name
    log.info("Active browser switched to '%s'.", name)
