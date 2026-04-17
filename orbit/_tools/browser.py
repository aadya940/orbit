import asyncio
import glob
import logging
import os
import shutil
import uuid
from typing import Optional

try:
    from playwright.async_api import async_playwright, BrowserContext, Page

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
    "--disable-blink-features=AutomationControlled",
]

_WEBDRIVER_MASK_SCRIPT = """
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    window.navigator.chrome = { runtime: {} };
    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
"""


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the browser.  Safe to call only once; raises if called again."""
        if not PLAYWRIGHT_AVAILABLE:
            raise BrowserManagerError(
                "Playwright is not installed. Run: pip install playwright && playwright install chromium"
            )

        async with self._lock:
            if self._playwright is not None:
                raise BrowserManagerError(
                    "BrowserManager.start() called while already running."
                )

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
                await context.add_init_script(_WEBDRIVER_MASK_SCRIPT)
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

        # Always attempt profile sync, even if close() raised
        if self._tmp_profile is not None:
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
            log.warning("Ephemeral profile dir missing; skipping sync: %s", tmp_profile)
            return

        _LOCK_FILES = ("SingletonLock", "SingletonCookie", "SingletonSocket")
        log.info(
            "Syncing ephemeral profile → persistent storage (%s)", _PERSISTENT_PROFILE
        )

        try:
            shutil.copytree(
                tmp_profile,
                _PERSISTENT_PROFILE,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(*_LOCK_FILES),
            )
        except Exception as exc:
            log.error("Profile sync failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Global singleton for tools that import this module directly.
#: Prefer using BrowserManager as an async context manager in new code.
global_browser = BrowserManager()
