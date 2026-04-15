import asyncio
import logging
from typing import Optional

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None

log = logging.getLogger("orbit.browser")

class BrowserManager:
    def __init__(self):
        self.playwright = None
        self.browser_context = None
        self.active_page = None

    async def start(self) -> None:
        if async_playwright is None:
            log.warning("Playwright is not installed. BrowserManager will be disabled.")
            return

        self.playwright = await async_playwright().start()
        
        flags = [
            "--force-renderer-accessibility",
            "--enable-accessibility",
            "--disable-gpu",
            "--no-sandbox",
            "--start-maximized",
            "--disable-blink-features=AutomationControlled"
        ]

        # Ephemeral Profile Copy pattern:
        # We copy the persistent profile to an ephemeral temp dir, stripping any locks.
        # This allows Docker to keep the base volume safe from corruption and file-lock deadlocks.
        import os
        import shutil
        
        self.src_profile = "/root/.config/google-chrome"
        self.tmp_profile = "/tmp/chrome-profile"

        if os.path.exists(self.tmp_profile):
            shutil.rmtree(self.tmp_profile, ignore_errors=True)

        if os.path.exists(self.src_profile):
            log.info("Copying persistent profile to ephemeral storage...")
            shutil.copytree(
                self.src_profile,
                self.tmp_profile,
                ignore=shutil.ignore_patterns("SingletonLock", "SingletonCookie", "SingletonSocket")
            )
        else:
            os.makedirs(self.tmp_profile, exist_ok=True)

        self.browser_context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=self.tmp_profile,
            headless=False,
            args=flags,
            ignore_https_errors=True,
            no_viewport=True  # Forces chromium DOM viewport to align with OS Window (for CUA parity)
        )

        # Mask Playwright WebDriver to prevent bot detection (Concern #2)
        await self.browser_context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.navigator.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        """)

    async def ensure_active_page(self) -> None:
        if self.playwright is None:
            await self.start()
        
        if self.active_page is None or self.active_page.is_closed():
            if self.browser_context.pages:
                self.active_page = self.browser_context.pages[0]
            else:
                self.active_page = await self.browser_context.new_page()

    async def stop(self) -> None:
        if self.browser_context:
            await self.browser_context.close()
            self.browser_context = None
            
            # Sync session changes back to the permanent volume cleanly
            import os
            import shutil
            if hasattr(self, 'tmp_profile') and os.path.exists(self.tmp_profile):
                log.info("Syncing ephemeral profile back to persistent storage...")
                shutil.copytree(
                    self.tmp_profile,
                    self.src_profile,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("SingletonLock", "SingletonCookie", "SingletonSocket")
                )

        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        self.active_page = None

# Export a global singleton for tools to use
global_browser = BrowserManager()
