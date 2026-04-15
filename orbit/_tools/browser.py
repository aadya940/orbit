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

        # Clean up stray SingletonLock if it exists from a previous crash
        import os
        import shutil
        lock_file = "/root/.config/google-chrome/SingletonLock"
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                log.info(f"Removed stale {lock_file}")
            except Exception as e:
                log.warning(f"Failed to remove {lock_file}: {e}")

        # In Docker, we map this directly. In normal usage, default local chrome profile
        # Use existing context to persist login and cache state.
        self.browser_context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir="/root/.config/google-chrome",
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
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        self.active_page = None

# Export a global singleton for tools to use
global_browser = BrowserManager()
