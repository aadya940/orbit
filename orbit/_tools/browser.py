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
            "--start-maximized"
        ]
        
        # In Docker, we map this directly. In normal usage, default local chrome profile
        # Use existing context to persist login and cache state.
        self.browser_context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir="/root/.config/google-chrome",
            headless=False,
            args=flags,
            ignore_https_errors=True,
            no_viewport=True  # Forces chromium DOM viewport to align with OS Window (for CUA parity)
        )
        
        # Provide an empty starting page
        if len(self.browser_context.pages) > 0:
            self.active_page = self.browser_context.pages[0]
        else:
            self.active_page = await self.browser_context.new_page()

    async def stop(self) -> None:
        if self.browser_context:
            await self.browser_context.close()
        if self.playwright:
            await self.playwright.stop()

# Export a global singleton for tools to use
global_browser = BrowserManager()
