import asyncio
import subprocess
import time
import requests
import atexit
import os
from pathlib import Path


class OculOSManager:
    def __init__(
        self,
        binary_path: str = r"C:\Users\aadya\Desktop\oculos\target\release\oculos.exe",
        verbose: bool = False,
    ):
        """Manages the lifecycle of the OculOS background daemon."""
        if binary_path is None:
            # Default to bundled binary inside the orbit package
            here = Path(__file__).resolve().parent
            if os.name == "nt":
                candidate = here / "_bin" / "oculos.exe"
            else:
                candidate = here / "_bin" / "oculos"
            binary_path = str(candidate)

        self.binary_path = Path(binary_path).resolve()
        self.process = None
        self.base_url = "http://127.0.0.1:7878"
        self.verbose = verbose

        atexit.register(self.stop)

    async def start(self):
        """Starts the OculOS HTTP server in the background. Await this from an async context."""
        if not self.binary_path.exists():
            raise FileNotFoundError(f"OculOS binary not found at: {self.binary_path}")

        if self.verbose:
            print("Starting OculOS daemon...")

        self.process = subprocess.Popen(
            [str(self.binary_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        await self._wait_for_health_check()

    async def _wait_for_health_check(self, timeout_seconds: int = 5):
        """Polls the HTTP endpoint until the server is ready. No fixed sleep; yield between polls."""
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    if self.verbose:
                        print("OculOS daemon is live and listening on port 7878.")
                    return
            except requests.exceptions.ConnectionError:
                await asyncio.sleep(0)

        self.stop()
        raise TimeoutError("OculOS server failed to start within the timeout period.")

    def stop(self):
        """Terminates the background process."""
        if self.process and self.process.poll() is None:
            if self.verbose:
                print("Shutting down OculOS daemon...")
            self.process.terminate()
            self.process.wait(timeout=3)
            if self.verbose:
                print("Daemon stopped.")
