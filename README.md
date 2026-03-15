## Orbit

<p>
<img src="logo.png" align="center">
</p>

Orbit is a general-purpose **computer-use agent** designed for ease of use and low token usage.

It can interact with your desktop and browser to:

- Fill forms
- Scrape jobs and collect them into a CSV
- Move/clean files and folders
- Apply to jobs on your behalf

…and much more.

Most agents either take repeated screenshots or paste the entire DOM tree into the LLM. Orbit instead uses the operating system’s accessibility tree, which lets it control both desktop apps and the browser with far less context bloat. When accessibility isn’t enough, it falls back to a vision-based agent.

For filesystem safety, other agents rely on complex mechanisms (virtual filesystems, SQLite-backed FS, WALs, etc.). Orbit takes a simpler approach: it never permanently deletes your files. Destructive operations send files and folders to the system Trash/Recycle Bin, so they remain recoverable.

## Installation

```bash
pip install orbit
```

Here is a minimal example:

```python
from orbit import Agent
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    agent = Agent(
        llm="gemini-3-pro-preview",
        task="Open Chrome and navigate to Wikipedia",
        verbose=False,  # set True to see tool and daemon logs
    )
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
```
