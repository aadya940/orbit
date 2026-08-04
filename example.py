"""Orbit examples: the composable SDK API.

Run one:
    python example.py simple
    python example.py extract
    python example.py control_flow
    python example.py cross_app
    python example.py tools
    python example.py custom_tool
"""

import asyncio
import sys

from dotenv import load_dotenv
from pydantic import BaseModel

import orbit
from orbit.tools import tool

load_dotenv()

LLM = "gemini/gemini-3.6-flash"


# 1. Simplest possible task ---------------------------------------------------

async def simple():
    """One instruction. The app opens, the work happens, the session cleans up."""
    async with orbit.session(llm=LLM) as s:
        result = await s.do("open the calculator and compute 7 times 8")
        print(result.status, "in", result.steps_used, "steps")


# 2. Typed extraction ---------------------------------------------------------

class Story(BaseModel):
    title: str
    points: int


class FrontPage(BaseModel):
    stories: list[Story]


async def extract():
    """`read` returns a validated Pydantic instance, or the run failed."""
    async with orbit.session(llm=LLM) as s:
        await s.navigate("https://news.ycombinator.com")
        result = await s.read("the top 5 stories with title and points", schema=FrontPage)

        for story in result.output.stories:
            print(f"{story.points:>4}  {story.title}")


# 3. Real control flow --------------------------------------------------------

async def control_flow():
    """`check` returns a real bool, so Python drives the branching."""
    async with orbit.session(llm=LLM) as s:
        await s.navigate("https://news.ycombinator.com")

        if await s.check("a login link is visible"):
            print("not signed in")
        else:
            print("already signed in")


# 4. One workflow across the browser and a desktop app ------------------------

class Contributor(BaseModel):
    username: str


async def cross_app():
    """The web half and the desktop half of a job, in a single session."""
    async with orbit.session(llm=LLM) as s:
        await s.navigate("https://github.com/numpy/numpy/graphs/contributors")
        await asyncio.sleep(3)  # the contributor graph renders late

        top = await s.read("the 2nd-highest contributor's login handle", schema=Contributor)
        user = top.output.username

        # ...now leave the browser entirely
        await s.navigate("gnome-text-editor")
        await asyncio.sleep(2)
        await s.do(f"type this into the document: numpy's #2 contributor is {user}")

        print("wrote to the editor:", await s.check(f"the document contains {user}"))


# 5. Tools alongside the screen ----------------------------------------------

async def tools():
    """Files, code, clipboard and web tools are available by default."""
    async with orbit.session(llm=LLM) as s:
        await s.navigate("https://news.ycombinator.com")
        await s.do("read the top 5 stories and save them to /tmp/stories.csv")

        print(open("/tmp/stories.csv").read())


# 6. Your own tool ------------------------------------------------------------

@tool("notify", "Show a desktop notification", {"message": {"type": "string"}})
async def notify(message: str) -> str:
    proc = await asyncio.create_subprocess_exec("notify-send", "Orbit", message)
    await proc.wait()
    return "notified"


async def custom_tool():
    """Register anything the agent should be able to do."""
    async with orbit.session(llm=LLM, tools=[notify]) as s:
        await s.navigate("https://news.ycombinator.com")
        await s.do("read the top story and send me a desktop notification about it")


EXAMPLES = {
    "simple": simple,
    "extract": extract,
    "control_flow": control_flow,
    "cross_app": cross_app,
    "tools": tools,
    "custom_tool": custom_tool,
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if name not in EXAMPLES:
        print(f"unknown example {name!r}. choose from: {', '.join(EXAMPLES)}")
        sys.exit(1)
    asyncio.run(EXAMPLES[name]())
