<p align="center">
  <img src="logo.png" alt="Orbit logo">
</p>


# Orbit

**Automate any app — desktop or browser — with composable, typed steps.**

Natural language controls the screen. Python controls the flow.

```bash
pip install orbit-cua
```


<p align="center">
  <a href="https://youtu.be/nll7Mmzwh00">
    <img src="demo_preview.svg" width="720" alt="Watch Orbit in action">
  </a>
</p>


```python
from pydantic import BaseModel
import orbit2 as orbit

class Stories(BaseModel):
    titles: list[str]

async with orbit.session(llm="gemini/gemini-3.6-flash") as s:
    await s.navigate("https://news.ycombinator.com")

    stories = await s.read("the top 5 story titles", schema=Stories)

    if await s.check("a login link is visible"):
        await s.do("click the login link")
```

Structured extraction into Pydantic models. Real conditionals. Real control flow. No prompt soup.

## Why

Agents loop, click the wrong thing, and can't be steered. Orbit splits execution into independent steps — **do · read · check · navigate · fill** — each with its own model, its own step budget, and typed output.

- **Goes where browser agents can't.** One workflow can read a website, drive a native desktop app, and come back.
- **Verifies every action.** If a click didn't actually change anything, Orbit notices and tries another way instead of confidently marching on.
- **Sees any app.** Cooperative apps are read through the accessibility tree and the live DOM. Canvas apps, custom-drawn UI and remote desktops fall back to pixels automatically.
- **Cheap model for clicks, powerful model for reasoning** — set `llm=` per step.
- **Hard step caps.** Nothing runs forever.

## Beyond the screen

Orbit ships with tools for the work surrounding a UI task, so one run does the whole job:

```python
async with orbit.session(llm="gemini/gemini-3.6-flash") as s:
    await s.navigate("https://news.ycombinator.com")
    await s.do("read the top 5 stories and save them to stories.csv")
```

Files (`read_file`, `write_file`, `read_csv`, `write_csv`, `find_files`), code (`run_python`, `run_command`), clipboard, and web (`web_search`, `fetch_url`) are available to the agent by default.

## Bring your own tools

```python
from orbit2.tools import tool

@tool("post_to_slack", "Send a message to Slack", {"text": {"type": "string"}})
async def post_to_slack(text: str) -> str:
    ...
    return "sent"

async with orbit.session(llm="gemini/gemini-3.6-flash", tools=[post_to_slack]) as s:
    await s.do("summarize this dashboard and post it to Slack")
```

## Simplest possible use

```python
async with orbit.session(llm="gemini/gemini-3.6-flash") as s:
    await s.do("open the calculator and compute 7 times 8")
```

Any model works via LiteLLM: set `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`.

## Verbs

| verb | what it does |
|---|---|
| `s.do(task)` | perform an action, stop when done |
| `s.read(task, schema=Model)` | extract typed data — returns a validated Pydantic instance |
| `s.check(condition)` | returns a real `bool`, for `if` statements |
| `s.navigate(target)` | open a URL or launch an app |
| `s.fill(form, data)` | fill a form from a dict |

Every call takes `llm=`, `max_steps=`, `guidance=` and `timeout=` overrides.

Results are typed: `result.ok`, `result.output`, `result.status`, `result.steps_used`, and `result.journal` — a full record of what happened.

## Learn more

- [`examples/`](examples/) for full workflows
- [`docs/`](docs/) for the SDK reference and platform support (Windows, Linux, macOS, Python 3.10+)

Respect the terms of service of whatever you automate.

Apache 2.0 · Thanks to OculOS
