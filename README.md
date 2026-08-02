<p align="center">
  <img src="logo.png" alt="Orbit logo">
</p>


# Orbit

**Automate any desktop or browser app with composable, typed steps.**

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
from orbit import Do, Read, Check, Navigate, session

async with session() as s:
    await Navigate("https://news.ycombinator.com", session=s, llm=model).run()

    stories = await Read("top 5 stories", schema=StoryList, session=s, llm=model).run()

    if await Check("a login button is visible", session=s, llm=model).check():
        await Do("click the login button", session=s, llm=model).run()
```

Structured extraction into Pydantic models. Real conditionals. Real control flow. No prompt soup.

## Why

Agents loop, click the wrong thing, and can't be steered. Orbit fixes this by splitting execution into independent steps (**Do · Read · Check · Navigate · Fill · Bootstrap**), each with its own model, its own call budget, and typed output.

- Cheap model for clicks, powerful model for reasoning
- Hard `max_steps` cap per step, nothing runs forever
- `planner=False` for low-latency direct execution
- Inject guidance mid-run when the agent struggles

And instead of pixels, Orbit reads the OS accessibility tree and the live DOM. Screenshots only when needed. Works across desktop apps, browsers, and Electron apps.

## Simplest possible use

```python
from orbit import Agent

result = await Agent(task="Open Chrome and go to Wikipedia", llm="gemini-3-pro-preview").run()
```

Any model works via LiteLLM: set `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`.

## Learn more

- [`examples/`](examples/) for full workflows, custom actions (`BaseActionAgent`), and environment setup with `Bootstrap`
- [`docs/`](docs/) for the full SDK reference, safety model, and platform support (Windows, Linux, macOS, Python 3.10+)

Safety defaults: no permanent file deletion, disk writes require human approval, every step is budget-capped. Respect the terms of service of what you automate.

Apache 2.0 · Thanks to OculOS
