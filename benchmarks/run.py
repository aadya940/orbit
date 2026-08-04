"""Parity benchmark runner.

Scores an Orbit version against benchmarks/tasks.yaml and writes a
results JSON. Run v1 first to freeze the baseline, then v2 to compare:

    python benchmarks/run.py --engine v1 --llm gemini-3-pro-preview
    python benchmarks/run.py --engine v2 --llm gemini-3-pro-preview
    python benchmarks/run.py --compare

Requires live LLM keys and (for desktop-tier tasks) a desktop session —
this is the integration gate, not part of the unit suite in tests2/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))  # make orbit/orbit importable from anywhere
RESULTS = HERE / "results"


def load_tasks(tiers: list[str] | None = None) -> list[dict]:
    tasks = yaml.safe_load((HERE / "tasks.yaml").read_text())
    if tiers:
        tasks = [t for t in tasks if t["tier"] in tiers]
    return tasks


async def run_task_v2(task: dict, llm: str) -> dict:
    import orbit

    started = time.monotonic()
    try:
        async with orbit.session(llm=llm, max_steps=25) as s:
            if task.get("start_url"):
                await s.navigate(task["start_url"])
            elif task.get("fixture"):
                await s.navigate((HERE / task["fixture"]).resolve().as_uri())
            result = await s.do(task["task"])
            passed = await _grade(s, task, result)
            return _record(task, passed, result.steps_used, started)
    except Exception as exc:  # noqa: BLE001 — benchmark must survive any engine failure
        return _record(task, False, 0, started, error=repr(exc))


async def run_task_v1(task: dict, llm: str) -> dict:
    from orbit import Agent  # v1 reference implementation

    started = time.monotonic()
    try:
        prompt = task["task"]
        if task.get("start_url"):
            prompt = f"Go to {task['start_url']}. Then: {prompt}"
        result = await Agent(task=prompt, llm=llm, max_steps=25).run()
        passed = getattr(result, "status", None) == "success"
        return _record(task, passed, getattr(result, "steps", 0) or 0, started)
    except Exception as exc:  # noqa: BLE001
        return _record(task, False, 0, started, error=repr(exc))


async def _grade(s, task: dict, result) -> bool:
    success = task["success"]
    kind = success["kind"]
    if not result.ok:
        return False
    if kind == "schema_output":
        items = getattr(result.output, "__iter__", None)
        return result.output is not None
    if kind == "url_contains":
        return await s.check(f"the current page url contains '{success['value']}'")
    if kind == "check":
        return await s.check(success["condition"])
    raise ValueError(f"unknown success kind {kind}")


def _record(task: dict, passed: bool, steps: int, started: float, error: str | None = None) -> dict:
    return {
        "id": task["id"],
        "tier": task["tier"],
        "passed": passed,
        "steps": steps,
        "seconds": round(time.monotonic() - started, 1),
        "error": error,
    }


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["v1", "v2"])
    p.add_argument("--llm", default="gemini-3-pro-preview")
    p.add_argument("--tiers", nargs="*", default=None)
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()

    RESULTS.mkdir(exist_ok=True)

    if args.compare:
        rows = {}
        for engine in ("v1", "v2"):
            path = RESULTS / f"{engine}.json"
            if path.exists():
                rows[engine] = json.loads(path.read_text())
        if len(rows) < 2:
            print("need both results/v1.json and results/v2.json — run with --engine first")
            return 1
        for engine, res in rows.items():
            rate = sum(r["passed"] for r in res) / len(res)
            print(f"{engine}: {rate:.0%} ({sum(r['passed'] for r in res)}/{len(res)})")
        v1 = sum(r["passed"] for r in rows["v1"])
        v2 = sum(r["passed"] for r in rows["v2"])
        print("PARITY GATE:", "PASS — v2 may ship" if v2 >= v1 else "FAIL — v2 not ready")
        return 0 if v2 >= v1 else 1

    if not args.engine:
        p.error("--engine or --compare required")
    runner = run_task_v2 if args.engine == "v2" else run_task_v1
    results = []
    for task in load_tasks(args.tiers):
        print(f"[{task['tier']}] {task['id']} ...", end=" ", flush=True)
        r = await runner(task, args.llm)
        print("PASS" if r["passed"] else f"FAIL {r.get('error') or ''}")
        results.append(r)
    out = RESULTS / f"{args.engine}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
