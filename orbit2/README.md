# Orbit v2 — clean core

Clean rewrite of Orbit's skeleton with the battle-tested leaf code
transplanted in. `master` keeps v1 intact; this package ships when it
meets or beats v1 on the parity benchmark in `benchmarks/`.

## Design invariants

1. **One interface, many backends.** The model sees ~6 uniform tools.
   Which backend serves them (DOM, accessibility tree, vision, keyboard)
   is the driver layer's runtime decision — never a prompt rule.
2. **Probing replaces assuming.** On first contact with a surface, each
   driver is scored (`CapabilityScore`: element count, label coverage,
   bounds sanity) and the fallback ladder is ordered by evidence.
3. **Verification replaces trusting.** Every action is followed by an
   observe → diff. No diff when one was expected ⇒ the action did not
   land ⇒ escalate the ladder mechanically, without LLM calls.

New-app behavior is principled, not enumerated: special-casing lives in
data (probe cache entries), not code branches or prompt rules.

## Layout

```
orbit2/
├── types.py          # Element/Observation/StateDiff/ActionResult/RunResult + error taxonomy
├── world.py          # per-session container — replaces every v1 global
├── loop.py           # owned observe→decide→act loop (no ADK)
├── llm.py            # thin LiteLLM wrapper behind an LLM protocol
├── session.py        # public API: Session with do/read/check/navigate/fill
├── policy.py         # safety policy + pluggable HITL approver
├── journal.py        # structured, replayable action log
├── drivers/
│   ├── base.py       # Driver protocol + fallback ladder + capability scoring
│   ├── dom.py        # Playwright/CDP (transplanted smart-DOM heuristics)
│   ├── accessibility.py  # OculOS daemon + a-tree (transplanted client)
│   ├── keyboard.py   # last-resort keyboard rung
│   ├── vision.py     # grounding-model rung (stub)
│   └── matching.py   # pure fuzzy target matcher (shared, fully unit-tested)
└── perception/
    └── probe.py      # surface capability probing
```

## Public API

```python
import orbit2

async with orbit2.session(llm="gemini-3-pro-preview") as s:
    await s.navigate("https://news.ycombinator.com")
    stories = await s.read("top 5 stories", schema=StoryList)
    if await s.check("a login button is visible"):
        await s.do("click the login button")
```

- Structured output is validated strictly — `result.output` is a real
  Pydantic instance or the run failed. No silent coercion.
- All failures are typed (`TargetNotFound`, `ActionHadNoEffect`,
  `BudgetExhausted`, `NeedsHuman`, …) and journaled.
- Two sessions in one process are fully isolated: no module globals.

## Testing

`tests2/` runs the entire loop, ladder, budget, and verb layer against a
scripted `FakeDriver` + `FakeLLM` with zero OS/network dependencies:

```
python -m pytest tests2/ -q
```

## Parity gate

`benchmarks/` defines the task suite both versions are scored on.
v2 replaces v1 only when its pass rate meets or beats v1's frozen
baseline. Until then, v1 remains the reference implementation.
