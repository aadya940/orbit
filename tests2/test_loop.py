from pydantic import BaseModel

from conftest import make_obs
from fake_driver import FakeDriver, FakeLLM, FakeScreen
from orbit2 import loop
from orbit2.llm import LLMReply, ToolCall
from orbit2.types import RunStatus, TargetNotFound
from orbit2.world import World


def tc(name, **args):
    return LLMReply(tool_call=ToolCall(name=name, arguments=args))


def make_world(max_steps=10, behavior=None):
    screen = FakeScreen([make_obs("Save"), make_obs("Save", "Done", url="http://done")])
    driver = FakeDriver(screen, name="dom", behavior=behavior)
    return World(drivers={"dom": driver}, max_steps=max_steps), driver


async def test_simple_do_task_success():
    world, driver = make_world()
    llm = FakeLLM([
        tc("act", kind="click", target="Save"),
        tc("finish", output="clicked"),
    ])
    result = await loop.run("click save", world, llm)
    assert result.status is RunStatus.SUCCESS
    assert result.output == "clicked"
    assert result.steps_used == 2
    assert len(driver.acted) == 1
    assert any(e["kind"] == "action" for e in result.journal)


async def test_budget_exhaustion():
    world, _ = make_world(max_steps=2)
    llm = FakeLLM([tc("act", kind="click", target="Save")])  # never finishes
    result = await loop.run("loop forever", world, llm)
    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.steps_used == 2


class Out(BaseModel):
    title: str


async def test_finish_with_valid_schema_output():
    world, _ = make_world()
    llm = FakeLLM([tc("finish", output={"title": "hello"})])
    result = await loop.run("read title", world, llm, schema=Out)
    assert result.status is RunStatus.SUCCESS
    assert isinstance(result.output, Out) and result.output.title == "hello"


async def test_invalid_output_retried_then_fixed():
    world, _ = make_world()
    llm = FakeLLM([
        tc("finish", output={"wrong": 1}),
        tc("finish", output={"title": "fixed"}),
    ])
    result = await loop.run("read title", world, llm, schema=Out)
    assert result.status is RunStatus.SUCCESS
    assert result.output.title == "fixed"
    # the validation error was fed back to the LLM
    assert any("failed validation" in (m.get("content") or "") for m in llm.calls[1])


async def test_invalid_output_three_times_fails():
    world, _ = make_world()
    llm = FakeLLM([tc("finish", output={"wrong": 1})])  # repeats forever
    result = await loop.run("read title", world, llm, schema=Out)
    assert result.status is RunStatus.FAILED
    assert result.error is not None and result.error.code == "output_invalid"
    assert llm.idx == 3  # initial + 2 retries


async def test_ask_human():
    world, _ = make_world()
    llm = FakeLLM([tc("ask_human", reason="captcha")])
    result = await loop.run("do thing", world, llm)
    assert result.status is RunStatus.NEEDS_HUMAN
    assert any(e["kind"] == "needs_human" for e in result.journal)


async def test_target_unresolvable_fed_back():
    world, _ = make_world(behavior={"click": TargetNotFound("gone")})
    llm = FakeLLM([
        tc("act", kind="click", target="Missing"),
        tc("finish", output="gave up gracefully"),
    ])
    result = await loop.run("click missing", world, llm)
    assert result.status is RunStatus.SUCCESS
    assert result.output == "gave up gracefully"
    assert any("target_unresolvable" in (m.get("content") or "") for m in llm.calls[1])
