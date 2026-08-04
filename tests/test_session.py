from pydantic import BaseModel

from conftest import make_obs
from fake_driver import FakeDriver, FakeLLM, FakeScreen
from orbit import Session, RunStatus
from orbit.llm import LLMReply, ToolCall


def tc(name, **args):
    return LLMReply(tool_call=ToolCall(name=name, arguments=args))


def fake_drivers():
    screen = FakeScreen([make_obs("Save"), make_obs("Save", "Done")])
    return {"dom": FakeDriver(screen, name="dom")}


async def test_do():
    llm = FakeLLM([tc("act", kind="click", target="Save"), tc("finish", output="done")])
    async with Session(llm=llm, drivers=fake_drivers()) as s:
        result = await s.do("click save")
    assert result.status is RunStatus.SUCCESS
    assert result.output == "done"
    # verb framing reaches the LLM
    assert any("ACTION: click save" in m.get("content", "") for m in llm.calls[0])


class Title(BaseModel):
    title: str


async def test_read_with_schema():
    llm = FakeLLM([tc("finish", output={"title": "Home"})])
    async with Session(llm=llm, drivers=fake_drivers()) as s:
        result = await s.read("page title", schema=Title)
    assert result.ok and isinstance(result.output, Title)
    assert result.output.title == "Home"


async def test_check_returns_bool():
    llm_true = FakeLLM([tc("finish", output={"result": True})])
    async with Session(llm=llm_true, drivers=fake_drivers()) as s:
        assert await s.check("save button exists") is True

    llm_bad = FakeLLM([tc("finish", output={"nope": 1})])  # never valid -> FAILED
    async with Session(llm=llm_bad, drivers=fake_drivers()) as s:
        assert await s.check("something") is False


async def test_navigate():
    llm = FakeLLM([tc("navigate", value="http://x.com"), tc("finish", output=None)])
    async with Session(llm=llm, drivers=fake_drivers()) as s:
        result = await s.navigate("http://x.com")
    assert result.status is RunStatus.SUCCESS


async def test_fill():
    llm = FakeLLM([
        tc("act", kind="fill", target="Save", value="a@b.com"),
        tc("finish", output="filled"),
    ])
    async with Session(llm=llm, drivers=fake_drivers()) as s:
        result = await s.fill("signup", {"email": "a@b.com"})
    assert result.ok
    assert any("a@b.com" in m.get("content", "") for m in llm.calls[0])


async def test_per_call_max_steps_override():
    llm = FakeLLM([tc("act", kind="click", target="Save")])  # never finishes
    async with Session(llm=llm, drivers=fake_drivers(), max_steps=40) as s:
        result = await s.do("spin", max_steps=3)
    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.steps_used == 3
