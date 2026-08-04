"""Tool layer: file/code/registry behavior (no network, no screen)."""

import json

import pytest

from orbit2.tools import (
    Tool,
    append_file,
    build_registry,
    default_tools,
    find_files,
    list_dir,
    read_csv,
    read_file,
    run_python,
    tool,
    write_csv,
    write_file,
)


async def test_write_then_read_roundtrip(tmp_path):
    p = tmp_path / "notes.txt"
    out = await write_file.call(path=str(p), content="hello orbit")
    assert "wrote" in out
    assert await read_file.call(path=str(p)) == "hello orbit"


async def test_append_creates_and_appends(tmp_path):
    p = tmp_path / "sub" / "log.txt"
    await append_file.call(path=str(p), content="a")
    await append_file.call(path=str(p), content="b")
    assert await read_file.call(path=str(p)) == "ab"


async def test_read_missing_file_is_a_message_not_a_crash(tmp_path):
    out = await read_file.call(path=str(tmp_path / "nope.txt"))
    assert "no such file" in out


async def test_list_and_find(tmp_path):
    (tmp_path / "a.csv").write_text("x")
    (tmp_path / "deep").mkdir()
    (tmp_path / "deep" / "b.csv").write_text("y")
    listing = await list_dir.call(path=str(tmp_path))
    assert "a.csv" in listing and "deep" in listing
    found = await find_files.call(pattern="**/*.csv", path=str(tmp_path))
    assert "a.csv" in found and "b.csv" in found


async def test_csv_roundtrip(tmp_path):
    p = tmp_path / "rows.csv"
    rows = [{"name": "ada", "n": "1"}, {"name": "grace", "n": "2"}]
    await write_csv.call(path=str(p), rows=json.dumps(rows))
    back = json.loads(await read_csv.call(path=str(p)))
    assert back == rows


async def test_run_python_returns_stdout():
    out = await run_python.call(code="print(6*7)")
    assert out.strip() == "42"


async def test_run_python_reports_errors_to_the_model():
    out = await run_python.call(code="raise ValueError('boom')")
    assert "ValueError" in out and "boom" in out


async def test_run_python_times_out():
    out = await run_python.call(code="import time; time.sleep(5)", timeout=1)
    assert "timed out" in out


async def test_output_is_truncated_not_unbounded():
    out = await run_python.call(code="print('x' * 50000)")
    assert "truncated" in out
    assert len(out) < 30_000


# -- registry ---------------------------------------------------------------

def test_default_registry_has_core_tools():
    names = set(build_registry())
    for expected in ("read_file", "write_file", "run_python", "web_search", "fetch_url"):
        assert expected in names


def test_custom_tool_registers_and_overrides():
    @tool("read_file", "custom override", {"path": {"type": "string"}})
    def custom(path: str) -> str:
        return "custom!"

    reg = build_registry([custom])
    assert reg["read_file"].description == "custom override"


def test_can_run_without_default_tools():
    @tool("only_mine", "just this", {})
    def mine() -> str:
        return "ok"

    reg = build_registry([mine], include_defaults=False)
    assert set(reg) == {"only_mine"}


async def test_custom_async_tool_is_callable():
    @tool("double", "double a number", {"n": {"type": "integer"}})
    async def double(n: int) -> str:
        return str(n * 2)

    reg = build_registry([double], include_defaults=False)
    assert await reg["double"].call(n=21) == "42"


def test_tool_schema_shape():
    schema = read_file.schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "read_file"
    assert "path" in fn["parameters"]["properties"]
    assert fn["parameters"]["required"] == ["path"]


async def test_non_string_results_are_json_encoded():
    @tool("rows", "return rows", {})
    def rows():
        return [{"a": 1}]

    assert json.loads(await rows.call()) == [{"a": 1}]
