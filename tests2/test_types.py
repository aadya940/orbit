import pytest
from pydantic import BaseModel

from conftest import make_obs
from orbit2.types import OutputInvalid, diff_observations, validate_output


def test_diff_appear_disappear():
    d = diff_observations(make_obs("Save", "Cancel"), make_obs("Save", "Submit"))
    assert d.changed
    assert any("Submit" in k for k in d.appeared)
    assert any("Cancel" in k for k in d.disappeared)


def test_diff_url_change():
    d = diff_observations(make_obs("A", url="http://a"), make_obs("A", url="http://b"))
    assert d.changed and d.url_changed


def test_diff_value_change():
    b = make_obs("Email", values={"Email": ""})
    a = make_obs("Email", values={"Email": "x@y.com"})
    d = diff_observations(b, a)
    assert d.changed and d.value_changes


def test_diff_modal_change():
    d = diff_observations(make_obs("A", modal_count=0), make_obs("A", modal_count=1))
    assert d.changed and d.new_modals == 1
    d2 = diff_observations(make_obs("A", modal_count=1), make_obs("A", modal_count=0))
    assert d2.closed_modals == 1


def test_diff_no_change():
    d = diff_observations(make_obs("A", url="u"), make_obs("A", url="u"))
    assert not d.changed
    assert d.summary() == "no observable change"


class Out(BaseModel):
    name: str
    count: int


def test_validate_valid_dict():
    out = validate_output({"name": "x", "count": 2}, Out)
    assert isinstance(out, Out) and out.count == 2


def test_validate_wrong_shape_raises():
    with pytest.raises(OutputInvalid):
        validate_output({"name": "x"}, Out)
    with pytest.raises(OutputInvalid):
        validate_output({"name": "x", "count": "not-an-int-at-all"}, Out)
    with pytest.raises(OutputInvalid):
        validate_output(42, Out)


def test_validate_no_schema_passthrough():
    assert validate_output({"a": 1}, None) == {"a": 1}
