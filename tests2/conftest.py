import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbit2.types import Element, Observation


def make_obs(*names, url=None, title="page", modal_count=0, focused_key=None,
             values=None, surface="browser:main"):
    values = values or {}
    els = [Element(role="button", name=n, value=values.get(n)) for n in names]
    return Observation(
        surface=surface, kind="browser", title=title, url=url,
        elements=els, modal_count=modal_count, focused_key=focused_key,
    )
