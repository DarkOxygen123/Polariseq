"""``docs/API.md`` is the code's own reference: current, and complete."""

from __future__ import annotations

from pathlib import Path

import pytest
from polariseq._api_reference import reference

API = Path(__file__).resolve().parents[3] / "docs" / "API.md"


def test_every_argument_is_described() -> None:
    text = reference()
    empty = [line for line in text.splitlines() if line.startswith("| `") and line.rstrip().endswith("|  |")]
    assert not empty, "arguments with no description (add them to the docstring's Args:):\n" + "\n".join(empty)


@pytest.mark.skipif(not API.exists(), reason="docs/API.md is not in this checkout")
def test_the_reference_is_current() -> None:
    assert API.read_text() == reference(), (
        "docs/API.md is out of date: python -m polariseq._api_reference > docs/API.md")
