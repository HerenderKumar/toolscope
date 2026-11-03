import json
from pathlib import Path

import pytest

from toolscope.catalog.build_catalog import extract_params, make_search_text

CATALOG = Path("data/catalog/tools.jsonl")


@pytest.fixture(scope="module")
def cards() -> list[dict]:
    if not CATALOG.exists():
        pytest.skip("run build_catalog first")
    return [json.loads(line) for line in CATALOG.read_text().splitlines()]


def test_line_count(cards):
    assert len(cards) == 525


def test_unique_ids(cards):
    assert len({c["tool_id"] for c in cards}) == len(cards)


def test_search_text_non_empty(cards):
    assert all(c["search_text"].strip() for c in cards)


def test_input_schema_is_object(cards):
    assert all(isinstance(c["input_schema"], dict) for c in cards)


def test_shared_name_kept_apart(cards):
    ids = {c["tool_id"] for c in cards}
    assert {"filesystem/read_file", "desktop-commander/read_file"} <= ids


def test_extract_params_and_search_text():
    schema = {"type": "object", "required": ["city"],
              "properties": {"city": {"type": "string", "description": "City name"},
                             "units": {"type": "string"}}}
    params = extract_params(schema)
    assert params == [
        {"name": "city", "type": "string", "description": "City name", "required": True},
        {"name": "units", "type": "string", "description": "", "required": False},
    ]
    text = make_search_text("Weather", "get_forecast", "Get a forecast", params)
    assert text == "Weather | get forecast | get_forecast | Get a forecast | city: City name | units"
