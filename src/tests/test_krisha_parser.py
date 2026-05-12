from __future__ import annotations

import json

import pytest

from backend.krisha import (
    clean_html_text,
    ensure_required_listing_fields,
    extract_jsdata,
    info_value,
    parameter_value,
    parse_floor_from_title,
)


def test_parse_floor_from_title_extracts_floor_pair():
    assert parse_floor_from_title("2-комнатная, 55 м², 5/12 этаж") == "5 из 12"


def test_parse_floor_from_title_returns_none_on_missing():
    assert parse_floor_from_title("Просто заголовок") is None
    assert parse_floor_from_title(None) is None


def test_clean_html_text_strips_tags_and_whitespace():
    assert clean_html_text("<span>  привет<br/>мир  </span>") == "привет мир"


def test_clean_html_text_unescapes_entities():
    assert clean_html_text("&laquo;hello&raquo;") == "«hello»"


def test_extract_jsdata_parses_inline_json():
    payload = {"advert": {"id": 123, "price": 1, "square": 1, "rooms": 1}}
    html_page = (
        '<html><body><script id="jsdata">window.data = '
        + json.dumps(payload)
        + ";</script></body></html>"
    )
    parsed = extract_jsdata(html_page)
    assert parsed["advert"]["id"] == 123


def test_extract_jsdata_raises_when_missing():
    with pytest.raises(ValueError):
        extract_jsdata("<html><body><p>no script</p></body></html>")


def test_info_value_returns_inner_text():
    html_page = """
    <div data-name="house.year">
        <div class="offer__advert-short-info">2020 г.п.</div>
    </div>
    """
    assert info_value(html_page, "house.year") == "2020 г.п."


def test_info_value_returns_none_when_block_absent():
    assert info_value("<html></html>", "house.year") is None


def test_parameter_value_returns_dd_text():
    html_page = '<dt data-name="ceiling">Потолки</dt><dd>2.9 м</dd>'
    assert parameter_value(html_page, "ceiling") == "2.9 м"


def test_ensure_required_listing_fields_raises_on_missing_coordinates():
    listing = {
        "listing_price_kzt": 1,
        "area_m2": 50,
        "rooms": 2,
        "lat": None,
        "lon": 76.9,
    }
    with pytest.raises(ValueError):
        ensure_required_listing_fields(listing)


def test_ensure_required_listing_fields_passes_without_optional_fields():
    listing = {
        "listing_price_kzt": 1,
        "area_m2": 50,
        "rooms": 2,
        "lat": 43.2,
        "lon": 76.9,
    }
    ensure_required_listing_fields(listing)
