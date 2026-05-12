from __future__ import annotations

import html
import json
import re
import urllib.request
from typing import Any
from urllib.parse import urlparse

USER_AGENT = "Mozilla/5.0"
REQUEST_TIMEOUT_SECONDS = 20


def parse_krisha_listing_url(url: str) -> dict[str, Any]:
    parsed_url = urlparse(url)
    if parsed_url.netloc and "krisha.kz" not in parsed_url.netloc:
        raise ValueError("Only krisha.kz listing URLs are supported.")

    page_html = fetch_html(url)
    data = extract_jsdata(page_html)
    advert = data.get("advert")
    if not isinstance(advert, dict):
        raise ValueError("Krisha page does not contain advert data.")

    address = advert.get("address") or {}
    coordinates = advert.get("map") or {}
    photos = advert.get("photos") or []

    listing = {
        "listing_id": required(advert.get("id"), "listing_id"),
        "listing_price_kzt": required(advert.get("price"), "listing_price_kzt"),
        "area_m2": required(advert.get("square"), "area_m2"),
        "rooms": required(advert.get("rooms"), "rooms"),
        "lat": required(coordinates.get("lat"), "lat"),
        "lon": required(coordinates.get("lon"), "lon"),
        "district": required(address.get("district"), "district"),
        "microdistrict": address.get("microdistrict"),
        "complex_id": advert.get("complexId"),
        "year_built": info_value(page_html, "house.year"),
        "floor_text": info_value(page_html, "flat.floor") or parse_floor_from_title(advert.get("title")),
        "house_type": info_value(page_html, "flat.building"),
        "condition": info_value(page_html, "flat.renovation"),
        "ceiling_height_text": parameter_value(page_html, "ceiling"),
        "bathroom_type": parameter_value(page_html, "flat.toilet"),
        "has_photo": bool(photos),
        "photo_count": len(photos),
        "source_url": url,
        "title": advert.get("title"),
        "address_title": advert.get("addressTitle"),
    }
    ensure_required_listing_fields(listing)
    return listing


def fetch_html(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8")


def extract_jsdata(page_html: str) -> dict[str, Any]:
    match = re.search(
        r'<script id="jsdata">\s*window\.data\s*=\s*(\{.*?\});\s*</script>',
        page_html,
        flags=re.S,
    )
    if match is None:
        raise ValueError("Krisha page does not contain window.data JSON.")
    return json.loads(match.group(1))


def info_value(page_html: str, data_name: str) -> str | None:
    match = re.search(
        rf'data-name="{re.escape(data_name)}".*?'
        r'<div class="offer__advert-short-info">(.*?)</div>',
        page_html,
        flags=re.S,
    )
    if match is None:
        return None
    return clean_html_text(match.group(1))


def parameter_value(page_html: str, data_name: str) -> str | None:
    match = re.search(
        rf'<dt data-name="{re.escape(data_name)}">.*?</dt>\s*<dd>(.*?)</dd>',
        page_html,
        flags=re.S,
    )
    if match is None:
        return None
    return clean_html_text(match.group(1))


def parse_floor_from_title(title: str | None) -> str | None:
    if not title:
        return None
    match = re.search(r"(\d+\s*/\s*\d+)\s*этаж", title)
    if match is None:
        return None
    return match.group(1).replace("/", " из ")


def clean_html_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def required(value: Any, name: str) -> Any:
    if value is None or value == "":
        raise ValueError(f"Could not parse required field: {name}")
    return value


def ensure_required_listing_fields(listing: dict[str, Any]):
    required_fields = (
        "listing_price_kzt",
        "area_m2",
        "rooms",
        "lat",
        "lon",
    )
    missing = [field for field in required_fields if listing.get(field) in (None, "")]
    if missing:
        raise ValueError(f"Could not parse required listing fields: {', '.join(missing)}")
