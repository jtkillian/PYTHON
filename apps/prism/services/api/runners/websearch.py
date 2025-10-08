"""Lightweight web search collector."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Iterable

import httpx
from bs4 import BeautifulSoup
from urllib.parse import parse_qs, urlparse

from ..models import (
    ArtifactRecord,
    ArtifactType,
    CollectorName,
    CollectorResult,
    HighlightRecord,
    ProvenanceRecord,
)


EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,}\d")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
BING_SEARCH_URL = "https://www.bing.com/search"
BING_HEADERS = {
    "User-Agent": "Mozilla/5.0",
}


def _decode_bing_url(url: str) -> str:
    if not url.startswith("https://www.bing.com/ck/a?"):
        return url
    encoded = parse_qs(urlparse(url).query).get("u", [""])[0]
    if not encoded:
        return url
    token = encoded[2:] if len(encoded) > 2 else encoded
    padding = "=" * ((-len(token)) % 4)
    try:
        return base64.urlsafe_b64decode(token + padding).decode("utf-8")
    except Exception:
        return url


def _extract_results(soup: BeautifulSoup, max_results: int) -> Iterable[dict]:
    seen: set[str] = set()
    for node in soup.select("li.b_algo"):
        link = node.select_one("h2 a") or node.select_one("div.header a")
        if not link:
            continue
        raw_url = link.get("href")
        if not raw_url:
            continue
        resolved_url = _decode_bing_url(raw_url)
        if resolved_url in seen:
            continue
        seen.add(resolved_url)
        title = link.get_text(strip=True) or resolved_url
        snippet_node = node.select_one("p")
        snippet = snippet_node.get_text(strip=True) if snippet_node else ""
        yield {"title": title, "url": resolved_url, "snippet": snippet}
        if len(seen) >= max_results:
            break


async def fetch_results(query: str, max_results: int = 10) -> list[dict]:
    async with httpx.AsyncClient(timeout=10.0, headers=BING_HEADERS) as client:
        response = await client.get(BING_SEARCH_URL, params={"q": query})
        response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    return list(_extract_results(soup, max_results))


async def run(query: str | None, workspace: Path) -> CollectorResult:
    result = CollectorResult(collector=CollectorName.WEBSEARCH)
    if not query:
        result.notes.append("No query provided; skipping web search")
        return result

    workspace.mkdir(parents=True, exist_ok=True)

    try:
        results = await fetch_results(query, max_results=10)
    except Exception as exc:
        result.notes.append(f"Web search failed: {exc}")
        results = []

    combined_text = "\n".join(f"{item['title']}\n{item['snippet']}" for item in results)
    emails = EMAIL_RE.findall(combined_text)
    phones = PHONE_RE.findall(combined_text)
    ips = [ip for ip in IP_RE.findall(combined_text) if ip != "0.0.0.0"]

    raw_filename = "websearch.json"
    raw_path = workspace / raw_filename

    for item in results:
        result.highlights.append(
            HighlightRecord(
                title=item["title"],
                description=item["url"],
                confidence=0.35,
            )
        )
        result.provenance.append(
            ProvenanceRecord(
                source=CollectorName.WEBSEARCH,
                reference=item["url"],
                description=item.get("snippet", "Web search snippet"),
            )
        )
    for email in set(emails):
        result.artifacts.append(
            ArtifactRecord(
                type=ArtifactType.EMAIL,
                value=email,
                confidence=0.3,
                source=CollectorName.WEBSEARCH,
                raw_path=raw_path,
            )
        )
    for phone in set(phones):
        result.artifacts.append(
            ArtifactRecord(
                type=ArtifactType.PHONE,
                value=phone,
                confidence=0.25,
                source=CollectorName.WEBSEARCH,
                raw_path=raw_path,
            )
        )
    for ip in set(ips):
        result.artifacts.append(
            ArtifactRecord(
                type=ArtifactType.IP,
                value=ip,
                confidence=0.2,
                source=CollectorName.WEBSEARCH,
                raw_path=raw_path,
            )
        )

    if results:
        result.raw_artifacts[raw_filename] = json.dumps(results, indent=2)

    # Write raw artifacts to disk before returning
    for filename, content in result.raw_artifacts.items():
        (workspace / filename).write_text(content, encoding="utf-8")

    return result
