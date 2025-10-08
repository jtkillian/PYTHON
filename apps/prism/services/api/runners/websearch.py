"""Lightweight web search collector."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup

from ..models import (
    ArtifactRecord,
    ArtifactType,
    CollectorName,
    CollectorResult,
    HighlightRecord,
    ProvenanceRecord,
)

SEARCH_URL = "https://duckduckgo.com/html/"
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,}\d")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


async def fetch_results(query: str, max_results: int = 10) -> List[dict]:
    async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": "PRISM/1.0"}) as client:
        response = await client.post(SEARCH_URL, data={"q": query})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        results = []
        for idx, result in enumerate(soup.select(".result")):
            if idx >= max_results:
                break
            title_elem = result.select_one(".result__title")
            link_elem = result.select_one("a.result__a")
            snippet_elem = result.select_one(".result__snippet")
            if not link_elem:
                continue
            results.append(
                {
                    "title": title_elem.get_text(strip=True) if title_elem else link_elem.get_text(strip=True),
                    "url": link_elem.get("href"),
                    "snippet": snippet_elem.get_text(strip=True) if snippet_elem else "",
                }
            )
        return results


async def run(query: Optional[str], workspace: Path) -> CollectorResult:
    result = CollectorResult(collector=CollectorName.WEBSEARCH)
    if not query:
        result.notes.append("No query provided; skipping web search")
        return result

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

    return result
