"""Wayback Machine collector runner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import httpx

from ..models import (
    ArtifactRecord,
    ArtifactType,
    CollectorName,
    CollectorResult,
    HighlightRecord,
    ProvenanceRecord,
)

WAYBACK_URL = "https://web.archive.org/cdx/search/cdx"


async def fetch_wayback(query: str) -> List[dict]:
    params = {
        "url": f"*{query}*",
        "output": "json",
        "limit": "25",
        "filter": "statuscode:200",
        "collapse": "digest",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(WAYBACK_URL, params=params)
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return []
        headers = payload[0]
        results = []
        for row in payload[1:]:
            record = dict(zip(headers, row))
            results.append(record)
        return results


async def run(query: Optional[str], workspace: Path) -> CollectorResult:
    result = CollectorResult(collector=CollectorName.WAYBACK)
    if not query:
        result.notes.append("No query provided; skipping Wayback")
        return result

    try:
        records = await fetch_wayback(query)
    except Exception as exc:
        result.notes.append(f"Wayback lookup failed: {exc}")
        records = []

    raw_filename = "wayback.json"
    raw_path = workspace / raw_filename

    for idx, record in enumerate(records[:10]):
        url = f"https://web.archive.org/web/{record.get('timestamp')}/{record.get('original')}"
        description = f"Archived {record.get('mime')} ({record.get('length', 'unknown')} bytes)"
        result.artifacts.append(
            ArtifactRecord(
                type=ArtifactType.URL,
                value=url,
                confidence=0.5,
                source=CollectorName.WAYBACK,
                raw_path=raw_path,
            )
        )
        result.highlights.append(
            HighlightRecord(
                title=f"Archived page match #{idx + 1}",
                description=url,
                confidence=0.45,
            )
        )
        result.provenance.append(
            ProvenanceRecord(
                source=CollectorName.WAYBACK,
                reference=url,
                description="Archived web capture",
            )
        )
    if records:
        result.raw_artifacts[raw_filename] = json.dumps(records, indent=2)

    if not records:
        result.notes.append("No Wayback results found")

    for filename, content in result.raw_artifacts.items():
        (workspace / filename).write_text(content, encoding="utf-8")

    return result
