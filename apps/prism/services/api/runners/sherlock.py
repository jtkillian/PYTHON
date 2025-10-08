"""Sherlock collector runner."""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

from ..models import (
    ArtifactRecord,
    ArtifactType,
    CollectorName,
    CollectorResult,
    HighlightRecord,
    ProvenanceRecord,
)


async def run(username: Optional[str], workspace: Path) -> CollectorResult:
    result = CollectorResult(collector=CollectorName.SHERLOCK)
    if not username:
        result.notes.append("No username provided; skipping Sherlock")
        return result

    command = None
    if shutil.which("sherlock"):
        command = ["sherlock", username, "--print-found", "--json"]
    elif shutil.which(sys.executable):
        command = [sys.executable, "-m", "sherlock", username, "--print-found", "--json"]

    if command:
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0 and stdout:
                try:
                    payload = json.loads(stdout.decode("utf-8", "ignore"))
                except json.JSONDecodeError:
                    payload = {}
                hits = payload.get(username, {})
                for service, metadata in hits.items():
                    if not metadata:
                        continue
                    url = metadata.get("url_main") or metadata.get("url")
                    if not url:
                        continue
                    filename = f"sherlock_{service}.txt"
                    target = workspace / filename
                    result.artifacts.append(
                        ArtifactRecord(
                            type=ArtifactType.USERNAME,
                            value=f"{username}@{service}",
                            confidence=0.8,
                            source=CollectorName.SHERLOCK,
                            raw_path=target,
                        )
                    )
                    result.highlights.append(
                        HighlightRecord(
                            title=f"Username match on {service}",
                            description=url,
                            confidence=0.7,
                        )
                    )
                    result.provenance.append(
                        ProvenanceRecord(
                            source=CollectorName.SHERLOCK,
                            reference=url,
                            description=f"Profile discovered on {service}",
                        )
                    )
                    result.raw_artifacts[filename] = url
            else:
                result.notes.append(
                    f"Sherlock exited with code {proc.returncode}: {stderr.decode('utf-8', 'ignore')[:200]}"
                )
        except Exception as exc:  # pragma: no cover - defensive
            result.notes.append(f"Sherlock execution error: {exc}")
    else:
        # fallback heuristics
        result.notes.append("Sherlock command not found; generating heuristic entries")
        fallback_services = ["github", "twitter", "instagram"]
        for service in fallback_services:
            url = f"https://{service}.com/{username}"
            filename = f"sherlock_{service}.txt"
            target = workspace / filename
            result.artifacts.append(
                ArtifactRecord(
                    type=ArtifactType.USERNAME,
                    value=f"{username}@{service}",
                    confidence=0.3,
                    source=CollectorName.SHERLOCK,
                    raw_path=target,
                )
            )
            result.highlights.append(
                HighlightRecord(
                    title=f"Potential profile on {service}",
                    description=url,
                    confidence=0.25,
                )
            )
            result.provenance.append(
                ProvenanceRecord(
                    source=CollectorName.SHERLOCK,
                    reference=url,
                    description=f"Heuristic profile on {service}",
                )
            )
            result.raw_artifacts[filename] = url

    for filename, content in result.raw_artifacts.items():
        target = workspace / filename
        target.write_text(content, encoding="utf-8")

    return result
