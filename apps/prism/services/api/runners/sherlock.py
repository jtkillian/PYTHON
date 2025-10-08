"""Sherlock collector runner."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

from ..models import (
    ArtifactRecord,
    ArtifactType,
    CollectorName,
    CollectorResult,
    HighlightRecord,
    ProvenanceRecord,
)


def _determine_command(username: str) -> list[str] | None:
    if shutil.which("sherlock"):
        return ["sherlock", username, "--print-found", "--json"]
    if shutil.which(sys.executable):
        return [
            sys.executable,
            "-m",
            "sherlock",
            username,
            "--print-found",
            "--json",
        ]
    return None


def _record_hit(
    result: CollectorResult,
    username: str,
    service: str,
    url: str,
    workspace: Path,
) -> None:
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


def _fallback_hits(result: CollectorResult, username: str, workspace: Path) -> None:
    result.notes.append("Sherlock command not found; generating heuristic entries")
    for service in ["github", "twitter", "instagram"]:
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


def _write_raw_artifacts(result: CollectorResult, workspace: Path) -> None:
    for filename, content in result.raw_artifacts.items():
        (workspace / filename).write_text(content, encoding="utf-8")


async def _run_cli_command(
    command: list[str],
    username: str,
    workspace: Path,
    result: CollectorResult,
) -> None:
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:  # pragma: no cover - defensive
        result.notes.append(f"Sherlock execution error: {exc}")
        return

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0 or not stdout:
        message = stderr.decode("utf-8", "ignore")[:200]
        result.notes.append(f"Sherlock exited with code {proc.returncode}: {message}")
        return

    try:
        payload = json.loads(stdout.decode("utf-8", "ignore"))
    except json.JSONDecodeError:  # pragma: no cover - defensive
        payload = {}

    hits = payload.get(username, {})
    for service, metadata in hits.items():
        url = (metadata or {}).get("url_main") or (metadata or {}).get("url")
        if not url:
            continue
        _record_hit(result, username, service, url, workspace)


async def run(username: str | None, workspace: Path) -> CollectorResult:
    result = CollectorResult(collector=CollectorName.SHERLOCK)
    if not username:
        result.notes.append("No username provided; skipping Sherlock")
        return result

    command = _determine_command(username)
    if command:
        await _run_cli_command(command, username, workspace, result)
    else:
        _fallback_hits(result, username, workspace)

    _write_raw_artifacts(result, workspace)
    return result
