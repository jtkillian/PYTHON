"""PhoneInfoga collector runner."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

from phonenumbers import (
    PhoneNumber,
    PhoneNumberFormat,
    carrier,
    format_number,
    is_valid_number,
    parse,
    region_code_for_number,
)

from ..models import (
    ArtifactRecord,
    ArtifactType,
    CollectorName,
    CollectorResult,
    HighlightRecord,
    ProvenanceRecord,
)


def _determine_command(phone: str) -> list[str] | None:
    if shutil.which("phoneinfoga"):
        return ["phoneinfoga", "scan", "-n", phone, "-o", "json"]
    if shutil.which(sys.executable):
        return [sys.executable, "-m", "phoneinfoga", "scan", "-n", phone, "-o", "json"]
    return None


def _store_basic_payload(
    result: CollectorResult,
    phone: str,
    basic: dict,
    workspace: Path,
) -> None:
    carrier = basic.get("carrier") or "Unknown"
    country = basic.get("country") or "Unknown"
    valid = basic.get("valid")
    description = f"Carrier: {carrier} | Country: {country} | Valid: {valid}"
    filename = "phoneinfoga_basic.json"
    target = workspace / filename

    result.highlights.append(
        HighlightRecord(
            title="Phone intelligence",
            description=description,
            confidence=0.75,
        )
    )
    result.artifacts.append(
        ArtifactRecord(
            type=ArtifactType.PHONE,
            value=phone,
            confidence=0.8,
            source=CollectorName.PHONEINFOGA,
            raw_path=target,
        )
    )
    result.raw_artifacts[filename] = json.dumps(basic, indent=2)
    result.provenance.append(
        ProvenanceRecord(
            source=CollectorName.PHONEINFOGA,
            reference="PhoneInfoga basic scan",
            description=description,
        )
    )


def _store_fallback_metadata(
    result: CollectorResult,
    parsed: PhoneNumber,
    workspace: Path,
) -> None:
    region = region_code_for_number(parsed) or "Unknown"
    valid = is_valid_number(parsed)
    carrier_name = carrier.name_for_number(parsed, "en") or "Unknown"
    description = f"Carrier: {carrier_name} | Region: {region} | Valid: {valid}"
    filename = "phoneinfoga_local.json"
    target = workspace / filename

    result.highlights.append(
        HighlightRecord(
            title="Phone metadata (local)",
            description=description,
            confidence=0.45,
        )
    )
    result.artifacts.append(
        ArtifactRecord(
            type=ArtifactType.PHONE,
            value=format_number(parsed, PhoneNumberFormat.INTERNATIONAL),
            confidence=0.4,
            source=CollectorName.PHONEINFOGA,
            raw_path=target,
        )
    )
    result.raw_artifacts[filename] = json.dumps(
        {
            "carrier": carrier_name,
            "region": region,
            "valid": valid,
        },
        indent=2,
    )
    result.provenance.append(
        ProvenanceRecord(
            source=CollectorName.PHONEINFOGA,
            reference="phonenumbers library",
            description=description,
        )
    )


async def _run_cli_command(
    command: list[str],
    phone: str,
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
        result.notes.append(f"PhoneInfoga execution error: {exc}")
        return

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0 or not stdout:
        message = stderr.decode("utf-8", "ignore")[:200]
        result.notes.append(f"PhoneInfoga exited with code {proc.returncode}: {message}")
        return

    try:
        payload = json.loads(stdout.decode("utf-8", "ignore"))
    except json.JSONDecodeError:  # pragma: no cover - defensive
        payload = {}

    basic = payload.get("data", {}).get("basic", {})
    if basic:
        _store_basic_payload(result, phone, basic, workspace)


def _write_raw_artifacts(result: CollectorResult, workspace: Path) -> None:
    for filename, content in result.raw_artifacts.items():
        (workspace / filename).write_text(content, encoding="utf-8")


async def run(phone: str | None, workspace: Path) -> CollectorResult:
    result = CollectorResult(collector=CollectorName.PHONEINFOGA)
    if not phone:
        result.notes.append("No phone provided; skipping PhoneInfoga")
        return result

    parsed: PhoneNumber | None = None
    try:
        parsed = parse(phone, None)
    except Exception:  # pragma: no cover - parsing errors expected for invalid numbers
        result.notes.append("Unable to parse phone number; continuing")

    command = _determine_command(phone)
    if command:
        await _run_cli_command(command, phone, workspace, result)
    else:
        result.notes.append("PhoneInfoga command not found; using phonenumbers metadata")

    if not result.raw_artifacts and parsed is not None:
        _store_fallback_metadata(result, parsed, workspace)

    _write_raw_artifacts(result, workspace)
    return result
