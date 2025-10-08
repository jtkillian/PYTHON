"""PhoneInfoga collector runner."""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import phonenumbers

from ..models import (
    ArtifactRecord,
    ArtifactType,
    CollectorName,
    CollectorResult,
    HighlightRecord,
    ProvenanceRecord,
)


async def run(phone: Optional[str], workspace: Path) -> CollectorResult:
    result = CollectorResult(collector=CollectorName.PHONEINFOGA)
    if not phone:
        result.notes.append("No phone provided; skipping PhoneInfoga")
        return result

    parsed = None
    try:
        parsed = phonenumbers.parse(phone, None)
    except Exception:  # pragma: no cover - parsing errors expected for invalid numbers
        result.notes.append("Unable to parse phone number; continuing")

    command = None
    if shutil.which("phoneinfoga"):
        command = ["phoneinfoga", "scan", "-n", phone, "-o", "json"]
    elif shutil.which(sys.executable):
        command = [sys.executable, "-m", "phoneinfoga", "scan", "-n", phone, "-o", "json"]

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
                basic = payload.get("data", {}).get("basic", {})
                if basic:
                    carrier = basic.get("carrier")
                    country = basic.get("country")
                    valid = basic.get("valid")
                    description = f"Carrier: {carrier or 'Unknown'} | Country: {country or 'Unknown'} | Valid: {valid}"
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
            else:
                result.notes.append(
                    f"PhoneInfoga exited with code {proc.returncode}: {stderr.decode('utf-8', 'ignore')[:200]}"
                )
        except Exception as exc:  # pragma: no cover - defensive
            result.notes.append(f"PhoneInfoga execution error: {exc}")
    else:
        result.notes.append("PhoneInfoga command not found; using phonenumbers metadata")
        if parsed:
            region = phonenumbers.region_code_for_number(parsed) or "Unknown"
            valid = phonenumbers.is_valid_number(parsed)
            carrier = phonenumbers.carrier.name_for_number(parsed, "en") or "Unknown"
            description = f"Carrier: {carrier} | Region: {region} | Valid: {valid}"
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
                    value=phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
                    confidence=0.4,
                    source=CollectorName.PHONEINFOGA,
                    raw_path=target,
                )
            )
            result.raw_artifacts[filename] = json.dumps(
                {
                    "carrier": carrier,
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

    for filename, content in result.raw_artifacts.items():
        (workspace / filename).write_text(content, encoding="utf-8")

    return result
