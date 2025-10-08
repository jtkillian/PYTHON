"""Unit tests for PRISM API utilities."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest
from apps.prism.services.api.models import PersonInput, ScanSummary, SummarySection
from apps.prism.services.api.runners import phoneinfoga, sherlock


api_app = cast(Any, importlib.import_module("apps.prism.services.api.app"))


@pytest.fixture(autouse=True)
def reset_settings() -> Iterator[None]:
    """Ensure settings caps are restored after each test."""
    original = {
        "cap_phones": api_app.settings.cap_phones,
        "cap_emails": api_app.settings.cap_emails,
        "cap_usernames": api_app.settings.cap_usernames,
        "cap_urls": api_app.settings.cap_urls,
        "cap_smart_picks": api_app.settings.cap_smart_picks,
        "cap_top_highlights": api_app.settings.cap_top_highlights,
    }
    yield
    for key, value in original.items():
        setattr(api_app.settings, key, value)


def test_group_artifacts_respects_caps() -> None:
    api_app.settings.cap_phones = 1
    api_app.settings.cap_emails = 1
    artifacts = [
        {"type": "phone", "value": "+15550100", "confidence": 0.9, "source": "collector"},
        {"type": "phone", "value": "+15550101", "confidence": 0.8, "source": "collector"},
        {"type": "email", "value": "alice@example.com", "confidence": 0.8, "source": "collector"},
        {"type": "email", "value": "bob@example.com", "confidence": 0.7, "source": "collector"},
    ]

    grouped = api_app.group_artifacts(artifacts)

    assert grouped == {
        "Phones": [
            {
                "value": "+15550100",
                "confidence": "0.90",
                "source": "collector",
            }
        ],
        "Emails": [
            {
                "value": "alice@example.com",
                "confidence": "0.80",
                "source": "collector",
            }
        ],
    }


def test_generate_mmr_deduplicates_and_limits() -> None:
    items = ["alpha", "beta", "alpha", "gamma"]
    result = api_app.generate_mmr(items, limit=2)

    assert len(result) == 2
    assert len(set(result)) == len(result)


def test_write_outputs_creates_markdown_and_json(tmp_path: Path) -> None:
    person = PersonInput(
        name="Jane Doe",
        phone=None,
        email=None,
        username=None,
        location=None,
        image_url=None,
    )
    summary = ScanSummary(
        slug="jane-doe",
        created_at=datetime.utcnow(),
        name="Jane Doe",
        highlights=[{"title": "Hit", "description": "desc", "confidence": "0.95"}],
        sections=[
            SummarySection(
                label="Phones",
                items=[{"value": "+15550100", "source": "collector", "confidence": "0.80"}],
            )
        ],
        smart_picks=[{"text": "Pick", "source": "collector"}],
        provenance=[{"source": "collector", "reference": "ref", "description": "desc"}],
        modules_run=["websearch"],
    )

    original_output = api_app.OUTPUT_DIR
    try:
        api_app.OUTPUT_DIR = tmp_path
        api_app.write_outputs(summary.slug, person, summary)
    finally:
        api_app.OUTPUT_DIR = original_output

    findings_md = tmp_path / summary.slug / "Findings.md"
    findings_json = tmp_path / summary.slug / "Findings.json"

    assert findings_md.exists()
    assert findings_json.exists()
    assert "Hit" in findings_md.read_text(encoding="utf-8")


def test_phoneinfoga_fallback_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(phoneinfoga, "_determine_command", lambda _phone: None)

    result = asyncio.run(phoneinfoga.run("+15550100", tmp_path))

    assert result.artifacts, "Expected fallback artifacts when CLI is unavailable"
    expected_raw = tmp_path / "phoneinfoga_local.json"
    assert expected_raw.exists()


def test_sherlock_fallback_generates_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sherlock, "_determine_command", lambda _username: None)

    result = asyncio.run(sherlock.run("osintfan", tmp_path))

    assert result.artifacts, "Expected heuristic artifacts when CLI is unavailable"
    for service in ["github", "twitter", "instagram"]:
        assert (tmp_path / f"sherlock_{service}.txt").exists()
