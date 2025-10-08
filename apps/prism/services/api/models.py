"""Data models and schemas for PRISM services."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl


class CollectorName(str, Enum):
    """Supported collector modules."""

    SHERLOCK = "sherlock"
    PHONEINFOGA = "phoneinfoga"
    WAYBACK = "wayback"
    WEBSEARCH = "websearch"


class ArtifactType(str, Enum):
    """Types of artifacts that can be stored for a person."""

    USERNAME = "username"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    LOCATION = "location"
    VEHICLE = "vehicle"
    PLATE = "plate"
    DEVICE = "device"
    IP = "ip"
    URL = "url"
    NOTE = "note"


@dataclass(slots=True)
class ArtifactRecord:
    """Normalized artifact data ready for persistence."""

    type: ArtifactType
    value: str
    confidence: float
    source: CollectorName
    raw_path: Path | None = None


@dataclass(slots=True)
class HighlightRecord:
    """Digest highlight data."""

    title: str
    description: str
    confidence: float


@dataclass(slots=True)
class ProvenanceRecord:
    """Reference material for artifacts."""

    source: CollectorName
    reference: str
    description: str


@dataclass(slots=True)
class CollectorResult:
    """Returned payload from a collector run."""

    collector: CollectorName
    raw_artifacts: dict[str, str] = field(default_factory=dict)
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    highlights: list[HighlightRecord] = field(default_factory=list)
    provenance: list[ProvenanceRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class ModuleToggles(BaseModel):
    """Toggle set for optional collectors and behaviors."""

    sherlock: bool = True
    phoneinfoga: bool = True
    wayback: bool = True
    websearch: bool = True
    save_html: bool = False
    screenshots: bool = False
    save_media: bool = False
    face_match: bool = False


class PersonInput(BaseModel):
    name: str = Field(..., description="Target full name")
    phone: str | None = Field(None, description="Phone number including country code")
    email: str | None = Field(None, description="Email address")
    username: str | None = Field(None, description="Username or handle")
    location: str | None = Field(None, description="Known location or address snippet")
    image_url: HttpUrl | None = Field(None, description="Optional image URL for face matching")


class ScanRequest(BaseModel):
    person: PersonInput
    toggles: ModuleToggles = Field(default_factory=ModuleToggles)


class SummarySection(BaseModel):
    label: str
    items: list[dict[str, str]]


class ScanSummary(BaseModel):
    slug: str
    created_at: datetime
    name: str
    highlights: list[dict[str, str]]
    sections: list[SummarySection]
    smart_picks: list[dict[str, str]]
    provenance: list[dict[str, str]]
    modules_run: list[str]


class ScanStatus(BaseModel):
    job_id: str
    status: str
    slug: str
    message: str | None = None


class ScanResponse(BaseModel):
    job_id: str
    slug: str
    summary: ScanSummary


class PersonRecord(BaseModel):
    id: int
    slug: str
    name: str
    phone: str | None
    email: str | None
    username: str | None
    created_at: datetime

    class Config:
        from_attributes = True


def as_dicts(records: Iterable[HighlightRecord]) -> list[dict[str, str]]:
    return [
        {"title": rec.title, "description": rec.description, "confidence": f"{rec.confidence:.2f}"}
        for rec in records
    ]


def flatten_artifacts(artifacts: Sequence[ArtifactRecord]) -> list[dict[str, str]]:
    return [
        {
            "type": artifact.type.value,
            "value": artifact.value,
            "confidence": f"{artifact.confidence:.2f}",
            "source": artifact.source.value,
            "raw_path": str(artifact.raw_path) if artifact.raw_path else None,
        }
        for artifact in artifacts
    ]
