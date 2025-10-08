"""FastAPI application for PRISM."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import orjson
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slugify import slugify

from . import storage
from .models import (
    ArtifactType,
    CollectorResult,
    ModuleToggles,
    PersonInput,
    ScanRequest,
    ScanResponse,
    ScanSummary,
    SummarySection,
)
from .runners import phoneinfoga, sherlock, wayback, websearch

# Integration stubs


def oracle_stub(summary: Dict) -> None:  # pragma: no cover - placeholder
    logging.getLogger("prism").debug("ORACLE stub called", extra={"summary": summary})


def aegis_stub(payload: Dict) -> None:  # pragma: no cover - placeholder
    logging.getLogger("prism").debug("AEGIS stub called", extra={"payload": payload})


def citadel_stub(_: Dict) -> None:  # pragma: no cover - placeholder
    logging.getLogger("prism").debug("CITADEL stub called")


PRISM_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PRISM_ROOT.parents[1]
OUTPUT_DIR = PRISM_ROOT / "output"
ARTIFACT_DIR = PRISM_ROOT / "data" / "artifacts"
FACE_MODEL = PRISM_ROOT / "models" / "face" / "model.onnx"


class Settings:
    def __init__(self) -> None:
        load_dotenv(REPO_ROOT / ".env", override=False)
        load_dotenv(PRISM_ROOT / ".env", override=True)
        load_dotenv(PRISM_ROOT / ".env.example", override=False)
        self.reload()

    def _bool(self, name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _int(self, name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None or not value.strip():
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def _list(self, name: str, default: List[str]) -> List[str]:
        value = os.getenv(name)
        if value is None:
            return default
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or default

    def reload(self) -> None:
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = self._int("API_PORT", 8000)
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.cap_top_highlights = self._int("CAP_TOP_HIGHLIGHTS", 10)
        self.cap_phones = self._int("CAP_PHONES", 5)
        self.cap_emails = self._int("CAP_EMAILS", 5)
        self.cap_usernames = self._int("CAP_USERNAMES", 10)
        self.cap_addresses = self._int("CAP_ADDRESSES", 5)
        self.cap_locations = self._int("CAP_LOCATIONS", 10)
        self.cap_vehicles = self._int("CAP_VEHICLES", 5)
        self.cap_plates = self._int("CAP_PLATES", 3)
        self.cap_devices = self._int("CAP_DEVICES", 5)
        self.cap_ips = self._int("CAP_IPS", 5)
        self.cap_urls = self._int("CAP_URLS", 10)
        self.cap_smart_picks = self._int("CAP_SMART_PICKS", 20)
        self.default_save_html = self._bool("DEFAULT_SAVE_HTML", False)
        self.default_screenshots = self._bool("DEFAULT_SCREENSHOTS", False)
        self.default_save_media = self._bool("DEFAULT_SAVE_MEDIA", False)
        self.default_face_match = self._bool("DEFAULT_FACE_MATCH", False)
        self.allowed_origins = self._list("ALLOWED_ORIGINS", ["http://localhost:8501"])

    def default_toggles(self) -> ModuleToggles:
        face_toggle = self.default_face_match and FACE_MODEL.exists()
        return ModuleToggles(
            sherlock=True,
            phoneinfoga=True,
            wayback=True,
            websearch=True,
            save_html=self.default_save_html,
            screenshots=self.default_screenshots,
            save_media=self.default_save_media,
            face_match=face_toggle,
        )


settings = Settings()
logger = logging.getLogger("prism")
logger.setLevel(settings.log_level)
logger.addHandler(logging.StreamHandler())

app = FastAPI(title="PRISM API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    settings.reload()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    await storage.init_db()


async def run_collectors(person: PersonInput, toggles: ModuleToggles, slug: str) -> List[CollectorResult]:
    workspace = OUTPUT_DIR / slug / "artifacts"
    workspace.mkdir(parents=True, exist_ok=True)

    collectors: List[asyncio.Task[CollectorResult]] = []

    if toggles.sherlock:
        collectors.append(asyncio.create_task(sherlock.run(person.username, workspace)))
    if toggles.phoneinfoga:
        collectors.append(asyncio.create_task(phoneinfoga.run(person.phone, workspace)))
    if toggles.wayback:
        search_term = person.email or person.phone or person.username or person.name
        collectors.append(asyncio.create_task(wayback.run(search_term, workspace)))
    if toggles.websearch:
        query = " ".join(filter(None, [person.name, person.username, person.email, person.phone or ""]))
        collectors.append(asyncio.create_task(websearch.run(query.strip() or None, workspace)))

    if not collectors:
        return []

    results: List[CollectorResult] = []
    for task in asyncio.as_completed(collectors):
        try:
            results.append(await task)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Collector execution failed: %s", exc)
    return results


SECTION_ORDER = [
    ArtifactType.PHONE.value,
    ArtifactType.EMAIL.value,
    ArtifactType.USERNAME.value,
    ArtifactType.ADDRESS.value,
    ArtifactType.LOCATION.value,
    ArtifactType.VEHICLE.value,
    ArtifactType.PLATE.value,
    ArtifactType.DEVICE.value,
    ArtifactType.IP.value,
    ArtifactType.URL.value,
]

def get_section_map() -> Dict[str, Tuple[str, int]]:
    return {
        ArtifactType.PHONE.value: ("Phones", settings.cap_phones),
        ArtifactType.EMAIL.value: ("Emails", settings.cap_emails),
        ArtifactType.USERNAME.value: ("Usernames", settings.cap_usernames),
        ArtifactType.ADDRESS.value: ("Addresses", settings.cap_addresses),
        ArtifactType.LOCATION.value: ("Locations", settings.cap_locations),
        ArtifactType.VEHICLE.value: ("Vehicles", settings.cap_vehicles),
        ArtifactType.PLATE.value: ("Plates", settings.cap_plates),
        ArtifactType.DEVICE.value: ("Devices", settings.cap_devices),
        ArtifactType.IP.value: ("IP Addresses", settings.cap_ips),
        ArtifactType.URL.value: ("Links", settings.cap_urls),
    }


def group_artifacts(artifacts: Iterable[dict]) -> Dict[str, List[dict]]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    section_map = get_section_map()
    for artifact in artifacts:
        artifact_type = artifact.get("type")
        label, limit = section_map.get(artifact_type, ("Other", 10))
        if len(buckets[label]) >= limit:
            continue
        buckets[label].append(
            {
                "value": artifact["value"],
                "confidence": f"{float(artifact.get('confidence', 0)):.2f}",
                "source": artifact.get("source") or "unknown",
            }
        )
    ordered: Dict[str, List[dict]] = {}
    for type_key in SECTION_ORDER:
        label, _ = section_map.get(type_key, (None, None))
        if label and label in buckets:
            ordered[label] = buckets[label]
    for label, items in buckets.items():
        if label not in ordered:
            ordered[label] = items
    return ordered


def generate_mmr(items: List[str], limit: int) -> List[str]:
    if not items:
        return []
    items = list(dict.fromkeys(items))
    selected: List[str] = []
    lambda_param = 0.7
    while items and len(selected) < limit:
        if not selected:
            selected.append(items.pop(0))
            continue
        candidates = []
        for item in items:
            relevance = len(item)
            diversity = min(len(item), *(abs(len(item) - len(s)) for s in selected)) if selected else len(item)
            score = lambda_param * relevance - (1 - lambda_param) * diversity
            candidates.append((score, item))
        candidates.sort(reverse=True, key=lambda x: x[0])
        best = candidates[0][1]
        selected.append(best)
        items.remove(best)
    return selected


def build_graph(slug: str, artifacts: Iterable[dict]) -> None:
    graph = nx.Graph()
    person_node = f"person:{slug}"
    graph.add_node(person_node, label="Person")
    for artifact in artifacts:
        node_id = f"{artifact['type']}:{artifact['value']}"
        graph.add_node(node_id, label=artifact["value"], type=artifact["type"])
        graph.add_edge(person_node, node_id, source=artifact.get("source"))
    graph_path = OUTPUT_DIR / slug / "graph.gexf"
    nx.write_gexf(graph, graph_path)


def build_map(slug: str, artifacts: Iterable[dict]) -> None:
    features = []
    for artifact in artifacts:
        if artifact["type"] not in {ArtifactType.LOCATION.value, ArtifactType.ADDRESS.value}:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": artifact["value"],
                    "source": artifact.get("source"),
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [0, 0],
                },
            }
        )
    geojson = {"type": "FeatureCollection", "features": features}
    map_path = OUTPUT_DIR / slug / "map.geojson"
    (OUTPUT_DIR / slug).mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")


def write_outputs(slug: str, person: PersonInput, summary: ScanSummary) -> None:
    output_dir = OUTPUT_DIR / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_lines = [f"# Findings for {person.name}", "", f"Generated: {summary.created_at.isoformat()}"]
    markdown_lines.append("\n## Top Highlights\n")
    for highlight in summary.highlights:
        markdown_lines.append(f"- **{highlight['title']}** — {highlight['description']} (confidence {highlight['confidence']})")
    markdown_lines.append("\n## Sections\n")
    for section in summary.sections:
        markdown_lines.append(f"### {section.label}")
        for item in section.items:
            markdown_lines.append(f"- {item['value']} ({item.get('source')}) — confidence {item.get('confidence')}")
        markdown_lines.append("")
    markdown_lines.append("\n## Smart Picks\n")
    for pick in summary.smart_picks:
        markdown_lines.append(f"- {pick['text']} (source: {pick['source']})")
    markdown_lines.append("\n## Provenance\n")
    for prov in summary.provenance:
        markdown_lines.append(f"- {prov['source']}: {prov['reference']} — {prov['description']}")
    (output_dir / "Findings.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    (output_dir / "Findings.json").write_bytes(orjson.dumps(summary.model_dump(mode='json'), option=orjson.OPT_INDENT_2))


async def orchestrate_scan(payload: ScanRequest) -> ScanSummary:
    slug_basis = slugify(payload.person.name) or slugify(payload.person.username or "")
    slug = slug_basis or f"person-{int(datetime.utcnow().timestamp())}"
    output_dir = OUTPUT_DIR / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    toggle_data = settings.default_toggles().model_dump()
    toggle_data.update(payload.toggles.model_dump())
    if FACE_MODEL.exists():
        toggle_data["face_match"] = toggle_data.get("face_match", False) or settings.default_face_match
    else:
        toggle_data["face_match"] = False
    toggles = ModuleToggles(**toggle_data)

    results = await run_collectors(payload.person, toggles, slug)

    async with storage.SessionLocal() as session:
        person = await storage.upsert_person(session, slug, payload.person)
        await storage.clear_previous(session, person)
        await storage.persist_results(session, person, results)
        await session.commit()
        data = await storage.fetch_person_summary(session, person)

    highlights = sorted(data["highlights"], key=lambda h: h.get("confidence", 0), reverse=True)[
        : settings.cap_top_highlights
    ]
    artifacts = data["artifacts"]
    provenance = data["provenance"]
    grouped = group_artifacts(artifacts)

    smart_pool = [f"{a['value']} ({a.get('source', 'unknown')})" for a in artifacts]
    smart_picks = [
        {
            "text": text,
            "source": text.split("(")[-1].rstrip(")") if "(" in text else "unknown",
        }
        for text in generate_mmr(smart_pool, settings.cap_smart_picks)
    ]

    summary = ScanSummary(
        slug=slug,
        created_at=datetime.utcnow(),
        name=payload.person.name,
        highlights=[
            {
                "title": h.get("title"),
                "description": h.get("description"),
                "confidence": f"{h.get('confidence', 0):.2f}",
            }
            for h in highlights
        ],
        sections=[SummarySection(label=label, items=items) for label, items in grouped.items()],
        smart_picks=smart_picks,
        provenance=[
            {
                "source": p.get("source") or "unknown",
                "reference": p.get("reference") or "n/a",
                "description": p.get("description") or "",
            }
            for p in provenance
        ],
        modules_run=[result.collector.value for result in results],
    )

    build_graph(slug, artifacts)
    build_map(slug, artifacts)
    write_outputs(slug, payload.person, summary)

    oracle_stub(summary.model_dump())
    aegis_stub({"slug": slug, "modules": summary.modules_run})
    citadel_stub({"slug": slug})

    return summary


@app.post("/scan", response_model=ScanResponse)
async def scan(payload: ScanRequest) -> ScanResponse:
    summary = await orchestrate_scan(payload)
    job_id = f"job-{summary.slug}-{int(summary.created_at.timestamp())}"
    return ScanResponse(job_id=job_id, slug=summary.slug, summary=summary)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "face_model": str(FACE_MODEL.exists()).lower()}
