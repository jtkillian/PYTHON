"""Streamlit UI for PRISM."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


PRISM_ROOT = CURRENT_FILE.parents[2]

load_dotenv(REPO_ROOT / ".env", override=False)
load_dotenv(PRISM_ROOT / ".env.example")
load_dotenv(PRISM_ROOT / ".env", override=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")
OUTPUT_DIR = PRISM_ROOT / "output"
FACE_MODEL = PRISM_ROOT / "models" / "face" / "model.onnx"


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


DEFAULT_TOGGLES = {
    "sherlock": True,
    "phoneinfoga": True,
    "wayback": True,
    "websearch": True,
    "save_html": _env_flag("DEFAULT_SAVE_HTML"),
    "screenshots": _env_flag("DEFAULT_SCREENSHOTS"),
    "save_media": _env_flag("DEFAULT_SAVE_MEDIA"),
    "face_match": FACE_MODEL.exists() and _env_flag("DEFAULT_FACE_MATCH"),
}


def call_api(endpoint: str, payload: dict | None = None) -> dict:
    with httpx.Client(timeout=60.0) as client:
        response = client.post(f"{API_URL}{endpoint}", json=payload)
        response.raise_for_status()
        return response.json()


def main() -> None:
    from apps.prism.services.ui.components.graph import render_graph
    from apps.prism.services.ui.components.highlights import render_highlights
    from apps.prism.services.ui.components.map_view import render_map
    from apps.prism.services.ui.components.sections import render_sections

    st.set_page_config(page_title="PRISM Console", layout="wide", page_icon="🛰️")
    st.title("PRISM — OSINT Digest Console")
    st.caption("Search across free OSINT sources and get a digest-first view of the results.")
    st.markdown(
        """
        <style>
        .highlight-card {
            background: rgba(15, 23, 42, 0.85);
            color: #f8fafc;
            padding: 1rem;
            border-radius: 0.75rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Run Configuration")
        st.write("Enable or disable modules before running a scan.")
        toggles = {}
        toggles["sherlock"] = st.checkbox("Sherlock", value=DEFAULT_TOGGLES["sherlock"])
        toggles["phoneinfoga"] = st.checkbox("PhoneInfoga", value=DEFAULT_TOGGLES["phoneinfoga"])
        toggles["wayback"] = st.checkbox("Wayback Machine", value=DEFAULT_TOGGLES["wayback"])
        toggles["websearch"] = st.checkbox("Web Search", value=DEFAULT_TOGGLES["websearch"])
        st.divider()
        toggles["save_html"] = st.checkbox(
            "Save HTML",
            value=DEFAULT_TOGGLES["save_html"],
        )
        toggles["screenshots"] = st.checkbox(
            "Take Screenshots",
            value=DEFAULT_TOGGLES["screenshots"],
        )
        toggles["save_media"] = st.checkbox(
            "Download Media",
            value=DEFAULT_TOGGLES["save_media"],
        )
        face_enabled = FACE_MODEL.exists()
        toggles["face_match"] = st.checkbox(
            "Face Match" if face_enabled else "Face Match (model required)",
            value=DEFAULT_TOGGLES["face_match"],
            disabled=not face_enabled,
        )
        st.caption("Paid modules (HIBP, Shodan, etc.) are stubbed and disabled by default.")

    st.subheader("Person of Interest")
    with st.form("scan-form"):
        name = st.text_input("Full Name", help="Required. Use the person's known name.")
        col1, col2, col3 = st.columns(3)
        with col1:
            phone = st.text_input("Phone Number", placeholder="+1 555-0100")
        with col2:
            email = st.text_input("Email Address")
        with col3:
            username = st.text_input("Username / Handle")
        location = st.text_input("Known Location")
        image_url = st.text_input("Image URL (optional)")
        submitted = st.form_submit_button("Run Scan", use_container_width=True)

    if submitted:
        if not name.strip():
            st.error("Name is required to perform a scan.")
        else:
            payload = {
                "person": {
                    "name": name,
                    "phone": phone or None,
                    "email": email or None,
                    "username": username or None,
                    "location": location or None,
                    "image_url": image_url or None,
                },
                "toggles": toggles,
            }
            with st.spinner("Running collectors..."):
                try:
                    response = call_api("/scan", payload)
                except Exception as exc:
                    st.error(f"Scan failed: {exc}")
                    return
            summary = response.get("summary", {})
            slug = summary.get("slug")
            st.success(f"Scan complete for {summary.get('name')} (slug: {slug})")

            st.write("### Top Highlights")
            render_highlights(summary.get("highlights", []))

            st.write("### Key Sections")
            render_sections(summary.get("sections", []))

            st.write("### Smart Picks")
            for pick in summary.get("smart_picks", []):
                st.write(f"• {pick.get('text')} — source {pick.get('source')}")

            st.write("### Provenance")
            provenance = summary.get("provenance", [])
            if provenance:
                for prov in provenance:
                    prov_text = (
                        f"- {prov.get('source')}: {prov.get('reference')} "
                        f"({prov.get('description')})"
                    )
                    st.write(prov_text)
            else:
                st.caption("No provenance entries recorded.")

            if slug:
                person_dir = OUTPUT_DIR / slug
                findings_md = person_dir / "Findings.md"
                findings_json = person_dir / "Findings.json"
                graph_path = person_dir / "graph.gexf"
                map_path = person_dir / "map.geojson"

                st.download_button(
                    "Download Findings.md",
                    data=findings_md.read_text(encoding="utf-8") if findings_md.exists() else "",
                    file_name=f"{slug}-Findings.md",
                )
                json_payload = (
                    findings_json.read_text(encoding="utf-8")
                    if findings_json.exists()
                    else json.dumps(summary, indent=2)
                )
                st.download_button(
                    "Download Findings.json",
                    data=json_payload,
                    file_name=f"{slug}-Findings.json",
                )

                st.write("### Relationship Graph")
                render_graph(graph_path)

                st.write("### Geo Map")
                render_map(map_path)


if __name__ == "__main__":
    main()
