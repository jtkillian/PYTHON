"""Map visualization component using folium."""
from __future__ import annotations

import json
from pathlib import Path

import folium
import streamlit as st
import streamlit.components.v1 as components


def render_map(geojson_path: Path) -> None:
    if not geojson_path.exists():
        st.info("Map data not available.")
        return

    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    if not data.get("features"):
        st.caption("No location intelligence captured yet.")
        return

    fmap = folium.Map(location=[0, 0], zoom_start=2)
    folium.GeoJson(data).add_to(fmap)
    html = fmap.get_root().render()
    components.html(html, height=400)
