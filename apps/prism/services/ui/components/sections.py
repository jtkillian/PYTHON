"""Render summary sections."""
from __future__ import annotations

from typing import Iterable, Mapping

import streamlit as st


def render_sections(sections: Iterable[Mapping[str, object]]) -> None:
    for section in sections:
        st.subheader(section.get("label", "Section"))
        items = section.get("items", []) or []
        if not items:
            st.caption("No data available.")
            continue
        for item in items:
            st.write(
                f"- {item.get('value')} — confidence {item.get('confidence')} (source: {item.get('source')})"
            )
