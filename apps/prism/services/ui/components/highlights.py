"""Streamlit components for displaying highlights."""
from __future__ import annotations

from collections.abc import Mapping
from html import escape

import streamlit as st


def render_highlights(highlights: list[Mapping[str, str]]) -> None:
    """Render highlight cards in a responsive layout."""

    if not highlights:
        st.info("No highlights available yet. Run a scan to populate the digest.")
        return

    cols = st.columns(2)
    for idx, highlight in enumerate(highlights):
        column = cols[idx % 2]
        title = escape(str(highlight.get("title", "")))
        description = escape(str(highlight.get("description", "")))
        confidence = escape(str(highlight.get("confidence", "")))
        with column:
            st.markdown(
                f"""
                <div class="highlight-card">
                    <strong>{title}</strong><br />
                    <span>{description}</span><br />
                    <small>Confidence: {confidence}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
