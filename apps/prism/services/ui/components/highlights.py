"""Streamlit components for displaying highlights."""
from __future__ import annotations

from typing import List, Mapping

import streamlit as st


def render_highlights(highlights: List[Mapping[str, str]]) -> None:
    """Render highlight cards in a responsive layout."""

    if not highlights:
        st.info("No highlights available yet. Run a scan to populate the digest.")
        return

    cols = st.columns(2)
    for idx, highlight in enumerate(highlights):
        column = cols[idx % 2]
        with column:
            st.markdown(
                f"""
                <div class="highlight-card">
                    <strong>{highlight.get('title')}</strong><br />
                    <span>{highlight.get('description')}</span><br />
                    <small>Confidence: {highlight.get('confidence')}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
