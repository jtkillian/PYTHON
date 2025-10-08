"""Render summary sections."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import streamlit as st


def render_sections(sections: Iterable[Mapping[str, object]]) -> None:
    for section in sections:
        st.subheader(section.get("label", "Section"))
        raw_items = section.get("items") or []
        items = cast(Sequence[Mapping[str, object]], raw_items)
        if not items:
            st.caption("No data available.")
            continue
        for item in items:
            entry = (
                f"- {item.get('value')} — confidence {item.get('confidence')} "
                f"(source: {item.get('source')})"
            )
            st.write(entry)
