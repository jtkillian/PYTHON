"""Graph visualization helpers."""
from __future__ import annotations

from pathlib import Path

from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components


def render_graph(graph_path: Path) -> None:
    if not graph_path.exists():
        st.warning("Graph file not found. Run a scan first.")
        return

    net = Network(height="450px", width="100%", bgcolor="#0f172a", font_color="white")
    net.barnes_hut()

    # Load graph from GEXF file using networkx.
    import networkx as nx

    graph = nx.read_gexf(graph_path)
    for node_id, data in graph.nodes(data=True):
        net.add_node(node_id, label=data.get("label", node_id), title=data.get("type", ""))
    for source, target, data in graph.edges(data=True):
        net.add_edge(source, target, title=data.get("source", ""))

    html_path = graph_path.with_suffix(".html")
    net.write_html(html_path)
    components.html(html_path.read_text(encoding="utf-8"), height=480, scrolling=True)
