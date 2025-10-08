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

    # Load the generated GEXF graph using networkx and render via PyVis.
    import networkx as nx

    graph = nx.read_gexf(graph_path)
    for node_id, data in graph.nodes(data=True):
        net.add_node(node_id, label=data.get("label", node_id), title=data.get("type", ""))
    for source, target, data in graph.edges(data=True):
        net.add_edge(source, target, title=data.get("source", ""))

    html = net.generate_html(notebook=False)
    components.html(html, height=480, scrolling=True)
