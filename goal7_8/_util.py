"""Shared helpers for goals 7/8. Graph = shared loader; igraph copies for speed."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "goal7_8"
OUT.mkdir(parents=True, exist_ok=True)
import igraph as ig
import networkx as nx
from common.load_graph import build_digraph

def load():
    """Return (nx DiGraph, igraph directed, igraph undirected, node list). Same node order in both igraphs."""
    g = build_digraph()
    nodes = list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in g.edges]
    gd = ig.Graph(n=len(nodes), edges=edges, directed=True)
    gd.vs["name"] = nodes
    gd.es["w"] = [g[u][v]["route_count"] for u, v in g.edges]
    gu = gd.copy(); gu.to_undirected(mode="collapse"); gu.simplify()
    return g, gd, gu, nodes
