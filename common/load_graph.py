"""Shared loader so every goal uses the identical graph. Run with .venv/bin/python."""
from pathlib import Path
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MISSING = r"\N"


def load_airports() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "airports.dat", na_values=[MISSING], keep_default_na=False)
    return df[df["iata"].notna() & (df["iata"] != "")].drop_duplicates("iata").set_index("iata")


def load_routes() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "routes.dat", na_values=[MISSING], keep_default_na=False)
    df = df.dropna(subset=["source", "dest"])
    return df[df["source"] != df["dest"]]


def build_digraph() -> nx.DiGraph:
    """Directed, unweighted-structure graph; edge attr route_count = #airline-routes."""
    routes = load_routes()
    airports = load_airports()
    g = nx.DiGraph()
    for (s, d), n in routes.groupby(["source", "dest"]).size().items():
        g.add_edge(s, d, route_count=int(n))
    for node in g.nodes:
        if node in airports.index:
            row = airports.loc[node]
            g.nodes[node].update(name=row["name"], city=row["city"], country=row["country"],
                                 lat=float(row["lat"]), lon=float(row["lon"]))
    return g


def build_undirected(g: nx.DiGraph | None = None) -> nx.Graph:
    return (g or build_digraph()).to_undirected()


def largest_wcc(g: nx.DiGraph) -> nx.DiGraph:
    return g.subgraph(max(nx.weakly_connected_components(g), key=len)).copy()
