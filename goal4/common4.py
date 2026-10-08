"""Shared helpers for Goal 4: data loading, partition metrics (disjoint + overlapping)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.load_graph import build_digraph, build_undirected, load_routes  # noqa: E402
from geo_maps import continent_of  # noqa: E402
OUT = ROOT / "outputs" / "goal4"; OUT.mkdir(parents=True, exist_ok=True)

def load_undirected():
    """Undirected PROJECTION of the directed graph (A-B edge iff A->B or B->A exists).
    Loses direction/reciprocity; unweighted (route_count discarded). Self-loops already removed."""
    return build_undirected(build_digraph())

def node_table(u):
    rows = []
    for n, d in u.nodes(data=True):
        c = d.get("country")
        rows.append(dict(node=n, country=c, lat=d.get("lat"), lon=d.get("lon"),
                         continent=continent_of(c, d.get("lon")) if c else None, degree=u.degree(n)))
    return pd.DataFrame(rows).set_index("node")

def conductance_per_community(u, comms):
    """phi(C)=cut(C)/min(vol(C), 2m-vol(C)); works for overlapping covers too. comms: list of sets."""
    two_m = 2 * u.number_of_edges(); out = []
    for c in comms:
        c = set(c); vol = sum(u.degree(n) for n in c)
        inside = sum(1 for n in c for v in u[n] if v in c) / 2
        cut = vol - 2 * inside; denom = min(vol, two_m - vol)
        out.append(cut / denom if denom > 0 else np.nan)
    return np.array(out)

def cover_stats(u, comms, label):
    """Common summary for a disjoint partition or an overlapping cover."""
    comms = [set(c) for c in comms]
    sizes = np.array([len(c) for c in comms]); phi = conductance_per_community(u, comms)
    covered = set().union(*comms) if comms else set()
    memb = {}
    for c in comms:
        for n in c: memb[n] = memb.get(n, 0) + 1
    ok = ~np.isnan(phi)
    vols = np.array([sum(u.degree(n) for n in c) for c in comms])
    return dict(method=label, n_communities=len(comms), n_covered=len(covered),
        node_coverage=len(covered) / u.number_of_nodes(), size_min=sizes.min(), size_median=float(np.median(sizes)),
        size_max=sizes.max(), n_size_ge10=int((sizes >= 10).sum()),
        mean_conductance=float(np.nanmean(phi)), median_conductance=float(np.nanmedian(phi)),
        size_weighted_conductance=float(np.average(phi[ok], weights=sizes[ok])),
        vol_weighted_conductance=float(np.average(phi[ok], weights=vols[ok])),
        n_overlap_nodes=sum(1 for v in memb.values() if v > 1),
        mean_memberships=float(np.mean(list(memb.values()))) if memb else np.nan,
        max_memberships=max(memb.values()) if memb else 0,
        shen_EQ=shen_eq(u, comms))

def shen_eq(u, comms):
    """Overlapping (extended) modularity EQ of Shen et al. 2009:
    EQ = 1/2m * sum_c sum_{i,j in c} 1/(O_i O_j) [A_ij - k_i k_j / 2m], O_i = #communities of i.
    For a disjoint partition (O_i=1) it equals Newman Q. Nodes in no community contribute nothing
    (they are NOT penalised), so EQ of a partial cover is not directly comparable to a full
    partition's Q -- see coverage. Known limitation: EQ is only one of several overlapping-Q variants."""
    nodes = list(u.nodes); idx = {n: i for i, n in enumerate(nodes)}
    O = np.zeros(len(nodes)); 
    for c in comms:
        for n in c: O[idx[n]] += 1
    k = np.array([u.degree(n) for n in nodes], float); two_m = 2.0 * u.number_of_edges(); tot = 0.0
    for c in comms:
        ii = np.array([idx[n] for n in c]); w = 1.0 / O[ii]
        sub = nx.to_scipy_sparse_array(u.subgraph(c), nodelist=list(c), format="csr")
        wc = np.array([1.0 / O[idx[n]] for n in c]); kc = k[ii]
        a_term = wc @ (sub @ wc)
        k_term = (wc * kc).sum() ** 2 / two_m
        tot += a_term - k_term
    return float(tot / two_m)

def labels_to_comms(labels):
    d = {}
    for n, l in labels.items(): d.setdefault(l, set()).add(n)
    return list(d.values())
