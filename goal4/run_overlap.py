"""Clique Percolation (networkx k_clique_communities) for several k on the undirected projection.
Complexity: maximal-clique enumeration is worst-case exponential; here it is measured and reported.
CPM is deterministic (no seeds). Nodes in no k-clique community are left uncovered (coverage < 1)."""
import time, json
import numpy as np, pandas as pd, networkx as nx
from common4 import *
u = load_undirected()
t = time.time(); cliques = list(nx.find_cliques(u)); t_cl = time.time() - t
sz = np.array([len(c) for c in cliques])
print(f"maximal cliques: {len(cliques)}, max size {sz.max()}, enumeration {t_cl:.1f}s")
json.dump(dict(n_maximal_cliques=len(cliques), max_clique_size=int(sz.max()), seconds=t_cl,
               size_hist={int(k): int((sz == k).sum()) for k in np.unique(sz)}), open(OUT / "cpm_cliques.json", "w"))

def cpm_sparse(g, cliques, k):
    """Exact CPM: k-clique communities = unions of maximal cliques (size>=k) linked when they share >=k-1 nodes.
    nx.community.k_clique_communities took >2 min for k=3 here (hub nodes lie in thousands of cliques and it
    scans adjacent cliques pairwise), so overlaps are computed with one sparse product B @ B.T instead.
    Verified equal to networkx on random graphs in verify_cpm.py."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    cl = [c for c in cliques if len(c) >= k]
    if not cl: return []
    idx = {n: i for i, n in enumerate(g.nodes)}
    rows_ = np.repeat(np.arange(len(cl)), [len(c) for c in cl]); cols_ = np.array([idx[n] for c in cl for n in c])
    B = sp.csr_matrix((np.ones(len(rows_), dtype=np.int32), (rows_, cols_)), shape=(len(cl), len(idx)))
    O = (B @ B.T).tocoo(); m = (O.data >= k - 1) & (O.row != O.col)
    A = sp.csr_matrix((np.ones(m.sum()), (O.row[m], O.col[m])), shape=(len(cl), len(cl)))
    _, lab = connected_components(A, directed=False)
    out = {}
    for c, l in zip(cl, lab): out.setdefault(l, set()).update(c)
    return list(out.values())

rows = []; saved = {}
for k in [3, 4, 5, 6, 8]:
    if k > sz.max(): continue
    t = time.time(); comms = [set(c) for c in cpm_sparse(u, cliques, k)]
    dt = time.time() - t
    if not comms: continue
    s = cover_stats(u, comms, f"CPM k={k}"); s["k"] = k; s["seconds"] = dt; rows.append(s); saved[k] = comms
    print(k, len(comms), round(dt, 1), "s", {x: s[x] for x in ["node_coverage", "shen_EQ", "mean_conductance", "n_overlap_nodes"]})
pd.DataFrame(rows).to_csv(OUT / "overlap_eval.csv", index=False)
recs = [dict(k=k, community=i, node=n) for k, cs in saved.items() for i, c in enumerate(cs) for n in sorted(c)]
pd.DataFrame(recs).to_csv(OUT / "cpm_memberships.csv", index=False)
print(pd.DataFrame(rows).T)
