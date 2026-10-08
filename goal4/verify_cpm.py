"""Check that the sparse CPM equals networkx k_clique_communities on small random graphs."""
import sys; sys.path.insert(0, "goal4")
import networkx as nx, numpy as np
src = open("goal4/run_overlap.py").read(); start = src.index("def cpm_sparse"); end = src.index("rows = []; saved")
import scipy; ns = {"np": np}; exec(src[start:end], ns)
ok = True
for seed in range(20):
    g = nx.gnp_random_graph(60, 0.12, seed=seed); cl = list(nx.find_cliques(g))
    for k in (3, 4, 5):
        a = {frozenset(c) for c in nx.community.k_clique_communities(g, k)}; b = {frozenset(c) for c in ns["cpm_sparse"](g, cl, k)}
        ok &= a == b
print("sparse CPM == networkx on 60 cases:", ok)
