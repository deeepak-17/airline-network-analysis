"""Louvain + Leiden (10 seeds each), stability, resolution sweep, weighted sensitivity."""
import itertools, json, time
import numpy as np, pandas as pd, networkx as nx, igraph as ig, leidenalg
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from common4 import *

SEEDS = list(range(10))
u = load_undirected(); nodes = list(u.nodes)
print("nodes", u.number_of_nodes(), "edges", u.number_of_edges(), "components", nx.number_connected_components(u))
Q = lambda lab: nx.community.modularity(u, labels_to_comms(lab))

def louvain(seed, res=1.0, weight="weight"):
    return community_louvain.best_partition(u, resolution=res, random_state=seed, weight=weight)
ig_g = ig.Graph(n=len(nodes), edges=[(nodes.index(a), nodes.index(b)) for a, b in u.edges()]) if False else None
idx = {n: i for i, n in enumerate(nodes)}
ig_g = ig.Graph(n=len(nodes), edges=[(idx[a], idx[b]) for a, b in u.edges()])
def leiden(seed, res=None):
    if res is None:
        p = leidenalg.find_partition(ig_g, leidenalg.ModularityVertexPartition, seed=seed, n_iterations=-1)
    else:  # resolution-parameter modularity
        p = leidenalg.find_partition(ig_g, leidenalg.RBConfigurationVertexPartition, seed=seed,
                                     resolution_parameter=res, n_iterations=-1)
    return {nodes[i]: c for c, mem in enumerate(p) for i in mem}

runs = {}
for name, fn in [("Louvain", louvain), ("Leiden", leiden)]:
    t = time.time(); runs[name] = [fn(s) for s in SEEDS]; print(name, "10 runs", round(time.time() - t, 1), "s")

# --- stability across seeds
rows = []
for name, parts in runs.items():
    qs = [Q(p) for p in parts]; ks = [len(set(p.values())) for p in parts]
    pair = [(nmi([a[n] for n in nodes], [b[n] for n in nodes]), ari([a[n] for n in nodes], [b[n] for n in nodes]))
            for a, b in itertools.combinations(parts, 2)]
    rows.append(dict(method=name, Q_mean=np.mean(qs), Q_std=np.std(qs), Q_min=min(qs), Q_max=max(qs),
        k_mean=np.mean(ks), k_min=min(ks), k_max=max(ks),
        pairwise_NMI_mean=np.mean([p[0] for p in pair]), pairwise_NMI_min=min(p[0] for p in pair),
        pairwise_ARI_mean=np.mean([p[1] for p in pair])))
    print(rows[-1])
pd.DataFrame(rows).to_csv(OUT / "disjoint_stability.csv", index=False)

# cross-method agreement (seed 0 vs seed 0, and best-Q partitions)
best = {nm: max(parts, key=Q) for nm, parts in runs.items()}
la = [best["Louvain"][n] for n in nodes]; lb = [best["Leiden"][n] for n in nodes]
cross = dict(NMI_best_louvain_vs_best_leiden=nmi(la, lb), ARI=ari(la, lb))
print(cross); json.dump(cross, open(OUT / "louvain_vs_leiden.json", "w"))

# --- evaluate best-Q partition of each method with the common metrics
stats = [cover_stats(u, labels_to_comms(best[nm]), f"{nm} (best-Q of 10 seeds)") for nm in best]
for s in stats: s["modularity_Q"] = Q(best[s["method"].split()[0]])
# --- resolution sweep (Leiden RB-configuration); shows resolution limit / granularity dependence
sweep = []
for res in [0.25, 0.5, 1.0, 2.0, 4.0]:
    p = leiden(0, res); c = labels_to_comms(p); phi = conductance_per_community(u, c)
    sweep.append(dict(resolution=res, n_communities=len(c), Q_at_gamma1=Q(p), size_max=max(map(len, c)),
        n_singletons=sum(len(x) == 1 for x in c), mean_conductance=float(np.nanmean(phi))))
pd.DataFrame(sweep).to_csv(OUT / "leiden_resolution_sweep.csv", index=False); print(pd.DataFrame(sweep))
# --- weighted sensitivity: weight = #airline-route rows summed over both directions
g = build_digraph()
for a, b in u.edges():
    u[a][b]["w"] = g.get_edge_data(a, b, {}).get("route_count", 0) + g.get_edge_data(b, a, {}).get("route_count", 0)
wp = [louvain(s, weight="w") for s in SEEDS[:5]]
wq = [nx.community.modularity(u, labels_to_comms(p), weight="w") for p in wp]
wnmi = np.mean([nmi([best["Louvain"][n] for n in nodes], [p[n] for n in nodes]) for p in wp])
print("weighted Louvain: Qw", np.mean(wq), "k", [len(set(p.values())) for p in wp], "NMI vs unweighted best", wnmi)
json.dump(dict(Qw_mean=float(np.mean(wq)), k=[len(set(p.values())) for p in wp], nmi_vs_unweighted=float(wnmi)),
          open(OUT / "louvain_weighted_sensitivity.json", "w"))
pd.DataFrame(stats).to_csv(OUT / "disjoint_eval.csv", index=False); print(pd.DataFrame(stats).T)
# save best partitions + per-seed partitions for downstream
for nm, p in best.items(): pd.Series(p, name="community").rename_axis("node").to_csv(OUT / f"partition_{nm.lower()}_best.csv")
pd.DataFrame({f"{nm}_s{s}": pd.Series(p) for nm, ps in runs.items() for s, p in enumerate(ps)}).to_csv(OUT / "partitions_all_seeds.csv")
