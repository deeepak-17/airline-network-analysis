"""Goal 1: structural characterisation."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.load_graph import build_digraph, build_undirected, largest_wcc, load_routes

OUT = ROOT / "outputs" / "goal1"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
R = {}

# ---- raw data audit (documented preprocessing) ----
import pandas as pd
raw = pd.read_csv(ROOT / "routes.dat", na_values=[r"\N"], keep_default_na=False)
R["raw_rows"] = len(raw)
R["raw_missing_src_or_dest"] = int(raw[["source", "dest"]].isna().sum().any() and raw[["source","dest"]].isna().any(axis=1).sum())
R["raw_self_loops"] = int((raw.source == raw.dest).sum())
R["raw_exact_duplicate_rows"] = int(raw.duplicated().sum())
R["rows_after_loader"] = len(load_routes())

G = build_digraph()
U = build_undirected(G)
n, m = G.number_of_nodes(), G.number_of_edges()
R.update(n=n, m_directed=m, m_undirected=U.number_of_edges(),
         density_directed=nx.density(G), density_undirected=nx.density(U),
         avg_in_deg=m/n, avg_total_deg_undirected=2*U.number_of_edges()/n,
         self_loops_in_graph=nx.number_of_selfloops(G),
         nodes_missing_metadata=sum(1 for v in G if "name" not in G.nodes[v]))
name = {v: G.nodes[v].get("name", "?") for v in G}
country = {v: G.nodes[v].get("country", "?") for v in G}

# ---- components ----
wcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
scc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
R.update(n_wcc=len(wcc), largest_wcc=len(wcc[0]), n_scc=len(scc), largest_scc=len(scc[0]),
         scc_singletons=sum(1 for c in scc if len(c) == 1))
LW = largest_wcc(G); LU = U.subgraph(max(nx.connected_components(U), key=len)).copy()
LS = G.subgraph(scc[0]).copy()
# exact distances: BFS from every node, n~3.4k -> feasible
def paths(g):
    tot = cnt = 0; diam = 0
    for s, d in nx.all_pairs_shortest_path_length(g):
        for t, l in d.items():
            if t != s: tot += l; cnt += 1; diam = max(diam, l)
    return tot / cnt, diam, cnt
R["undirected_LCC_avg_path"], R["undirected_LCC_diameter"], _ = paths(LU)
R["directed_LSCC_avg_path"], R["directed_LSCC_diameter"], _ = paths(LS)
a, dm, c = paths(G)
R["directed_reachable_pair_frac"] = c / (n*(n-1)); R["directed_avg_path_reachable_pairs"] = a
R["directed_diameter_over_reachable"] = dm

# ---- degree distribution ----
indeg = pd.Series(dict(G.in_degree())); outdeg = pd.Series(dict(G.out_degree()))
totdeg = pd.Series(dict(U.degree()))
for k, s in [("in", indeg), ("out", outdeg), ("total_undirected", totdeg)]:
    R[f"deg_{k}"] = dict(mean=float(s.mean()), median=float(s.median()), max=int(s.max()),
                         max_airport=s.idxmax(), std=float(s.std()))
R["frac_total_deg_1"] = float((totdeg == 1).mean())
def ccdf(x):
    x = np.sort(np.asarray(x)); y = 1 - np.arange(len(x)) / len(x); return x, y
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a_, (lab, s) in zip(ax, [("in-degree", indeg), ("out-degree", outdeg), ("total degree (undirected)", totdeg)]):
    vc = s[s > 0].value_counts(normalize=True).sort_index()
    a_.loglog(vc.index, vc.values, "o", ms=3, alpha=.6, label="P(k)")
    x, y = ccdf(s[s > 0]); a_.loglog(x, y, "-", color="C3", label="CCDF P(K>=k)")
    a_.set(title=lab, xlabel="k", ylabel="probability"); a_.legend()
fig.suptitle("Degree distributions (log-log): is the tail a straight line?"); fig.tight_layout()
fig.savefig(OUT / "degree_distributions.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(5, 4.2))
ax.hist(totdeg, bins=60); ax.set_yscale("log"); ax.set(xlabel="total degree", ylabel="airports (log)", title="Total degree histogram")
fig.tight_layout(); fig.savefig(OUT / "degree_histogram.png", dpi=150); plt.close(fig)

# ---- centralities ----
# Betweenness: exact Brandes O(nm) ~ 1.3e8 ops, feasible; run exactly on directed graph.
bc = nx.betweenness_centrality(G, normalized=True)
# Closeness: networkx directed closeness uses INCOMING distance & Wasserman-Faust
# scaling for disconnected graphs; also compute on undirected largest component.
cc_dir = nx.closeness_centrality(G, wf_improved=True)
cc_und = nx.closeness_centrality(LU)
# Eigenvector: undirected projection (converges). Directed (in-edge) fails on non-strongly-connected; use LSCC variant.
ev_und = nx.eigenvector_centrality_numpy(LU)  # LCC only; nodes outside LCC -> NaN (not comparable)
ev_dir_lscc = nx.eigenvector_centrality(LS, max_iter=5000, tol=1e-10)
dc_in = nx.in_degree_centrality(G); dc_out = nx.out_degree_centrality(G); dc_und = nx.degree_centrality(U)
cen = pd.DataFrame({"name": pd.Series(name), "country": pd.Series(country), "in_deg": indeg, "out_deg": outdeg,
    "deg_und": totdeg, "deg_cent_und": pd.Series(dc_und), "betweenness_dir": pd.Series(bc),
    "closeness_dir_wf": pd.Series(cc_dir), "closeness_und_LCC": pd.Series(cc_und),
    "eigenvector_und": pd.Series(ev_und), "eigenvector_dir_LSCC": pd.Series(ev_dir_lscc)})
cen.index.name = "iata"; cen.to_csv(OUT / "centralities_all.csv")
tops = []
for col in ["deg_und", "betweenness_dir", "closeness_und_LCC", "eigenvector_und"]:
    t = cen.sort_values(col, ascending=False).head(10)[["name", "country", col]].reset_index()
    t.insert(0, "measure", col); t.insert(1, "rank", range(1, 11)); t = t.rename(columns={col: "value"}); tops.append(t)
pd.concat(tops).to_csv(OUT / "top10_centralities.csv", index=False)
sp = cen[["deg_und", "betweenness_dir", "closeness_und_LCC", "eigenvector_und"]].corr(method="spearman")
sp.to_csv(OUT / "centrality_spearman.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].loglog(cen.deg_und, cen.betweenness_dir.clip(lower=1e-7), "o", ms=3, alpha=.5)
ax[0].set(xlabel="total degree", ylabel="betweenness (directed, clipped 1e-7)", title="Degree vs betweenness")
for v in cen.sort_values("betweenness_dir", ascending=False).head(6).index:
    ax[0].annotate(v, (cen.deg_und[v], cen.betweenness_dir[v]), fontsize=8)
ax[1].scatter(cen.deg_und, cen.eigenvector_und, s=5, alpha=.5); ax[1].set_xscale("log")
ax[1].set(xlabel="total degree", ylabel="eigenvector (undirected)", title="Degree vs eigenvector")
fig.tight_layout(); fig.savefig(OUT / "degree_vs_centralities.png", dpi=150); plt.close(fig)

# ---- assortativity ----
R["assort_undirected_degree"] = nx.degree_assortativity_coefficient(U)
for sx in ["in", "out"]:
    for sy in ["in", "out"]:
        R[f"assort_directed_{sx}_{sy}"] = nx.degree_assortativity_coefficient(G, x=sx, y=sy)
# Null reference: degree-preserving rewiring of U (double-edge swaps), 20 reps
rng = np.random.default_rng(SEED); nulls = []
for i in range(20):
    H = U.copy(); nx.double_edge_swap(H, nswap=10*U.number_of_edges(), max_tries=100*10*U.number_of_edges(), seed=int(rng.integers(1e9)))
    nulls.append(nx.degree_assortativity_coefficient(H))
R["assort_null_rewired_mean"], R["assort_null_rewired_std"] = float(np.mean(nulls)), float(np.std(nulls))
# average neighbour degree vs k
knn = nx.average_neighbor_degree(U); kk = pd.DataFrame({"k": totdeg, "knn": pd.Series(knn)})
g = kk.groupby("k").knn.mean()
fig, ax = plt.subplots(figsize=(5.5, 4.2)); ax.loglog(g.index, g.values, "o-", ms=4)
ax.set(xlabel="degree k", ylabel="mean neighbour degree <k_nn>(k)", title="Degree correlation: decreasing => disassortative")
fig.tight_layout(); fig.savefig(OUT / "knn_vs_k.png", dpi=150); plt.close(fig)
R["knn_slope_loglog"] = float(np.polyfit(np.log(g.index), np.log(g.values), 1)[0])

# ---- transitivity / clustering / reciprocity ----
R["transitivity_undirected"] = nx.transitivity(U)
R["avg_clustering_undirected"] = nx.average_clustering(U)
R["avg_clustering_undirected_excl_deg_lt2"] = float(np.mean([c for v, c in nx.clustering(U).items() if U.degree(v) >= 2]))
R["avg_clustering_directed_fagiolo"] = nx.average_clustering(G)  # networkx directed generalisation (Fagiolo)
# ER expected: p ~ density
R["ER_expected_clustering_approx"] = R["density_undirected"]
cl = pd.Series(nx.clustering(U)); cdeg = pd.DataFrame({"k": totdeg, "c": cl}).query("k>=2").groupby("k").c.mean()
fig, ax = plt.subplots(figsize=(5.5, 4.2)); ax.loglog(cdeg.index, cdeg.values, "o", ms=4)
ax.set(xlabel="degree k", ylabel="mean local clustering C(k)", title="Clustering vs degree")
fig.tight_layout(); fig.savefig(OUT / "clustering_vs_degree.png", dpi=150); plt.close(fig)
R["reciprocity_edge_fraction"] = nx.reciprocity(G)
mu = sum(1 for u, v in G.edges if G.has_edge(v, u)); R["reciprocal_directed_edges"] = mu
R["reciprocal_pairs"] = mu // 2
R["one_way_edges"] = m - mu
# reciprocity reference: for same density, ER expectation = density
R["reciprocity_ER_expectation"] = R["density_directed"]
# weighted (route_count) reciprocity: share of routes on reciprocated edges
tot_w = sum(d["route_count"] for _, _, d in G.edges(data=True))
rec_w = sum(d["route_count"] for u, v, d in G.edges(data=True) if G.has_edge(v, u))
R["reciprocity_route_weighted"] = rec_w / tot_w

# ---- k-core ----
Uc = U.copy(); Uc.remove_edges_from(nx.selfloop_edges(Uc))
core = pd.Series(nx.core_number(Uc)); R["max_core_undirected"] = int(core.max())
kmax = int(core.max())
rows = []
for k in range(0, kmax + 1):
    rows.append(dict(k=k, shell_size=int((core == k).sum()), core_size=int((core >= k).sum())))
pd.DataFrame(rows).to_csv(OUT / "kcore_sizes.csv", index=False)
inner = core[core == kmax].index
pd.DataFrame({"iata": inner, "name": [name[v] for v in inner], "country": [country[v] for v in inner],
              "deg_und": totdeg[inner].values, "betweenness": cen.betweenness_dir[inner].values}
             ).sort_values("deg_und", ascending=False).to_csv(OUT / "innermost_core.csv", index=False)
R["innermost_core_size"] = len(inner)
R["innermost_core_countries"] = pd.Series([country[v] for v in inner]).value_counts().to_dict()
R["innermost_core_is_clique_density"] = nx.density(Uc.subgraph(inner))
R["spearman_core_vs_degree"] = float(core.corr(totdeg.reindex(core.index), method="spearman"))
R["spearman_core_vs_betweenness"] = float(core.corr(cen.betweenness_dir.reindex(core.index), method="spearman"))
# directed in/out cores (in-degree and out-degree based)
cen["core_und"] = core; cen.to_csv(OUT / "centralities_all.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ks = pd.DataFrame(rows)
ax[0].bar(ks.k, ks.shell_size); ax[0].set(xlabel="k", ylabel="airports in k-shell", title="k-shell sizes")
ax[1].semilogy(ks.k, ks.core_size, "o-"); ax[1].set(xlabel="k", ylabel="airports in k-core", title="k-core sizes (nested)")
fig.tight_layout(); fig.savefig(OUT / "kcore.png", dpi=150); plt.close(fig)
# top-10 among max core by degree vs highest-degree airports not in innermost core
R["top10_deg_in_innermost_core"] = [v for v in totdeg.sort_values(ascending=False).head(10).index if v in set(inner)]

json.dump(R, open(OUT / "metrics.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float)); print(sp.round(3)); print(pd.concat(tops).to_string())
print(pd.DataFrame(rows).to_string())
print(pd.read_csv(OUT / "innermost_core.csv").to_string())
