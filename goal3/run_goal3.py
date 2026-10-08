"""Goal 3: PageRank, Personalised PageRank (India), full sparse-matrix SimRank."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, networkx as nx, scipy.sparse as sp
from scipy.stats import spearmanr, kendalltau
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.load_graph import build_digraph

OUT = ROOT / "outputs" / "goal3"
OUT.mkdir(parents=True, exist_ok=True)
ALPHA, TOPK = 0.85, 15
g = build_digraph()
nodes = list(g.nodes)
print("nodes", g.number_of_nodes(), "edges", g.number_of_edges())
missing_meta = [n for n in nodes if "country" not in g.nodes[n]]
print("nodes w/o airports.dat metadata:", len(missing_meta))

# ---- PageRank (dangling nodes handled by networkx: uniform redistribution)
pr = nx.pagerank(g, alpha=ALPHA, tol=1e-12, max_iter=500)
prw = nx.pagerank(g, alpha=ALPHA, weight="route_count", tol=1e-12, max_iter=500)
# ---- Personalised PR: restart uniformly on Indian airports
india = [n for n in nodes if g.nodes[n].get("country") == "India"]
pers = {n: (1.0 / len(india) if n in set(india) else 0.0) for n in nodes}
ppr = nx.pagerank(g, alpha=ALPHA, personalization=pers, dangling=pers, tol=1e-12, max_iter=500)
print("India airports in graph:", len(india))

# ---- Feature table
btw = nx.betweenness_centrality(g, k=500, seed=42)  # APPROXIMATE (500 sampled sources)
df = pd.DataFrame({
    "airport": nodes,
    "name": [g.nodes[n].get("name", "") for n in nodes],
    "country": [g.nodes[n].get("country", "") for n in nodes],
    "in_deg": [g.in_degree(n) for n in nodes],
    "out_deg": [g.out_degree(n) for n in nodes],
    "pagerank": [pr[n] for n in nodes],
    "pagerank_w": [prw[n] for n in nodes],
    "ppr_india": [ppr[n] for n in nodes],
    "betweenness_approx": [btw[n] for n in nodes],
})
ug = g.to_undirected()
df["degree_undirected"] = [ug.degree(n) for n in nodes]
df["total_deg"] = df.in_deg + df.out_deg
for c in ["degree_undirected", "pagerank", "pagerank_w", "ppr_india", "betweenness_approx"]:
    df["rank_" + c] = df[c].rank(ascending=False, method="min").astype(int)
df.to_csv(OUT / "centrality_table.csv", index=False)

def top(col, k=TOPK, extra=()):
    return df.sort_values(col, ascending=False).head(k)[["airport", "name", "country", *extra, col]]
for col, ex in [("pagerank", ["in_deg", "out_deg"]), ("pagerank_w", []), ("ppr_india", []),
                ("degree_undirected", []), ("betweenness_approx", [])]:
    t = top(col, extra=ex); t.to_csv(OUT / f"top15_{col}.csv", index=False)
    print(f"\n== top-{TOPK} {col}\n", t.to_string(index=False))
# PPR excluding Indian airports: what global airports matter to India
t = df[~df.airport.isin(india)].sort_values("ppr_india", ascending=False).head(TOPK)
t = t.assign(ppr_over_pr=t.ppr_india / t.pagerank)[["airport", "name", "country", "ppr_india", "pagerank", "ppr_over_pr"]]
t.to_csv(OUT / "top15_ppr_india_foreign.csv", index=False); print("\n== top-15 non-Indian by PPR\n", t.to_string(index=False))
# lift: PPR/PR
df["ppr_lift"] = df.ppr_india / df.pagerank
t = df[(df.degree_undirected >= 20)].sort_values("ppr_lift", ascending=False).head(TOPK)[["airport", "name", "country", "ppr_lift"]]
t.to_csv(OUT / "top15_ppr_lift_deg20.csv", index=False); print("\n== top-15 PPR/PR lift (deg>=20)\n", t.to_string(index=False))

# ---- Correlations
rows = []
for a, b in [("degree_undirected", "pagerank"), ("in_deg", "pagerank"), ("total_deg", "pagerank"),
             ("degree_undirected", "pagerank_w"), ("pagerank", "pagerank_w"), ("pagerank", "betweenness_approx"),
             ("degree_undirected", "betweenness_approx"), ("pagerank", "ppr_india"), ("degree_undirected", "ppr_india")]:
    rows.append((a, b, spearmanr(df[a], df[b])[0], kendalltau(df[a], df[b])[0]))
corr = pd.DataFrame(rows, columns=["a", "b", "spearman", "kendall"]); corr.to_csv(OUT / "rank_correlations.csv", index=False)
print("\n", corr.to_string(index=False))
ov = {k: len(set(top("pagerank", k).airport) & set(top("degree_undirected", k).airport)) for k in (10, 15, 50, 100)}
print("top-k overlap PR vs degree:", ov)

# ---- Rank gap + neighbour-quality test
df["rank_gap"] = df.degree_undirected.rank(ascending=False, method="average") - df.pagerank.rank(ascending=False, method="average")  # >0: PR rank better than degree rank
nbr_pr = {n: np.mean([pr[m] for m in g.predecessors(n)]) if g.in_degree(n) else 0.0 for n in nodes}
df["mean_inneighbour_pr"] = [nbr_pr[n] for n in nodes]
# degree-adjusted residual of log PR on log in-degree (PR is mainly driven by in-links)
m = df.in_deg > 0
x = np.log(df.loc[m, "in_deg"]); y = np.log(df.loc[m, "pagerank"])
b1, b0 = np.polyfit(x, y, 1)
df.loc[m, "pr_resid"] = y - (b0 + b1 * x)
print(f"\nlog PR ~ log in_deg: slope {b1:.3f}, R^2 {np.corrcoef(x,y)[0,1]**2:.3f}")
mm = df[m & (df.in_deg >= 5)].copy()
r1 = spearmanr(mm.pr_resid, np.log(mm.mean_inneighbour_pr))
# null: shuffle neighbour quality within in-degree strata to confirm the link isn't an in-degree artefact
rng = np.random.default_rng(42)
mm["bin"] = pd.qcut(mm.in_deg, 10, duplicates="drop")
null = []
for _ in range(1000):
    sh = mm.groupby("bin", observed=True)["mean_inneighbour_pr"].transform(lambda s: rng.permutation(s.values))
    null.append(spearmanr(mm.pr_resid, np.log(sh))[0])
print(f"Spearman(PR residual, log mean in-neighbour PR), in_deg>=5: rho={r1[0]:.3f} p={r1[1]:.2g}; "
      f"stratified-shuffle null mean={np.mean(null):.3f} sd={np.std(null):.3f}")
# recipient/donor split
mod = mm[(mm.degree_undirected.between(5, 60))]
q = mod.pr_resid.quantile([.1, .9])
hi, lo = mod[mod.pr_resid >= q[.9]], mod[mod.pr_resid <= q[.1]]
print(f"moderate deg(5-60) n={len(mod)}: mean in-nbr PR top-decile resid {hi.mean_inneighbour_pr.mean():.2e} vs bottom {lo.mean_inneighbour_pr.mean():.2e}; "
      f"mean in-deg {hi.in_deg.mean():.1f} vs {lo.in_deg.mean():.1f}")
# Mechanism: a node receives PR(m)/outdeg(m) from each in-neighbour m -> neighbour *exclusivity* matters, not only neighbour PR.
df["mean_log_nbr_outdeg"] = [np.mean([np.log(g.out_degree(p)) for p in g.predecessors(n)]) if g.in_degree(n) else np.nan for n in nodes]
df["mean_nbr_share"] = [np.mean([pr[p] / g.out_degree(p) for p in g.predecessors(n)]) if g.in_degree(n) else np.nan for n in nodes]
mm = df[df.in_deg >= 5].copy()
for c in ["mean_log_nbr_outdeg", "mean_nbr_share"]:
    print(f"Spearman(PR resid, {c}) = {spearmanr(mm.pr_resid, mm[c])[0]:.3f}")
def _r2(cols):
    X = np.column_stack([np.ones(len(mm))] + [mm[c].values for c in cols]); y_ = np.log(mm.pagerank.values)
    beta, *_ = np.linalg.lstsq(X, y_, rcond=None); res = y_ - X @ beta
    return 1 - res.var() / y_.var(), beta
mm["log_in"] = np.log(mm.in_deg); mm["log_nbrpr"] = np.log(mm.mean_inneighbour_pr); mm["log_nbrshare"] = np.log(mm.mean_nbr_share)
for cols in (["log_in"], ["log_in", "log_nbrpr"], ["log_in", "mean_log_nbr_outdeg"], ["log_in", "log_nbrshare"]):
    r2, b = _r2(cols); print(f"OLS log PR ~ {cols}: R^2={r2:.3f} coefs={np.round(b,3)}")
# share of PR-mass-from-top-hubs: fraction of in-neighbours that are in top-50 PR
top50 = set(top("pagerank", 50).airport)
df["frac_nbr_top50"] = [np.mean([p in top50 for p in g.predecessors(n)]) if g.in_degree(n) else 0 for n in nodes]
print("Spearman(PR resid, frac in-nbrs in PR top50):", spearmanr(df.loc[mm.index, "pr_resid"], df.loc[mm.index, "frac_nbr_top50"]))
cand = df[df.degree_undirected.between(20, 120)]  # "moderate degree" band (ranks ~100-600)
cols = ["airport", "name", "country", "degree_undirected", "rank_degree_undirected", "rank_pagerank", "rank_gap", "mean_inneighbour_pr", "frac_nbr_top50"]
gain = cand.sort_values("rank_gap", ascending=False).head(15)[cols]; gain.to_csv(OUT / "rankgap_pr_better_than_degree.csv", index=False)
loss = cand.sort_values("rank_gap").head(15)[cols]; loss.to_csv(OUT / "rankgap_pr_worse_than_degree.csv", index=False)
print("\n== moderate degree, PR rank >> degree rank\n", gain.to_string(index=False))
print("\n== moderate degree, PR rank << degree rank\n", loss.to_string(index=False))
df.to_csv(OUT / "centrality_table.csv", index=False)

# ---- Scatter
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sc = ax[0].scatter(df.degree_undirected, df.pagerank, s=6, alpha=.4, c=np.log10(df.mean_inneighbour_pr + 1e-9), cmap="viridis")
ax[0].set(xscale="log", yscale="log", xlabel="undirected degree", ylabel="PageRank", title="Degree vs PageRank (colour: log10 mean in-neighbour PR)")
plt.colorbar(sc, ax=ax[0])
for _, r in df.nlargest(8, "pagerank").iterrows(): ax[0].annotate(r.airport, (r.degree_undirected, r.pagerank), fontsize=8)
for _, r in pd.concat([gain.head(4), loss.head(4)]).iterrows(): ax[0].annotate(r.airport, (r.degree_undirected, df.set_index("airport").loc[r.airport, "pagerank"]), fontsize=7, color="red")
mm2 = mm
ax[1].scatter(np.log10(mm2.mean_inneighbour_pr), mm2.pr_resid, s=6, alpha=.4)
ax[1].set(xlabel="log10 mean PageRank of in-neighbours", ylabel="log PR residual given in-degree",
          title=f"Does neighbour quality explain PR beyond degree? rho={r1[0]:.2f}")
plt.tight_layout(); plt.savefig(OUT / "degree_vs_pagerank.png", dpi=150); plt.close()

# ---- SimRank (full, sparse matrix iteration). Uses DIRECTED in-neighbours (original definition).
# S_{t+1} = C * W^T S_t W with diagonal reset to 1, W column-normalised adjacency (W[i,j]=A[i,j]/indeg(j)).
# This is the standard matrix approximation of Jeh-Widom (diag forced to 1), run for K iterations, not exact limit.
C, K = 0.8, 30
idx = {n: i for i, n in enumerate(nodes)}
A = nx.to_scipy_sparse_array(g, nodelist=nodes, weight=None, format="csc", dtype=float)
indeg = np.asarray(A.sum(axis=0)).ravel(); indeg[indeg == 0] = 1
W = sp.csc_matrix(A @ sp.diags(1.0 / indeg))
S = np.eye(len(nodes))
for it in range(K):
    Sn = C * (W.T @ (W.T @ S).T).T  # = C * W^T S W  (S symmetric)
    Sn = np.asarray(Sn); np.fill_diagonal(Sn, 1.0)
    delta = np.abs(Sn - S).max(); S = Sn
    if delta < 1e-4: print('converged'); break
    print(f"SimRank iter {it+1}: max change {delta:.2e}")
S = (S + S.T) / 2
np.save(OUT / "simrank.npy", S.astype(np.float32))
iu = np.triu_indices(len(nodes), 1)
vals = S[iu]
indeg_arr = np.array([g.in_degree(n) for n in nodes]); okp = (indeg_arr[iu[0]] >= 3) & (indeg_arr[iu[1]] >= 3)  # drop trivial indeg-1 pairs (sim=C)
order = np.argsort(-np.where(okp, vals, -1))[:30]
pairs = pd.DataFrame({"a": [nodes[iu[0][o]] for o in order], "b": [nodes[iu[1][o]] for o in order], "simrank": vals[order]})
for c in "ab":
    pairs[c + "_name"] = [g.nodes[x].get("name", "") for x in pairs[c]]; pairs[c + "_country"] = [g.nodes[x].get("country", "") for x in pairs[c]]
pairs["a_indeg"] = [g.in_degree(x) for x in pairs.a]; pairs["b_indeg"] = [g.in_degree(x) for x in pairs.b]
pairs["km"] = [np.nan] * len(pairs)
def hav(a, b):
    la1, lo1, la2, lo2 = map(np.radians, [g.nodes[a]["lat"], g.nodes[a]["lon"], g.nodes[b]["lat"], g.nodes[b]["lon"]])
    h = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 12742 * np.arcsin(np.sqrt(h))
pairs["km"] = [hav(a, b) if "lat" in g.nodes[a] and "lat" in g.nodes[b] else np.nan for a, b in zip(pairs.a, pairs.b)]
pairs.to_csv(OUT / "simrank_top_pairs.csv", index=False); print("\n== top SimRank pairs\n", pairs.head(20).to_string(index=False))
# Is SimRank related to distance? correlation over random pairs with S>0
rs = np.random.default_rng(42); sel = rs.choice(len(vals), 200000, replace=False)
sel = sel[vals[sel] > 0]
d = np.array([hav(nodes[iu[0][s]], nodes[iu[1][s]]) if "lat" in g.nodes[nodes[iu[0][s]]] and "lat" in g.nodes[nodes[iu[1][s]]] else np.nan for s in sel[:20000]])
ok = ~np.isnan(d); print("Spearman(SimRank, distance km) over sampled nonzero pairs:", spearmanr(vals[sel[:20000]][ok], d[ok])[0], "n", ok.sum(),
      "| fraction nonzero pairs:", float((vals > 0).mean()))
rowsq = []
for q in ["DEL", "BOM", "JFK", "LHR", "ATL", "BLR"]:
    if q not in idx: continue
    s = S[idx[q]].copy(); s[idx[q]] = -1; s[indeg_arr < 3] = -1  # require in-degree>=3 candidates
    for o in np.argsort(-s)[:8]:
        rowsq.append((q, nodes[o], g.nodes[nodes[o]].get("name", ""), g.nodes[nodes[o]].get("country", ""), s[o], g.in_degree(nodes[o])))
nn = pd.DataFrame(rowsq, columns=["query", "airport", "name", "country", "simrank", "in_deg"]); nn.to_csv(OUT / "simrank_nearest.csv", index=False)
print("\n== nearest neighbours\n", nn.to_string(index=False))
# overlap between SimRank-NN and Jaccard of in-neighbour sets sanity
for q in ["DEL", "BOM"]:
    top_nn = nn[nn["query"] == q].airport.tolist()
    qs = set(g.predecessors(q)); print(q, "Jaccard of in-neighbours with top SimRank NNs:", [round(len(qs & set(g.predecessors(a))) / max(1, len(qs | set(g.predecessors(a)))), 2) for a in top_nn])
