"""Geographic + alliance interpretation of Leiden/Louvain/CPM communities, plots. Needs run_disjoint.py and run_overlap.py first."""
import numpy as np, pandas as pd, networkx as nx, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from common4 import *
from alliances import alliance_of, ALLIANCES
rng = np.random.default_rng(0)
u = load_undirected(); T = node_table(u)
part = {nm: pd.read_csv(OUT / f"partition_{nm}_best.csv", index_col=0)["community"] for nm in ["leiden", "louvain"]}
cpm = pd.read_csv(OUT / "cpm_memberships.csv"); ov = pd.read_csv(OUT / "overlap_eval.csv")
K_MAP = 4 if 4 in set(cpm.k) else int(cpm.k.min())
lab = T.dropna(subset=["country"]).index  # nodes lacking airports.dat metadata are excluded from geo scores
print("nodes with country:", len(lab), "of", len(T))

def purity(labels, truth, shuffle=False):
    df = pd.DataFrame(dict(c=labels, t=truth)).dropna()
    if shuffle: df["t"] = rng.permutation(df["t"].values)
    return df.groupby(["c", "t"]).size().groupby("c").max().sum() / len(df)

def geo_scores(name, labels):  # labels: Series node->community (hard)
    l = labels.reindex(lab).dropna(); r = dict(method=name, n_nodes_scored=len(l))
    for tgt in ["country", "continent"]:
        tr = T.loc[l.index, tgt]
        r[f"NMI_{tgt}"] = nmi(tr, l); r[f"ARI_{tgt}"] = ari(tr, l)
        r[f"purity_{tgt}"] = purity(l.values, tr.values)
        r[f"purity_{tgt}_shuffled_null"] = float(np.mean([purity(l.values, tr.values, True) for _ in range(50)]))
    return r
rows = [geo_scores("Leiden", part["leiden"]), geo_scores("Louvain", part["louvain"])]
# CPM hardened: each covered node -> its LARGEST community (ties -> lowest id); loses overlap information
for k, g in cpm.groupby("k"):
    size = g.groupby("community").size(); g = g.assign(sz=g["community"].map(size)).sort_values(["node", "sz", "community"], ascending=[True, False, True])
    hard = g.drop_duplicates("node").set_index("node")["community"]
    rows.append(geo_scores(f"CPM k={k} (hardened, covered nodes only)", hard))
    rows.append(geo_scores(f"  Leiden restricted to same nodes as CPM k={k}", part["leiden"].reindex(hard.index)))
geo = pd.DataFrame(rows); geo.to_csv(OUT / "geo_alignment.csv", index=False); print(geo.round(3).to_string())

# --- alliance analysis
R = load_routes().copy(); R["alliance"] = R["airline"].map(alliance_of)
base = R.alliance.value_counts(normalize=True); base_nc = R[R.codeshare.fillna("") != "Y"].alliance.value_counts(normalize=True)
print("global route-row alliance share:\n", base.round(3).to_dict(), "\nno-codeshare:", base_nc.round(3).to_dict())
def community_alliance(labels, name):
    R2 = R.assign(cs=R.source.map(labels), cd=R.dest.map(labels)); R2 = R2[(R2.cs == R2.cd) & R2.cs.notna()]
    out = []
    for c, g in R2.groupby("cs"):
        sh = g.alliance.value_counts(normalize=True); ali = sh.drop("None", errors="ignore")
        row = dict(community=c, n_intra_route_rows=len(g), **{f"share_{a}": sh.get(a, 0.0) for a in list(ALLIANCES) + ["None"]})
        row["dominant_alliance"] = ali.idxmax() if len(ali) else "None"
        row["dominant_enrichment_vs_global"] = (ali.max() / base[ali.idxmax()]) if len(ali) else np.nan
        out.append(row)
    return pd.DataFrame(out).set_index("community")
AL = community_alliance(part["leiden"], "leiden")
# per-airport dominant alliance among alliance-operated routes touching it (None if no alliance route)
ends = pd.concat([R[["source", "alliance"]].rename(columns={"source": "node"}), R[["dest", "alliance"]].rename(columns={"dest": "node"})])
ends = ends[ends.alliance != "None"]
dom = ends.groupby("node").alliance.agg(lambda s: s.value_counts().idxmax()).reindex(T.index).fillna("None")
pa = part["leiden"]
align = dict(NMI_leiden_vs_airport_dominant_alliance=nmi(dom.loc[pa.index], pa),
             NMI_continent_vs_airport_dominant_alliance=nmi(dom.loc[lab], T.loc[lab, "continent"]),
             NMI_leiden_vs_continent=nmi(T.loc[lab, "continent"], pa.reindex(lab)))
print(align); pd.Series(align).to_csv(OUT / "alliance_nmi.csv", header=["value"])

# --- Leiden community profile table
prof = []
phi = conductance_per_community(u, labels_to_comms(pa.to_dict())); cl = list(labels_to_comms(pa.to_dict()))
for c, members in sorted(pa.groupby(pa).groups.items(), key=lambda x: -len(x[1])):
    m = list(members); g = T.loc[m]; top = g.sort_values("degree", ascending=False).head(5).index.tolist()
    cc = g.country.value_counts(); ct = g.continent.value_counts()
    prof.append(dict(community=c, size=len(m), top_country=cc.index[0] if len(cc) else None, country_purity=cc.iloc[0] / cc.sum() if len(cc) else np.nan,
        top_continent=ct.index[0] if len(ct) else None, continent_purity=ct.iloc[0] / ct.sum() if len(ct) else np.nan,
        conductance=conductance_per_community(u, [set(m)])[0], top_hubs=" ".join(top)))
prof = pd.DataFrame(prof).set_index("community").join(AL, how="left"); prof.to_csv(OUT / "leiden_community_profile.csv"); print(prof.head(15).round(2).to_string())

# --- plots
big = prof.index[:12]; cmap = plt.get_cmap("tab20")
fig, ax = plt.subplots(1, 3, figsize=(22, 6)); coord = T.dropna(subset=["lat", "lon"])
def draw(a, colors, title, legend=None):
    a.scatter(coord.lon, coord.lat, c="#cccccc", s=3); 
    for key, (idxs, col) in colors.items(): a.scatter(coord.loc[idxs, "lon"], coord.loc[idxs, "lat"], c=[col], s=7, label=key)
    a.set_title(title); a.set_xlabel("longitude"); a.set_ylabel("latitude"); a.legend(fontsize=6, markerscale=1.5, ncol=2, loc="lower left")
col = {f"L{c} {prof.loc[c,'top_hubs'].split()[0]} (n={prof.loc[c,'size']})": (coord.index.intersection(pa.index[pa == c]), cmap(i)) for i, c in enumerate(big)}
draw(ax[0], col, "Leiden communities (12 largest coloured, rest grey)")
g4 = cpm[cpm.k == K_MAP]; sizes = g4.groupby("community").size().sort_values(ascending=False).head(12)
col = {f"C{c} (n={s})": (coord.index.intersection(g4[g4.community == c].node), cmap(i)) for i, (c, s) in enumerate(sizes.items())}
draw(ax[1], col, f"Clique percolation k={K_MAP} (12 largest; overlap nodes drawn under last colour)")
cont = {c: (coord.index[coord.continent == c], cmap(i)) for i, c in enumerate(sorted(coord.continent.dropna().unique()))}
draw(ax[2], cont, "Reference: continent (hand-built mapping)")
plt.tight_layout(); plt.savefig(OUT / "map_communities.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
sz = {"Leiden": pa.value_counts().values, "Louvain": part["louvain"].value_counts().values}
for k in sorted(cpm.k.unique()): sz[f"CPM k={k}"] = cpm[cpm.k == k].groupby("community").size().values
for n, s in sz.items():
    s = np.sort(s); ax[0].loglog(s, 1 - np.arange(len(s)) / len(s), marker="o", ms=3, label=n)
ax[0].set_xlabel("community size"); ax[0].set_ylabel("CCDF"); ax[0].set_title("Community-size distributions"); ax[0].legend(fontsize=7)
de = pd.read_csv(OUT / "disjoint_eval.csv"); allm = pd.concat([de, ov]); lbl = [m.replace(" (best-Q of 10 seeds)", "") for m in allm.method]
ax[1].bar(lbl, allm.shen_EQ); ax[1].set_title("Modularity (disjoint) / Shen EQ (CPM; partial cover!)"); ax[1].tick_params(axis="x", rotation=60)
w = 0.4; x = np.arange(len(lbl)); ax[2].bar(x - w/2, allm.mean_conductance, w, label="mean"); ax[2].bar(x + w/2, allm.vol_weighted_conductance, w, label="volume-weighted")
ax[2].set_xticks(x); ax[2].set_xticklabels(lbl, rotation=60); ax[2].set_title("Conductance (lower = better separated)"); ax[2].legend()
plt.tight_layout(); plt.savefig(OUT / "community_eval.png", dpi=150); plt.close()

hm = prof.loc[big, [f"share_{a}" for a in list(ALLIANCES) + ["None"]]]; hm.index = [f"L{c} {prof.loc[c,'top_hubs'].split()[0]} ({prof.loc[c,'top_continent']})" for c in big]
fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(hm.values, cmap="viridis", aspect="auto"); ax.set_yticks(range(len(hm))); ax.set_yticklabels(hm.index, fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels(list(ALLIANCES) + ["None"], rotation=30)
for i in range(hm.shape[0]):
    for j in range(4): ax.text(j, i, f"{hm.values[i,j]:.2f}", ha="center", va="center", color="w", fontsize=7)
ax.set_title("Alliance share of intra-community route rows\n(global: " + ", ".join(f"{k} {v:.2f}" for k, v in base.items()) + ")", fontsize=8)
plt.colorbar(im); plt.tight_layout(); plt.savefig(OUT / "alliance_by_community.png", dpi=150); plt.close()

# --- continent composition of the 12 largest Leiden communities (explains low country purity of mixed communities)
comp = pd.crosstab(pa.reindex(T.index), T.continent).loc[big]; comp.insert(0, "size", prof.loc[big, "size"])
comp.to_csv(OUT / "leiden_continent_composition.csv"); print(comp.to_string())
