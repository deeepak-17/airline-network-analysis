"""GOAL 8: structural anomalies. Re-run: .venv/bin/python goal7_8/goal8_anomaly.py
Features on the UNDIRECTED projection (degree, exact betweenness, clustering, avg neighbour degree, core number)
plus directed PageRank. Two baselines:
 (1) OLS on log10(betweenness) ~ log10(degree), airports with betweenness>0; residual z-score; also betweenness/degree.
 (2) Isolation Forest and LOF on standardised log features (multivariate; contamination=0.02, seed fixed).
'Anomaly' = deviation from the fitted/empirical bulk, a flag for inspection, not a verdict.
"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from _util import load, OUT
SEED = 42
g, gd, gu, nodes = load()
ug = gu
deg = np.array(ug.degree()); bet = np.array(ug.betweenness())
clu = np.nan_to_num(np.array(ug.transitivity_local_undirected(mode="zero")))
core = np.array(ug.coreness()); pr = np.array(gd.pagerank())
nbr = np.array([np.mean([deg[j] for j in ug.neighbors(i)]) if deg[i] else 0 for i in range(len(nodes))])
df = pd.DataFrame(dict(iata=nodes, degree=deg, betweenness=bet, clustering=clu, pagerank=pr, core=core, avg_nbr_deg=nbr))
df["name"] = [g.nodes[n].get("name", "") for n in nodes]; df["country"] = [g.nodes[n].get("country", "") for n in nodes]
df = df[df.degree > 0].reset_index(drop=True)
# (1) regression residual
m = df.betweenness > 0
x, y = np.log10(df.degree[m]), np.log10(df.betweenness[m])
b1, b0 = np.polyfit(x, y, 1); res = y - (b0 + b1 * x)
df["bet_resid_z"] = np.nan; df.loc[m, "bet_resid_z"] = (res - res.mean()) / res.std()
df["bet_over_deg"] = df.betweenness / df.degree
print(f"log-log fit: log10(B) = {b0:.3f} + {b1:.3f} log10(k); R2={np.corrcoef(x,y)[0,1]**2:.3f}; n={m.sum()}; zero-betweenness airports={(~m).sum()}")
# (2) multivariate detectors
F = pd.DataFrame(dict(ld=np.log10(df.degree), lb=np.log10(df.betweenness + 1e-6 * df.betweenness.max()),
                      cl=df.clustering, lp=np.log10(df.pagerank), core=df.core, lnd=np.log10(df.avg_nbr_deg)))
Z = StandardScaler().fit_transform(F)
iso = IsolationForest(n_estimators=500, contamination=0.02, random_state=SEED).fit(Z)
df["iso_score"] = -iso.score_samples(Z)   # higher = more anomalous
lof = LocalOutlierFactor(n_neighbors=30, contamination=0.02).fit(Z)
df["lof_score"] = -lof.negative_outlier_factor_
df["iso_rank"] = df.iso_score.rank(ascending=False); df["lof_rank"] = df.lof_score.rank(ascending=False)
df["resid_rank"] = df.bet_resid_z.rank(ascending=False)
print("Spearman iso vs lof:", spearmanr(df.iso_score, df.lof_score)[0], " iso vs resid:", spearmanr(df.iso_score[m], df.bet_resid_z[m])[0])
top = 15; cols = ["iata", "name", "country", "degree", "betweenness", "pagerank", "clustering", "core", "avg_nbr_deg"]
iso_top = df.nlargest(top, "iso_score")[cols + ["iso_score", "lof_rank", "bet_resid_z"]]
res_top = df[df.degree >= 5].nlargest(top, "bet_resid_z")[cols + ["bet_resid_z", "bet_over_deg", "iso_rank"]]
bd_top = df.nlargest(top, "bet_over_deg")[cols + ["bet_over_deg", "bet_resid_z"]]
top_iso_set = set(df.nsmallest(top, "iso_rank").iata); top_res_set = set(res_top.iata); top_lof = set(df.nsmallest(top, "lof_rank").iata)
print("overlap top15 iso&resid:", len(top_iso_set & top_res_set), " iso&lof:", len(top_iso_set & top_lof))
df.to_csv(OUT / "anomaly_all_airports.csv", index=False)
iso_top.to_csv(OUT / "anomaly_top15_isolation_forest.csv", index=False)
res_top.to_csv(OUT / "anomaly_top15_betweenness_residual_deg_ge5.csv", index=False)
bd_top.to_csv(OUT / "anomaly_top15_betweenness_over_degree.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 30)
for t, d in [("ISOLATION FOREST", iso_top), ("RESIDUAL (deg>=5)", res_top), ("BET/DEG", bd_top)]:
    print("\n==", t); print(d.round(4).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
mm = df.betweenness > 0
ax[0].scatter(df.degree[mm], df.betweenness[mm], s=6, c="lightgray")
xs = np.linspace(x.min(), x.max(), 50); ax[0].plot(10**xs, 10**(b0 + b1 * xs), "r-", lw=1, label="log-log OLS")
for _, r in res_top.head(10).iterrows():
    ax[0].scatter(r.degree, r.betweenness, c="C1", s=18); ax[0].annotate(r.iata, (r.degree, r.betweenness), fontsize=8)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("degree (undirected)"); ax[0].set_ylabel("betweenness"); ax[0].legend()
ax[0].set_title("Betweenness vs degree; top residual airports (deg>=5)")
sc = ax[1].scatter(np.log10(df.degree), np.log10(df.pagerank), c=df.iso_score, s=8, cmap="viridis")
for _, r in iso_top.head(12).iterrows():
    ax[1].annotate(r.iata, (np.log10(r.degree), np.log10(r.pagerank)), fontsize=8, color="r")
plt.colorbar(sc, label="Isolation Forest score"); ax[1].set_xlabel("log10 degree"); ax[1].set_ylabel("log10 PageRank")
ax[1].set_title("Isolation Forest (6 structural features)")
fig.tight_layout(); fig.savefig(OUT / "anomaly_plots.png", dpi=150)
