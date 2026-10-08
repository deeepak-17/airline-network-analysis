"""GOAL 8b: degree-controlled Isolation Forest. Re-run (after goal8_anomaly.py): .venv/bin/python goal7_8/goal8b_degree_controlled.py
Motivation: plain IF on raw features mainly flags the largest hubs (extreme in every feature = 'rare', not 'odd combination').
Here each feature (log betweenness, log PageRank, clustering, log avg-neighbour-degree, core) is residualised on log10(degree)
by OLS (deg>=2 airports), so IF isolates unusual profiles GIVEN their degree. seed fixed."""
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from _util import OUT
df = pd.read_csv(OUT / "anomaly_all_airports.csv"); d = df[df.degree >= 2].copy()
x = np.log10(d.degree)
feats = {"lb": np.log10(d.betweenness + 1.0), "lp": np.log10(d.pagerank), "cl": d.clustering, "lnd": np.log10(d.avg_nbr_deg), "core": d.core}
R = {}
for k, v in feats.items():
    c = np.polyfit(x, v, 2); R[k] = v - np.polyval(c, x)   # quadratic in log-degree
Z = StandardScaler().fit_transform(pd.DataFrame(R))
iso = IsolationForest(n_estimators=500, contamination=0.02, random_state=42).fit(Z)
d["iso_deg_controlled"] = -iso.score_samples(Z)
for k in R: d["res_" + k] = R[k]
cols = ["iata", "name", "country", "degree", "betweenness", "pagerank", "clustering", "core", "avg_nbr_deg", "iso_deg_controlled"] + ["res_" + k for k in R]
top = d.nlargest(15, "iso_deg_controlled")[cols]; top.to_csv(OUT / "anomaly_top15_isoforest_degree_controlled.csv", index=False)
pd.set_option("display.width", 250); print(top.round(3).to_string(index=False))
