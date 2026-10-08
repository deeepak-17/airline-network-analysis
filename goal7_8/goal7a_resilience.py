"""GOAL 7a: structural node-removal resilience on the UNDIRECTED projection (full graph, 1% batches).
Metrics (all normalised by ORIGINAL N / intact value):
 - LCC fraction = |largest component of survivors| / N0
 - #components among survivors
 - global efficiency ESTIMATE: fixed random sample of S source nodes (same for all strategies);
   E ~ sum_{alive s} sum_t 1/d(s,t) / (S*(N0-1)); removed nodes contribute 0. Approximate (sampling), not exact.
"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _util import load, OUT
N_SRC, N_RANDOM, BATCH_FRAC, SEED = 300, 20, 0.01, 42
g, gd, gu, nodes = load(); N = gu.vcount()
print("N", N, "undirected edges", gu.ecount())
rng = np.random.default_rng(SEED)
src = rng.choice(N, N_SRC, replace=False)
batch = max(1, round(BATCH_FRAC * N))
steps = list(range(0, N, batch)) + [N]
pr = np.array(gd.pagerank(directed=True))
bet0 = np.array(gu.betweenness())
deg0 = np.array(gu.degree())

def metrics(alive):
    keep = np.flatnonzero(alive)
    if len(keep) == 0: return 0.0, 0, 0.0
    sub = gu.induced_subgraph(keep)
    comp = sub.connected_components()
    lcc = max(comp.sizes()) / N
    # map sample sources to subgraph indices
    pos = -np.ones(N, int); pos[keep] = np.arange(len(keep))
    s = pos[src]; s = s[s >= 0]
    if len(s) == 0: return lcc, len(comp), 0.0
    D = np.array(sub.distances(source=s.tolist()), float)
    with np.errstate(divide="ignore"):
        inv = np.where(D > 0, 1.0 / D, 0.0)  # inf->0, self(0)->0
    return lcc, len(comp), inv.sum() / (N_SRC * (N - 1))

def run(order_fn, label):
    """order_fn(alive) -> next node indices to remove (len = batch). Records metrics before each batch."""
    alive = np.ones(N, bool); rows = []
    for k, n_rem in enumerate(steps):
        lcc, nc, eff = metrics(alive)
        rows.append((n_rem / N, lcc, nc, eff))
        if n_rem >= N: break
        nxt = steps[k + 1] - n_rem
        alive[order_fn(alive, nxt)] = False
    return pd.DataFrame(rows, columns=["frac_removed", "lcc_frac", "n_components", "efficiency"]).assign(strategy=label)

def static_order(score):
    order = np.argsort(-score, kind="stable")
    ptr = {"i": 0}
    def f(alive, n):
        out = order[ptr["i"]:ptr["i"] + n]; ptr["i"] += n; return out
    return f

def adaptive_bet(alive, n):
    keep = np.flatnonzero(alive); sub = gu.induced_subgraph(keep)
    b = np.array(sub.betweenness()); return keep[np.argsort(-b, kind="stable")[:n]]

def adaptive_deg(alive, n):
    keep = np.flatnonzero(alive); sub = gu.induced_subgraph(keep)
    return keep[np.argsort(-np.array(sub.degree()), kind="stable")[:n]]

res = []
rand_runs = []
for sd in range(N_RANDOM):
    r = np.random.default_rng(1000 + sd); perm = r.permutation(N); ptr = {"i": 0}
    def f(alive, n, perm=perm, ptr=ptr):
        out = perm[ptr["i"]:ptr["i"] + n]; ptr["i"] += n; return out
    rand_runs.append(run(f, "random").assign(seed=sd)); print("random seed", sd, flush=True)
rand = pd.concat(rand_runs)
rand_mean = rand.groupby("frac_removed")[["lcc_frac", "n_components", "efficiency"]].mean().reset_index().assign(strategy="random")
rand_std = rand.groupby("frac_removed")[["lcc_frac", "efficiency"]].std().reset_index()
rand.to_csv(OUT / "resilience_random_all_seeds.csv", index=False)
strategies = {
    "betweenness (initial ranking)": static_order(bet0),
    "betweenness (adaptive)": adaptive_bet,
    "PageRank (initial ranking)": static_order(pr),
    "degree (initial ranking)": static_order(deg0),
    "degree (adaptive)": adaptive_deg,
}
allr = [rand_mean]
for lab, fn in strategies.items():
    allr.append(run(fn, lab)); print(lab, flush=True)
df = pd.concat(allr)
df["efficiency_rel"] = df["efficiency"] / df.loc[df.frac_removed == 0, "efficiency"].iloc[0]
df.to_csv(OUT / "resilience_curves.csv", index=False)

# fraction removed where LCC < 50% (of ORIGINAL N), linear interpolation between batch steps
def cross(d, col="lcc_frac", thr=0.5):
    d = d.sort_values("frac_removed"); x, y = d.frac_removed.values, d[col].values
    i = np.flatnonzero(y < thr)
    if len(i) == 0: return np.nan
    i = i[0]
    return x[i] if i == 0 else x[i-1] + (thr - y[i-1]) * (x[i] - x[i-1]) / (y[i] - y[i-1])
rows = []
for lab, d in df.groupby("strategy"):
    rows.append(dict(strategy=lab, frac_LCC_below_50pct=cross(d), frac_efficiency_below_50pct_of_intact=cross(d, "efficiency_rel")))
# random: also spread across seeds
rc = [cross(d) for _, d in rand.groupby("seed")]
rce = [cross(d.assign(efficiency_rel=d.efficiency / d.efficiency.iloc[0]), "efficiency_rel") for _, d in rand.groupby("seed")]
rows[[r["strategy"] for r in rows].index("random")].update(lcc50_seed_std=np.std(rc), eff50_seed_std=np.std(rce))
summ = pd.DataFrame(rows).sort_values("frac_LCC_below_50pct"); summ.to_csv(OUT / "resilience_summary.csv", index=False)
print(summ.to_string())
print("intact efficiency est:", df.loc[df.frac_removed == 0, "efficiency"].iloc[0], " intact LCC", df.loc[df.frac_removed == 0, "lcc_frac"].iloc[0])

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for lab, d in df.groupby("strategy"):
    ls = "-" if lab in ("random",) or "adaptive" in lab or "PageRank" in lab or "initial" in lab else "-"
    ax[0].plot(d.frac_removed, d.lcc_frac, label=lab); ax[1].plot(d.frac_removed, d.efficiency_rel, label=lab)
r = rand_mean.merge(rand_std, on="frac_removed", suffixes=("", "_sd"))
ax[0].fill_between(r.frac_removed, r.lcc_frac - r.lcc_frac_sd, r.lcc_frac + r.lcc_frac_sd, alpha=.2, color="C0")
ax[0].set_ylabel("LCC size / original N"); ax[1].set_ylabel("global efficiency (sampled) / intact")
for a in ax: a.set_xlabel("fraction of airports removed"); a.grid(alpha=.3)
ax[0].axhline(.5, color="k", ls=":", lw=1); ax[0].legend(fontsize=7)
fig.suptitle("Structural node-removal resilience (undirected projection; efficiency from 300 sampled sources)")
fig.tight_layout(); fig.savefig(OUT / "resilience_curves.png", dpi=150)
