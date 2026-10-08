"""GOAL 7b: Independent Cascade (IC) failure propagation. Re-run: .venv/bin/python goal7_8/goal7b_cascade.py
Model (structural, NOT an operational simulator):
 - A disrupted airport u gets ONE chance to disrupt each out-neighbour v (route u->v), success prob p_uv.
 - 'fixed':    p_uv = p
 - 'weighted': p_uv = 1-(1-p)^route_count(u,v)  (each airline route is an independent transmission channel)
 - No recovery inside the cascade: IC on a static graph == SIR with recovery after 1 step (same final size).
 - Directed graph used (disruption flows along route direction); no capacity, rerouting or time.
Seeds: top-K by degree (total), betweenness (undirected), PageRank, and K random airports (K=20).
Mean-field threshold reference: p_c ~ 1/lambda_max(A) (directed adjacency spectral radius), a heuristic
(ignores correlations / finite size); empirical 'threshold' = smallest p with mean cascade >= 5% of N.
"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, scipy.sparse as sp
from scipy.sparse.linalg import eigs
from multiprocessing import Pool
from _util import load, OUT
K, TRIALS, SEED = 20, 200, 7
PS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
g, gd, gu, nodes = load(); N = gd.vcount()
el = gd.get_edgelist()
A = sp.csr_matrix((np.ones(len(el)), ([e[0] for e in el], [e[1] for e in el])), shape=(N, N))
indptr, indices = A.indptr, A.indices
# route_count aligned with CSR order
E = pd.DataFrame(gd.get_edgelist(), columns=["u", "v"]); E["w"] = gd.es["w"]
M = sp.csr_matrix((E.w.values, (E.u.values, E.v.values)), shape=(N, N)); assert (M.indptr == A.indptr).all() and (M.indices == A.indices).all()
wts = M.data
lam = abs(eigs(A.astype(float), k=1, which="LR", return_eigenvectors=False)[0])
print("lambda_max(A) =", lam, " mean-field p_c ~", 1 / lam)

def cascade(seed, pe, rng):
    act = np.zeros(N, bool); act[seed] = True; frontier = [seed]; size = 1
    while frontier:
        nxt = []
        for u in frontier:
            s, e = indptr[u], indptr[u+1]
            if s == e: continue
            hit = rng.random(e - s) < pe[s:e]
            for v in indices[s:e][hit]:
                if not act[v]: act[v] = True; nxt.append(v)
        size += len(nxt); frontier = nxt
    return size

def work(args):
    p, model, name, seeds = args
    pe = np.full(len(wts), p) if model == "fixed" else 1 - (1 - p) ** wts
    rng = np.random.default_rng(SEED + int(p * 1e4))
    per_seed = [np.mean([cascade(s, pe, rng) for _ in range(TRIALS)]) for s in seeds]
    return dict(p=p, model=model, seed_set=name, mean_size=np.mean(per_seed), sd_across_seeds=np.std(per_seed),
                mean_frac=np.mean(per_seed) / N)

if __name__ == "__main__":
    pr = np.array(gd.pagerank()); bet = np.array(gu.betweenness()); deg = np.array(gd.degree())
    rng0 = np.random.default_rng(SEED)
    sets = {"top degree": np.argsort(-deg)[:K], "top betweenness": np.argsort(-bet)[:K],
            "top PageRank": np.argsort(-pr)[:K], "random": rng0.choice(N, K, replace=False)}
    pd.DataFrame({k: [nodes[i] for i in v] for k, v in sets.items()}).to_csv(OUT / "cascade_seed_sets.csv", index=False)
    jobs = [(p, m, n, s) for p in PS for m in ("fixed", "weighted") for n, s in sets.items()]
    with Pool() as pool: res = pool.map(work, jobs)
    df = pd.DataFrame(res); df.to_csv(OUT / "cascade_results.csv", index=False)
    print(df.pivot_table(index=["model", "p"], columns="seed_set", values="mean_frac").round(4).to_string())
    # empirical threshold-like point
    rows = []
    for (m, n), d in df.groupby(["model", "seed_set"]):
        d = d.sort_values("p"); hit = d[d.mean_frac >= 0.05]
        rows.append(dict(model=m, seed_set=n, p_at_5pct=hit.p.iloc[0] if len(hit) else np.nan))
    th = pd.DataFrame(rows); th["meanfield_pc"] = 1 / lam; th.to_csv(OUT / "cascade_thresholds.csv", index=False); print(th.to_string())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for a, m in zip(ax, ("fixed", "weighted")):
        for n, d in df[df.model == m].groupby("seed_set"):
            d = d.sort_values("p"); a.errorbar(d.p, d.mean_frac, d.sd_across_seeds / N, marker="o", label=n, capsize=2)
        a.axvline(1 / lam, color="k", ls=":", label="1/lambda_max"); a.set_xscale("log"); a.set_title(f"{m} p"); a.set_xlabel("p"); a.grid(alpha=.3)
    ax[0].set_ylabel("expected cascade size / N"); ax[0].legend(fontsize=8)
    fig.suptitle(f"Independent Cascade on directed route graph ({K} seeds/class, {TRIALS} trials/seed; bars = sd across seeds)")
    fig.tight_layout(); fig.savefig(OUT / "cascade_curves.png", dpi=150)
