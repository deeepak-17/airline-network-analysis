"""Goal 2: growth-model comparison. Run: .venv/bin/python goal2/run_goal2.py"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, networkx as nx, powerlaw
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from common.load_graph import build_digraph, build_undirected

warnings.filterwarnings("ignore")
OUT = ROOT / "outputs" / "goal2"
SEEDS = list(range(10))

def lcc(g):
    return g.subgraph(max(nx.connected_components(g), key=len)).copy()

def metrics(g):
    """Undirected simple graph metrics; path length on largest component (exact BFS)."""
    deg = np.array([d for _, d in g.degree()])
    c = lcc(g)
    return dict(n=g.number_of_nodes(), m=g.number_of_edges(), mean_deg=deg.mean(), max_deg=deg.max(),
                transitivity=nx.transitivity(g), avg_clustering=nx.average_clustering(g),
                lcc_frac=c.number_of_nodes()/g.number_of_nodes(),
                avg_path=nx.average_shortest_path_length(c),
                assortativity=nx.degree_assortativity_coefficient(g)), deg

def ccdf(x):
    x = np.sort(x); y = 1 - np.arange(len(x))/len(x); return x, y

G = build_undirected(build_digraph())
G = nx.Graph(G); G.remove_edges_from(nx.selfloop_edges(G))
emp, edeg = metrics(G)
n, m = emp["n"], emp["m"]; kbar = 2*m/n
print("EMPIRICAL", emp)

# ---------------- model generation ----------------
p_er = 2*m/(n*(n-1))
m_ba = max(1, round(kbar/2)); k_ws = int(2*round(kbar/2))
print(f"ER p={p_er:.3e}; BA m={m_ba} (edges {m_ba*(n-m_ba)} vs {m}); WS k={k_ws}")

# WS: tune p over a grid, pick p whose mean clustering (3 seeds) is nearest empirical transitivity
# AND report path length at every p (clustering vs path-length tradeoff curve).
ws_rows = []
for p in [0.001,0.003,0.01,0.03,0.1,0.2,0.3,0.5,0.8,1.0]:
    r = []
    for s in SEEDS[:3]:
        g = nx.connected_watts_strogatz_graph(n, k_ws, p, tries=100, seed=s)
        r.append((nx.transitivity(g), nx.average_shortest_path_length(g)))
    r = np.mean(r, 0); ws_rows.append(dict(p=p, transitivity=r[0], avg_path=r[1]))
ws_sweep = pd.DataFrame(ws_rows); ws_sweep.to_csv(OUT/"ws_p_sweep.csv", index=False)
print(ws_sweep)
p_ws = float(ws_sweep.iloc[(ws_sweep.transitivity-emp["transitivity"]).abs().argmin()].p)
print("WS p chosen (matches transitivity):", p_ws)

gens = {
 "ER": lambda s: nx.gnp_random_graph(n, p_er, seed=s),
 "WS": lambda s: nx.watts_strogatz_graph(n, k_ws, p_ws, seed=s),
 "BA": lambda s: nx.barabasi_albert_graph(n, m_ba, seed=s),
}
rows, degs, ks_rows = [], {}, []
for name, f in gens.items():
    pooled = []
    for s in SEEDS:
        g = f(s); mt, d = metrics(g); mt.update(model=name, seed=s)
        mt["KS_deg"] = ks_2samp(edeg, d).statistic
        rows.append(mt); pooled.append(d)
    degs[name] = np.concatenate(pooled)
res = pd.DataFrame(rows); res.to_csv(OUT/"model_metrics_per_seed.csv", index=False)
cols = ["m","mean_deg","max_deg","transitivity","avg_clustering","lcc_frac","avg_path","assortativity","KS_deg"]
summ = res.groupby("model")[cols].agg(["mean","std"])
summ.columns = [f"{a}_{b}" for a,b in summ.columns]
emp_row = pd.DataFrame([{f"{k}_mean": emp.get(k, 0.0) for k in cols if k!="KS_deg"}], index=["EMPIRICAL"])
summ = pd.concat([emp_row, summ]); summ.to_csv(OUT/"model_comparison.csv")
print(summ.T.to_string())

# ---------------- power-law family fitting ----------------
fit = powerlaw.Fit(edeg, discrete=True, verbose=False)
fit_full = powerlaw.Fit(edeg, discrete=True, xmin=edeg.min(), verbose=False)
out = dict(xmin=fit.xmin, alpha=fit.power_law.alpha, KS_D=fit.power_law.D,
           n_tail=int((edeg>=fit.xmin).sum()), n=len(edeg))
print("PL fit", out)
cands = ["power_law","truncated_power_law","lognormal","exponential","stretched_exponential"]
cmp_rows = []
for i,a in enumerate(cands):
    for b in cands[i+1:]:
        R,p = fit.distribution_compare(a,b, normalized_ratio=True)
        cmp_rows.append(dict(A=a,B=b,R=R,p=p,favours=a if R>0 else b, significant=p<0.05))
cmp = pd.DataFrame(cmp_rows); cmp.to_csv(OUT/"distribution_comparisons.csv", index=False); print(cmp)
par = {"power_law_alpha":fit.power_law.alpha,
 "tpl_alpha":fit.truncated_power_law.parameter1,"tpl_lambda":fit.truncated_power_law.parameter2,
 "lognormal_mu":fit.lognormal.mu,"lognormal_sigma":fit.lognormal.sigma,
 "exp_lambda":fit.exponential.Lambda}
ksd = {d:getattr(fit,d).D for d in cands}
pd.DataFrame([{**out,**par,**{f"KS_{k}":v for k,v in ksd.items()}}]).to_csv(OUT/"powerlaw_fit.csv", index=False)
print(par, ksd)
# sensitivity: same comparison with fixed xmin values
sens=[]
for xm in [1,2,3,5,10,20]:
    f=powerlaw.Fit(edeg,discrete=True,xmin=xm,verbose=False)
    for b in ["truncated_power_law","lognormal","exponential"]:
        R,p=f.distribution_compare("power_law",b,normalized_ratio=True)
        sens.append(dict(xmin=xm,n_tail=int((edeg>=xm).sum()),alpha=f.power_law.alpha,vs=b,R_pl_minus=R,p=p))
pd.DataFrame(sens).to_csv(OUT/"xmin_sensitivity.csv",index=False); print(pd.DataFrame(sens))

# ---------------- figures ----------------
fig, ax = plt.subplots(1,2,figsize=(12,4.6))
x,y = ccdf(edeg); ax[0].loglog(x,y,'k.',ms=4,label="Airlines (empirical)")
for nm,c in zip(degs,["tab:blue","tab:green","tab:red"]):
    x2,y2=ccdf(degs[nm]); ax[0].loglog(x2,y2,color=c,lw=1.5,label=f"{nm} (10 seeds pooled)")
ax[0].set_xlabel("degree k"); ax[0].set_ylabel("P(K >= k)"); ax[0].set_ylim(1e-4,1.5)
ax[0].set_title("Degree CCDF: empirical vs generative models"); ax[0].legend()
fit.plot_ccdf(ax=ax[1],color='k',marker='.',linestyle='',label="Empirical")
fit.power_law.plot_ccdf(ax=ax[1],color='tab:red',label=f"power law (a={fit.power_law.alpha:.2f})")
fit.truncated_power_law.plot_ccdf(ax=ax[1],color='tab:purple',label="truncated power law")
fit.lognormal.plot_ccdf(ax=ax[1],color='tab:orange',label="lognormal")
fit.exponential.plot_ccdf(ax=ax[1],color='tab:green',ls='--',label="exponential")
ax[1].axvline(fit.xmin,color='grey',ls=':'); ax[1].set_title(f"Tail fits (xmin={int(fit.xmin)})")
ax[1].set_xlabel("degree k"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT/"degree_ccdf_models_and_fits.png",dpi=150); plt.close()

fig,ax=plt.subplots(1,2,figsize=(11,4.2))
b=np.arange(0,edeg.max()+2)
ax[0].hist(edeg,bins=np.unique(np.logspace(0,np.log10(edeg.max()+1),30).astype(int)),density=True,alpha=.6,color='k',label="Empirical")
for nm,c in zip(degs,["tab:blue","tab:green","tab:red"]):
    ax[0].hist(degs[nm],bins=np.unique(np.logspace(0,np.log10(max(degs[nm].max(),2)+1),30).astype(int)),density=True,histtype='step',color=c,lw=1.5,label=nm)
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel("degree"); ax[0].set_ylabel("density (log bins)"); ax[0].legend(); ax[0].set_title("Degree histogram (log bins)")
ax[1].plot(ws_sweep.avg_path,ws_sweep.transitivity,'o-',color='tab:green',label="WS sweep over p")
ax[1].scatter([emp["avg_path"]],[emp["transitivity"]],c='k',marker='*',s=200,label="Empirical",zorder=5)
for nm,c in zip(["ER","BA"],["tab:blue","tab:red"]):
    s=res[res.model==nm]; ax[1].errorbar(s.avg_path.mean(),s.transitivity.mean(),xerr=s.avg_path.std(),yerr=s.transitivity.std(),fmt='s',color=c,label=nm)
ax[1].set_yscale('log'); ax[1].set_xlabel("avg shortest path (LCC)"); ax[1].set_ylabel("transitivity"); ax[1].legend(); ax[1].set_title("Clustering vs path length")
plt.tight_layout(); plt.savefig(OUT/"hist_and_clustering_vs_path.png",dpi=150); plt.close()
print("done")
