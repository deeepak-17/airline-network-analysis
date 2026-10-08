"""Goal 6, step 3 - embedding-based link prediction on the SAME split as Goal 5.
Run: .venv/bin/python goal5_6/goal6_link_prediction.py   (needs goal5 + goal6_embeddings first)

Regimes (all tested on split_test_pos vs split_test_neg_*; test edges are in NO embedding graph):
  cosine@G_train        unsupervised: cosine(u,v) of embeddings learned on G_train (no training pairs)
  cosine@G_inner        same with the G_inner embeddings
  clf@G_inner  (clean)  edge-feature classifier trained on val_pos/val_neg (never seen by G_inner walks),
                        features and test scoring from the same G_inner embedding. No leakage, no shift.
  clf@G_train naive     classic protocol: embed G_train, train on ALL G_train edges (positives, which the
                        walks have seen) vs naive_neg. Optimistic bias: positives have been 'memorised'
                        during embedding, so the classifier learns 'seen edge' rather than 'plausible edge'.
  hybrid@G_inner        embedding Hadamard + the Goal-5 structural features (from G_inner) in one classifier.
Edge operators for the undirected pair (symmetric on purpose): Hadamard u*v, L1 |u-v|, average (u+v)/2.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goal6_embeddings import EMB_SEEDS, cosine, load_emb  # noqa: E402
from lp_common import OUT, FeatureGraph, evaluate, load_graph, load_split, test_sets  # noqa: E402

OPS = {"hadamard": lambda a, b: a * b, "l1": lambda a, b: np.abs(a - b), "average": lambda a, b: (a + b) / 2}


def edge_feats(E, pairs, op):
    return OPS[op](E[pairs[:, 0]], E[pairs[:, 1]])


def lr(seed):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed))


def rf(seed):
    return RandomForestClassifier(300, min_samples_leaf=3, n_jobs=-1, random_state=seed)


def run(method, seed, s, n, fg_inner, tsets, rows, scores):
    E_in, E_tr = load_emb("inner", method, seed), load_emb("train", method, seed)
    tag = "node2vec-SGNS" if method == "sgns" else "PPMI-SVD(ablation)"
    pv = np.vstack([s["val_pos"], s["val_neg"]])
    yv = np.r_[np.ones(len(s["val_pos"])), np.zeros(len(s["val_neg"]))]
    pn = np.vstack([s["train"], s["naive_neg"]])
    yn = np.r_[np.ones(len(s["train"])), np.zeros(len(s["naive_neg"]))]
    fitted = {}
    for op in OPS:
        fitted[("clean", op, "LR")] = lr(seed).fit(edge_feats(E_in, pv, op), yv)
        fitted[("naive", op, "LR")] = lr(seed).fit(edge_feats(E_tr, pn, op), yn)
    fitted[("clean", "hadamard", "RF")] = rf(seed).fit(edge_feats(E_in, pv, "hadamard"), yv)
    fitted[("naive", "hadamard", "RF")] = rf(seed).fit(edge_feats(E_tr, pn, "hadamard"), yn)
    hyb_tr = lambda E, P: np.hstack([edge_feats(E, P, "hadamard"), fg_inner.features(P).to_numpy()])  # noqa: E731
    hyb = lr(seed).fit(hyb_tr(E_in, pv), yv)

    def add(name, kind, regime, neg, y, sc, pred=None):
        rows.append(dict(method=f"{tag}: {name}", kind=kind, regime=regime, negatives=neg, seed=seed, **evaluate(y, sc, seed, pred)))
        scores[(seed, f"{tag}: {name}", neg)] = (y, sc)

    for neg, (P, y) in tsets.items():
        add("cosine", "embedding (unsupervised)", "cosine@G_train", neg, y, cosine(E_tr, P))
        add("cosine", "embedding (unsupervised)", "cosine@G_inner", neg, y, cosine(E_in, P))
        for (reg, op, clf), m in fitted.items():
            E = E_in if reg == "clean" else E_tr
            pr = m.predict_proba(edge_feats(E, P, op))[:, 1]
            r = "clf@G_inner clean" if reg == "clean" else "clf@G_train naive (seen-edge bias)"
            add(f"{clf} {op}", "embedding + ML", r, neg, y, pr, (pr >= 0.5).astype(int))
        pr = hyb.predict_proba(hyb_tr(E_in, P))[:, 1]
        add("LR hadamard+structural", "embedding + structural ML", "hybrid@G_inner", neg, y, pr, (pr >= 0.5).astype(int))


def main():
    nodes, edges, *_ = load_graph()
    n = len(nodes)
    s = load_split(nodes)
    fg_inner = FeatureGraph(s["inner"], n)
    tsets = test_sets(s)
    rows, scores = [], {}
    for seed in EMB_SEEDS:
        for method in ("sgns", "ppmi_svd"):
            run(method, seed, s, n, fg_inner, tsets, rows, scores)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "goal6_lp_all_seeds.csv", index=False)
    metrics = ["auc", "ap", "p@npos", "p@100", "p@1000", "precision@0.5", "recall@0.5"]
    g = df.groupby(["negatives", "kind", "regime", "method"])[metrics]
    emb = g.mean().round(4).join(g.std().round(4), rsuffix="_sd").reset_index()
    emb["n_runs"] = len(EMB_SEEDS)
    emb.to_csv(OUT / "goal6_lp_mean_sd_over_embedding_seeds.csv", index=False)

    # ---- single comparison table: Goal 5 canonical split (same split file) + Goal 6 embeddings
    g5 = pd.read_csv(OUT / "goal5_results_canonical.csv")
    g5 = g5[["negatives", "kind", "regime", "method"] + [m for m in metrics if m in g5.columns]]
    g5["n_runs"] = 1
    both = pd.concat([g5, emb[g5.columns.tolist()]], ignore_index=True)
    both["goal"] = np.where(both.kind.str.contains("embedding"), "6", "5")
    both = both.sort_values(["negatives", "auc"], ascending=[True, False])
    both.to_csv(OUT / "goal5_vs_goal6_comparison_table.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 48, "display.max_rows", 300)
    for neg in ("uniform", "degree"):
        print(f"\n=== negatives: {neg} (Goal-5 rows: split seed 42; Goal-6 rows: mean over {len(EMB_SEEDS)} embedding seeds) ===")
        print(both[both.negatives == neg][["goal", "method", "regime", "auc", "ap", "p@npos", "p@100", "p@1000"]].to_string(index=False))

    plot_roc(scores, df)


def plot_roc(scores, df):
    keys = {"node2vec-SGNS: cosine": "cosine G_train", "node2vec-SGNS: LR hadamard": "LR hadamard (clean)",
            "node2vec-SGNS: LR l1": "LR L1 (clean)",
            "node2vec-SGNS: LR hadamard+structural": "hybrid (emb+structural)"}
    g5 = pd.read_csv(OUT / "goal5_canonical_test_scores.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, neg in zip(ax, ("uniform", "degree")):
        y = test_sets(load_split(load_graph()[0]))[neg][1]
        for col, lab in (("adamic_adar", "Adamic-Adar"), ("jaccard", "Jaccard"), ("pref_attachment", "Pref. attachment"),
                         ("LogReg(structural) [shifted: feats G_train]", "LogReg structural (G_train feats)")):
            c = f"{col}|{neg}"
            if c in g5:
                fpr, tpr, _ = roc_curve(y, g5[c].dropna().to_numpy())
                a.plot(fpr, tpr, lw=1.3, label=lab)
        for (seed, m, ng), (yy, sc) in scores.items():
            if seed != 0 or ng != neg:
                continue
            for k, lab in keys.items():
                if m == k:
                    fpr, tpr, _ = roc_curve(yy, sc)
                    a.plot(fpr, tpr, lw=1.3, ls="--", label=lab)
        a.plot([0, 1], [0, 1], "k:", lw=0.8)
        a.set(title=f"Test ROC, {neg} negatives", xlabel="FPR", ylabel="TPR")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "goal6_lp_roc.png", dpi=150)


if __name__ == "__main__":
    main()
