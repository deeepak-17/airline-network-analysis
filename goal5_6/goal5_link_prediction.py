"""Goal 5 - link prediction on the undirected projection.  Run: .venv/bin/python goal5_6/goal5_link_prediction.py

1. Build/save the canonical split (seed 42) to outputs/goal5_6/split_*.csv (reused by Goal 6).
2. Unsupervised heuristics (features from G_train): CN, Jaccard, Adamic-Adar, RA, PA (+ geographic-distance
   baseline that does not use the graph at all).
3. Supervised LR / RF / HistGB on structural features. Trained on val pairs with features from G_inner;
   tested on test pairs with features from G_train ("shifted": denser graph than at training time) and from
   G_inner ("matched").
4. Robustness: whole pipeline repeated over 5 independent splits (seeds 42..46).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lp_common import (OUT, SEED, FeatureGraph, evaluate, haversine_km, load_graph, make_split,  # noqa: E402
                       save_split, test_sets)

HEURISTICS = ["common_neighbours", "jaccard", "adamic_adar", "resource_allocation", "pref_attachment"]
N_SEEDS = 5


def models(seed):
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)),
        "RandomForest": RandomForestClassifier(300, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "HistGradBoost": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed),
    }


def run_split(s, n, lat, lon, seed):
    """Return (rows, scores) where rows is a list of result dicts and scores {(method, neg): (y, score)}."""
    fg_train, fg_inner = FeatureGraph(s["train"], n), FeatureGraph(s["inner"], n)
    pairs_val = np.vstack([s["val_pos"], s["val_neg"]])
    y_val = np.r_[np.ones(len(s["val_pos"])), np.zeros(len(s["val_neg"]))]
    clfs = {}
    for name, m in models(seed).items():
        clfs[name] = m.fit(fg_inner.features(pairs_val), y_val)  # features from G_inner only

    rows, scores = [], {}

    def add(method, kind, regime, neg, y, sc, pred=None):
        rows.append(dict(method=method, kind=kind, regime=regime, negatives=neg, seed=seed, **evaluate(y, sc, seed, pred)))
        scores[(method if kind != 'supervised ML' else f'{method} [{regime}]', neg)] = (y, sc)

    for neg, (pairs, y) in test_sets(s).items():
        h = fg_train.heuristics(pairs)
        for c in HEURISTICS:
            add(c, "heuristic", "G_train", neg, y, h[c].to_numpy())
        d = haversine_km(lat, lon, pairs)
        add("geo_distance(-km)", "non-graph baseline", "lat/lon", neg, y, -np.nan_to_num(d, nan=np.nanmedian(d)))
        for regime, fg in (("shifted: feats G_train", fg_train), ("matched: feats G_inner", fg_inner)):
            X = fg.features(pairs)
            for name, clf in clfs.items():
                p = clf.predict_proba(X)[:, 1]
                add(f"{name}(structural)", "supervised ML", regime, neg, y, p, (p >= 0.5).astype(int))
    return rows, scores


def plot_roc(scores, s, fname):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, neg in zip(ax, ("uniform", "degree")):
        for (m, ng), (y, sc) in scores.items():
            if ng != neg or ("(structural)" in m and not m.startswith("LogReg(structural) [shifted")):
                continue
            fpr, tpr, _ = roc_curve(y, sc)
            a.plot(fpr, tpr, label=m, lw=1.4)
        a.plot([0, 1], [0, 1], "k:", lw=0.8)
        a.set(title=f"Test ROC, {neg} negatives", xlabel="FPR", ylabel="TPR")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def main():
    nodes, edges, lat, lon, _, _ = load_graph()
    n = len(nodes)
    all_rows, canonical_scores = [], None
    for k in range(N_SEEDS):
        seed = SEED + k
        s = make_split(edges, n, seed)
        if k == 0:
            save_split(s, nodes)
            print({key: len(v) for key, v in s.items()}, "| nodes", n, "| edges", len(edges))
        rows, scores = run_split(s, n, lat, lon, seed)
        all_rows += rows
        if k == 0:
            canonical_scores = scores
            plot_roc(scores, s, "goal5_roc.png")
            pd.DataFrame({f"{m}|{ng}": pd.Series(sc) for (m, ng), (_, sc) in scores.items()}).to_csv(
                OUT / "goal5_canonical_test_scores.csv", index=False)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "goal5_results_all_seeds.csv", index=False)
    df[df.seed == SEED].to_csv(OUT / "goal5_results_canonical.csv", index=False)
    metrics = [c for c in df.columns if c in ("auc", "ap", "p@npos", "p@100", "p@1000", "precision@0.5", "recall@0.5")]
    g = df.groupby(["negatives", "kind", "regime", "method"])[metrics]
    summ = g.mean().round(4).join(g.std().round(4), rsuffix="_sd").reset_index()
    summ.to_csv(OUT / "goal5_results_mean_sd_over_5_splits.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(summ[["negatives", "regime", "method", "auc", "auc_sd", "ap", "p@npos", "p@npos_sd", "p@100", "p@1000"]].to_string(index=False))


if __name__ == "__main__":
    main()
