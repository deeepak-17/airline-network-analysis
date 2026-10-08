"""Goal 6, step 1 - learn node2vec embeddings on the TRAINING graphs only.
Run: .venv/bin/python goal5_6/goal6_embeddings.py        (about 20-25 min on a laptop CPU; Goal 5 must run first)

Graphs embedded (never the full graph, so held-out test edges cannot influence any vector):
  G_train : 85% of edges  (used for node classification and the 'unsupervised cosine' link score)
  G_inner : G_train minus the supervised-training positives (used where a classifier is trained on val pairs)
(p, q) are chosen on the VALIDATION pairs only: cosine-AUC of val_pos vs val_neg on G_inner embeddings.
Outputs: emb_<graph>_<method>_seed<k>.npy, goal6_pq_grid.csv, goal6_best_pq.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lp_common import OUT, load_graph, load_split  # noqa: E402
from node2vec_np import biased_walks, ppmi_svd, sgns  # noqa: E402

DIM, NUM_WALKS, WALK_LEN, WINDOW, NEG = 64, 10, 40, 5, 5
EPOCHS, LR0 = 3, 0.1
EMB_SEEDS = (0, 1, 2)
PQ_GRID = [(1, 1), (0.5, 2), (2, 0.5), (1, 0.5), (1, 2)]   # (p, q); (1,1) = DeepWalk
TUNE_SEED = 100


def embed(edges, n, method, p, q, seed):
    w = biased_walks(edges, n, NUM_WALKS, WALK_LEN, p, q, seed)
    if method == "sgns":
        return sgns(w, n, DIM, WINDOW, NEG, EPOCHS, LR0, seed=seed)
    return ppmi_svd(w, n, DIM, WINDOW, seed=seed)


def cosine(E, pairs):
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return (E[pairs[:, 0]] * E[pairs[:, 1]]).sum(1)


def emb_path(graph, method, seed):
    return OUT / f"emb_{graph}_{method}_seed{seed}.npy"


def load_emb(graph, method, seed):
    return np.load(emb_path(graph, method, seed))


def main():
    nodes, _, *_ = load_graph()
    n = len(nodes)
    s = load_split(nodes)
    pairs = np.vstack([s["val_pos"], s["val_neg"]])
    y = np.r_[np.ones(len(s["val_pos"])), np.zeros(len(s["val_neg"]))]

    grid = []
    for p, q in PQ_GRID:
        auc = roc_auc_score(y, cosine(embed(s["inner"], n, "sgns", p, q, TUNE_SEED), pairs))
        grid.append(dict(p=p, q=q, val_cosine_auc=auc))
        print(f"p={p} q={q} val cosine AUC={auc:.4f}", flush=True)
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "goal6_pq_grid.csv", index=False)
    best = grid.sort_values("val_cosine_auc", ascending=False).iloc[0]
    p, q = float(best.p), float(best.q)
    json.dump(dict(p=p, q=q, dim=DIM, num_walks=NUM_WALKS, walk_len=WALK_LEN, window=WINDOW, neg=NEG,
                   epochs=EPOCHS, lr0=LR0, tune_seed=TUNE_SEED), open(OUT / "goal6_best_pq.json", "w"), indent=1)
    print("selected (p,q) =", (p, q), flush=True)

    for seed in EMB_SEEDS:
        for graph in ("inner", "train"):
            for method in ("sgns", "ppmi_svd"):
                np.save(emb_path(graph, method, seed), embed(s[graph], n, method, p, q, seed))
            print("saved embeddings", graph, "seed", seed, flush=True)


if __name__ == "__main__":
    main()
