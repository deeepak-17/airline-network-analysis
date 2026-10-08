"""Goal 6, step 2 - downstream node classification from embeddings learned on G_train only.
Run: .venv/bin/python goal5_6/goal6_node_tasks.py   (needs goal6_embeddings first)

Task A  continent. Label = region prefix of the airports.dat tz database name (Europe/, Asia/, Africa/,
        Australia/ + Pacific/ -> Oceania, America/ -> North or South America by a country list; Pacific/ in the USA
        -> North America). Atlantic/, Indian/, Arctic/, Antarctica/ and nodes absent from airports.dat are DROPPED
        (ambiguous or unlabelled). Labels come from metadata, NOT from the graph, so there is no label leakage;
        embeddings are unsupervised, so CV test nodes' labels never touch them (transductive but label-free).
Task B  hub status = top 5% of nodes by degree. Label is DEGREE-DERIVED, hence two variants:
        B1 label from FULL-graph degree (includes the held-out edges; features from G_train)
        B2 label from G_train degree (the very graph that was embedded) -> near-trivial; shown only to
           quantify the leakage the task description warns about.
Baselines: majority class; degree-only model (log G_train degree) - the honest competitor for hub status.
Evaluation: stratified 5-fold CV (fixed seed), out-of-fold predictions -> accuracy, precision, recall, F1
(macro for continent, positive-class for hub), confusion matrix; mean +- sd of macro/positive F1 across folds.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goal6_embeddings import EMB_SEEDS, load_emb  # noqa: E402
from lp_common import OUT, SEED, load_graph, load_split  # noqa: E402

HUB_FRAC, MIN_CLASS = 0.05, 30
SOUTH_AMERICA = {"Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "French Guiana", "Guyana",
                 "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela", "Falkland Islands"}


def continent(country, tz):
    if not isinstance(tz, str):
        return None
    region = tz.split("/")[0]
    if region == "America":
        return "South America" if country in SOUTH_AMERICA else "North America"
    if region in ("Europe", "Asia", "Africa"):
        return region
    if region in ("Australia", "Pacific"):
        return "North America" if country == "United States" else "Oceania"
    return None


def classifiers(seed):
    return {"LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed)),
            "RandomForest": RandomForestClassifier(300, min_samples_leaf=2, n_jobs=-1, random_state=seed)}


def cv_eval(X, y, model, seed, binary_pos=None):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pred = cross_val_predict(model, X, y, cv=skf)
    fold_f1 = []
    for _, te in skf.split(X, y):
        fold_f1.append(f1_score(y[te], pred[te], average="macro" if binary_pos is None else "binary",
                                **({} if binary_pos is None else {"pos_label": binary_pos}), zero_division=0))
    if binary_pos is None:
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    else:
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=binary_pos, zero_division=0)
    return dict(accuracy=accuracy_score(y, pred), precision=p, recall=r, f1=f, f1_fold_mean=np.mean(fold_f1),
                f1_fold_sd=np.std(fold_f1)), pred


def run_task(task, X_emb, X_deg, y, seed, rows, cms, binary_pos=None):
    cands = {"majority": (DummyClassifier(strategy="most_frequent"), X_deg),
             "degree-only LogReg": (classifiers(seed)["LogReg"], X_deg),
             "degree-only RF": (classifiers(seed)["RandomForest"], X_deg)}
    for name, m in classifiers(seed).items():
        cands[f"node2vec {name}"] = (m, X_emb)
    for name, (m, X) in cands.items():
        res, pred = cv_eval(X, y, m, seed, binary_pos)
        rows.append(dict(task=task, model=name, emb_seed=seed, n=len(y), **res))
        if seed == EMB_SEEDS[0] and name.startswith(("node2vec", "majority")):
            cms[(task, name)] = (np.unique(y), confusion_matrix(y, pred, labels=np.unique(y)))


def plot_projection(E, labels, mask, fname):
    Z_pca = PCA(2, random_state=SEED).fit_transform(E)
    Z_tsne = TSNE(2, perplexity=30, init="pca", random_state=SEED).fit_transform(E)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for a, Z, t in zip(ax, (Z_pca, Z_tsne), ("PCA", "t-SNE (perplexity 30)")):
        a.scatter(Z[~mask, 0], Z[~mask, 1], s=4, c="lightgrey", label="unlabelled / dropped")
        for c in sorted(set(labels[mask])):
            m = mask & (labels == c)
            a.scatter(Z[m, 0], Z[m, 1], s=6, label=f"{c} ({m.sum()})")
        a.set_title(f"{t} of node2vec embeddings learned on G_train")
        a.set_xticks([]); a.set_yticks([])
    ax[1].legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def plot_cms(cms, fname):
    items = list(cms.items())
    fig, axes = plt.subplots(1, len(items), figsize=(4.2 * len(items), 4))
    for a, ((task, name), (labs, cm)) in zip(np.atleast_1d(axes), items):
        a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        a.set_xticks(range(len(labs))); a.set_yticks(range(len(labs)))
        a.set_xticklabels(labs, rotation=60, ha="right", fontsize=7); a.set_yticklabels(labs, fontsize=7)
        for i in range(len(labs)):
            for j in range(len(labs)):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="k")
        a.set_title(f"{task}\n{name}", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def main():
    nodes, edges, _, _, country, tz = load_graph()
    n = len(nodes)
    s = load_split(nodes)
    deg_train = np.bincount(s["train"].ravel(), minlength=n)
    deg_full = np.bincount(edges.ravel(), minlength=n)
    cont = np.array([continent(c, t) for c, t in zip(country, tz)], dtype=object)
    counts = pd.Series([c for c in cont if c is not None]).value_counts()
    keep_cls = set(counts[counts >= MIN_CLASS].index)
    mask = np.array([c in keep_cls for c in cont])
    print("continent labels (after dropping unlabelled/ambiguous):", counts.to_dict(), "| labelled nodes", mask.sum(), "of", n)
    hub_full = deg_full >= np.quantile(deg_full, 1 - HUB_FRAC)
    hub_train = deg_train >= np.quantile(deg_train, 1 - HUB_FRAC)
    print("hub threshold (full degree >=)", np.quantile(deg_full, 1 - HUB_FRAC), "hubs:", hub_full.sum(),
          "| overlap of full- and train-degree hub sets:", (hub_full & hub_train).sum())
    Xdeg = np.log(deg_train)[:, None]

    rows, cms = [], {}
    for seed in EMB_SEEDS:
        E = load_emb("train", "sgns", seed)
        run_task("A continent", E[mask], Xdeg[mask], cont[mask].astype(str), seed, rows, cms)
        run_task("B1 hub (label: full-graph degree)", E, Xdeg, hub_full.astype(int), seed, rows, cms, binary_pos=1)
        run_task("B2 hub (label: G_train degree; leaky)", E, Xdeg, hub_train.astype(int), seed, rows, cms, binary_pos=1)
        E2 = load_emb("train", "ppmi_svd", seed)
        for task, X, y, bp in (("A continent", E2[mask], cont[mask].astype(str), None),
                               ("B1 hub (label: full-graph degree)", E2, hub_full.astype(int), 1)):
            for name, m in classifiers(seed).items():
                res, _ = cv_eval(X, y, m, seed, bp)
                rows.append(dict(task=task, model=f"PPMI-SVD {name} (ablation)", emb_seed=seed, n=len(y), **res))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "goal6_node_tasks_all_seeds.csv", index=False)
    g = df.groupby(["task", "model"])[["accuracy", "precision", "recall", "f1", "f1_fold_mean", "f1_fold_sd"]]
    summ = g.mean().round(4).join(g.std().round(4)[["f1"]].rename(columns={"f1": "f1_sd_over_emb_seeds"})).reset_index()
    summ.to_csv(OUT / "goal6_node_tasks_summary.csv", index=False)
    pd.set_option("display.width", 220, "display.max_colwidth", 45)
    print(summ.to_string(index=False))

    pd.DataFrame([dict(task=t, model=m, classes="|".join(map(str, labs)), cm=cm.tolist()) for (t, m), (labs, cm) in cms.items()]
                 ).to_csv(OUT / "goal6_confusion_matrices.csv", index=False)
    plot_cms({k: v for k, v in cms.items() if k[1] == "node2vec LogReg" and not k[0].startswith("B2")}, "goal6_confusion_matrices.png")
    plot_projection(load_emb("train", "sgns", EMB_SEEDS[0]), cont.astype(object), mask, "goal6_embedding_projection_continent.png")


if __name__ == "__main__":
    main()
