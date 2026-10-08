"""Shared utilities for Goal 5 (link prediction) and Goal 6 (embeddings).

Design (all on the UNDIRECTED projection of the directed route graph; direction is lost):
  full graph G  --(hold out TEST_FRAC edges, spanning forest kept)-->  G_train + test_pos
  G_train       --(hold out VAL_FRAC edges, spanning forest kept)-->   G_inner + val_pos
Learned models are trained on (val_pos, val_neg) with features from G_inner, so training
pairs are never edges of the graph that produced their features (no self-leakage) and are
disjoint from test pairs. Test pairs are scored with features from G_train (or G_inner).
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.load_graph import build_undirected, load_airports  # noqa: E402

OUT = ROOT / "outputs" / "goal5_6"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_FRAC = 0.15   # fraction of ALL edges held out as test positives
VAL_FRAC = 0.15    # fraction of TRAIN edges held out as supervised-training positives
KS = (100, 1000)   # extra precision@k values (plus k = #positives)


# ----------------------------------------------------------------------------- graph / split
def load_graph():
    """Return (nodes, edges[m,2] int array with i<j, lat, lon, country, tz) for the undirected graph."""
    g = build_undirected()
    nodes = sorted(g.nodes)
    idx = {a: i for i, a in enumerate(nodes)}
    edges = np.array(sorted((min(idx[u], idx[v]), max(idx[u], idx[v])) for u, v in g.edges()))
    ap = load_airports().reindex(nodes)
    return nodes, edges, ap["lat"].to_numpy(float), ap["lon"].to_numpy(float), ap["country"], ap["tz"]


def _forest_mask(edges, n, order):
    """Kruskal over edges in `order`; True where the edge belongs to a random spanning forest."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mask = np.zeros(len(edges), bool)
    for e in order:
        a, b = find(edges[e, 0]), find(edges[e, 1])
        if a != b:
            parent[a] = b
            mask[e] = True
    return mask


def holdout(edges, n, frac, rng):
    """Hold out `frac` of edges, never removing an edge of a random spanning forest, so the
    remaining graph has exactly the same connected components (nothing becomes unreachable).
    Side effect: bridge/leaf edges are never test positives (no cold-start nodes in the test)."""
    order = rng.permutation(len(edges))
    forest = _forest_mask(edges, n, order)
    cand = [e for e in order if not forest[e]]
    k = int(round(frac * len(edges)))
    if k > len(cand):
        raise ValueError("not enough non-forest edges to hold out")
    held = np.zeros(len(edges), bool)
    held[cand[:k]] = True
    return edges[~held], edges[held]


def _keys(pairs, n):
    return set((pairs[:, 0].astype(np.int64) * n + pairs[:, 1]).tolist())


def sample_negatives(n, count, forbidden, rng, p=None):
    """`count` distinct unordered pairs (i<j) that are NOT in `forbidden` (set of i*n+j keys)."""
    out, seen = [], set()
    while len(out) < count:
        m = max(1000, (count - len(out)) * 3)
        a, b = rng.choice(n, m, p=p), rng.choice(n, m, p=p)
        for i, j in zip(a.tolist(), b.tolist()):
            if i == j:
                continue
            if i > j:
                i, j = j, i
            k = i * n + j
            if k in forbidden or k in seen:
                continue
            seen.add(k)
            out.append((i, j))
            if len(out) == count:
                break
    return np.array(out)


def make_split(edges, n, seed=SEED):
    rng = np.random.default_rng(seed)
    train, test_pos = holdout(edges, n, TEST_FRAC, rng)
    inner, val_pos = holdout(train, n, VAL_FRAC, rng)
    full = _keys(edges, n)
    deg_train = np.bincount(train.ravel(), minlength=n).astype(float)
    t_uni = sample_negatives(n, len(test_pos), full, rng)
    t_deg = sample_negatives(n, len(test_pos), full | _keys(t_uni, n), rng, p=deg_train / deg_train.sum())
    forb = full | _keys(t_uni, n) | _keys(t_deg, n)
    v_neg = sample_negatives(n, len(val_pos), forb, rng)
    naive_neg = sample_negatives(n, len(train), forb | _keys(v_neg, n), rng)
    s = dict(train=train, test_pos=test_pos, inner=inner, val_pos=val_pos,
             test_neg_uniform=t_uni, test_neg_degree=t_deg, val_neg=v_neg, naive_neg=naive_neg)
    check_split(s, edges, n)
    return s


def check_split(s, edges, n):
    """Sanity checks: partitions, disjointness, negatives are true non-edges."""
    full = _keys(edges, n)
    tr, te, inn, va = (_keys(s[k], n) for k in ("train", "test_pos", "inner", "val_pos"))
    assert not tr & te and tr | te == full, "train/test must partition edges"
    assert not inn & va and inn | va == tr, "inner/val must partition train"
    negs = {k: _keys(s[k], n) for k in ("test_neg_uniform", "test_neg_degree", "val_neg", "naive_neg")}
    for k, v in negs.items():
        assert not v & full, f"{k} contains a true edge"
        assert len(v) == len(s[k]), f"{k} has duplicates"
    names = list(negs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not negs[names[i]] & negs[names[j]], f"{names[i]} overlaps {names[j]}"
    assert nx.number_connected_components(_nx(s["train"], n)) == nx.number_connected_components(_nx(edges, n))
    assert nx.number_connected_components(_nx(s["inner"], n)) == nx.number_connected_components(_nx(edges, n))


def _nx(edges, n):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(map(tuple, edges.tolist()))
    return g


def save_split(s, nodes, tag=""):
    arr = np.array(nodes)
    for k, v in s.items():
        pd.DataFrame({"u": arr[v[:, 0]], "v": arr[v[:, 1]]}).to_csv(OUT / f"split_{k}{tag}.csv", index=False)


def load_split(nodes):
    """Reload the saved split (IATA csv -> indices) so Goal 6 uses the identical split."""
    idx = {a: i for i, a in enumerate(nodes)}
    s = {}
    for k in ("train", "test_pos", "inner", "val_pos", "test_neg_uniform", "test_neg_degree", "val_neg", "naive_neg"):
        d = pd.read_csv(OUT / f"split_{k}.csv", keep_default_na=False)
        a, b = d.u.map(idx).to_numpy(), d.v.map(idx).to_numpy()
        s[k] = np.column_stack([np.minimum(a, b), np.maximum(a, b)])
    return s


# ----------------------------------------------------------------------------- structural features
class FeatureGraph:
    """Adjacency + degree + PageRank of ONE graph; all pair features come from this graph only."""

    def __init__(self, edges, n):
        r = np.concatenate([edges[:, 0], edges[:, 1]])
        c = np.concatenate([edges[:, 1], edges[:, 0]])
        self.A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
        self.deg = np.asarray(self.A.sum(1)).ravel()
        with np.errstate(divide="ignore"):
            self.aa_w = np.where(self.deg > 1, 1 / np.log(np.maximum(self.deg, 2)), 0.0)
            self.ra_w = np.where(self.deg > 0, 1 / np.maximum(self.deg, 1), 0.0)
        pr = nx.pagerank(_nx(edges, n))
        self.pr = np.array([pr[i] for i in range(n)])

    def heuristics(self, pairs):
        I, J = pairs[:, 0], pairs[:, 1]
        M = self.A[I].multiply(self.A[J]).tocsr()
        cn = np.asarray(M.sum(1)).ravel()
        du, dv = self.deg[I], self.deg[J]
        return pd.DataFrame({
            "common_neighbours": cn,
            "jaccard": cn / np.maximum(du + dv - cn, 1),
            "adamic_adar": M @ self.aa_w,
            "resource_allocation": M @ self.ra_w,
            "pref_attachment": du * dv,
        })

    def features(self, pairs):
        """Heuristics + symmetric (min/max) degree and PageRank features for a supervised model."""
        f = self.heuristics(pairs)
        I, J = pairs[:, 0], pairs[:, 1]
        for c in ("common_neighbours", "adamic_adar", "resource_allocation", "pref_attachment"):
            f[c] = np.log1p(f[c])
        d = np.column_stack([self.deg[I], self.deg[J]])
        p = np.column_stack([self.pr[I], self.pr[J]])
        f["deg_min"], f["deg_max"] = np.log(d.min(1)), np.log(d.max(1))
        f["pr_min"], f["pr_max"] = np.log(p.min(1)), np.log(p.max(1))
        return f


def haversine_km(lat, lon, pairs):
    la1, lo1, la2, lo2 = (np.radians(x) for x in (lat[pairs[:, 0]], lon[pairs[:, 0]], lat[pairs[:, 1]], lon[pairs[:, 1]]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


# ----------------------------------------------------------------------------- evaluation
def evaluate(y, score, seed=SEED, threshold_pred=None):
    """AUC, AP and precision@k for k in (#positives, 100, 1000). Ties in score are broken
    randomly (fixed seed) so that discrete heuristics (e.g. common neighbours = 0) are not
    rewarded or penalised by input order."""
    y, score = np.asarray(y), np.asarray(score, float)
    npos = int(y.sum())
    tb = np.random.default_rng(seed).random(len(y))
    order = np.lexsort((tb, -score))
    res = {"auc": roc_auc_score(y, score), "ap": average_precision_score(y, score)}
    for k in (npos, *KS):
        k = min(k, len(y))
        res[f"p@{'npos' if k == npos else k}"] = y[order[:k]].mean()
    if threshold_pred is not None:
        tp = int(((threshold_pred == 1) & (y == 1)).sum())
        res["precision@0.5"] = tp / max(int((threshold_pred == 1).sum()), 1)
        res["recall@0.5"] = tp / max(npos, 1)
    return res


def test_sets(s):
    """{name: (pairs, y)} for the two negative-sampling schemes."""
    out = {}
    for name in ("uniform", "degree"):
        pairs = np.vstack([s["test_pos"], s[f"test_neg_{name}"]])
        out[name] = (pairs, np.r_[np.ones(len(s["test_pos"])), np.zeros(len(s[f"test_neg_{name}"]))])
    return out
