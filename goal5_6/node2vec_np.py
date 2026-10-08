"""From-scratch node2vec in numpy/scipy (gensim is unavailable on Python 3.14).

  biased_walks : 2nd-order biased random walks (Grover & Leskovec 2016), exact via rejection sampling,
                 vectorised over all walkers.  p = return parameter, q = in-out parameter.
  sgns         : skip-gram with negative sampling trained by minibatch SGD (the algorithm used by
                 word2vec/gensim, re-implemented; noise distribution = unigram^0.75).
  ppmi_svd     : documented alternative (Levy & Goldberg 2014): SGNS implicitly factorises a shifted
                 PMI matrix; here the walk co-occurrence PPMI matrix is factorised by truncated SVD.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds


def csr_adj(edges, n):
    r = np.concatenate([edges[:, 0], edges[:, 1]])
    c = np.concatenate([edges[:, 1], edges[:, 0]])
    return sp.csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))


def biased_walks(edges, n, num_walks, walk_len, p, q, seed):
    """Return int32 array [num_walks*n, walk_len]. Every node must have degree >= 1."""
    rng = np.random.default_rng(seed)
    A = csr_adj(edges, n)
    A.sort_indices()
    indptr, indices = A.indptr.astype(np.int64), A.indices.astype(np.int64)
    deg = np.diff(indptr)
    assert deg.min() >= 1, "isolated node in walk graph"
    keys = np.repeat(np.arange(n, dtype=np.int64), deg) * n + indices  # sorted (row-major, sorted cols)
    W = n * num_walks
    walks = np.empty((W, walk_len), dtype=np.int64)
    walks[:, 0] = np.tile(np.arange(n), num_walks)
    cur = walks[:, 0]
    walks[:, 1] = indices[indptr[cur] + (rng.random(W) * deg[cur]).astype(np.int64)]
    wmax = max(1 / p, 1.0, 1 / q)
    for t in range(2, walk_len):
        cur, prev = walks[:, t - 1], walks[:, t - 2]
        nxt = np.empty(W, dtype=np.int64)
        pending = np.arange(W)
        while pending.size:
            c, pv = cur[pending], prev[pending]
            cand = indices[indptr[c] + (rng.random(pending.size) * deg[c]).astype(np.int64)]
            k = pv * n + cand
            pos = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
            is_nb = keys[pos] == k                                   # cand adjacent to prev -> weight 1
            w = np.where(cand == pv, 1 / p, np.where(is_nb, 1.0, 1 / q))
            ok = rng.random(pending.size) * wmax < w                 # rejection sampling
            nxt[pending[ok]] = cand[ok]
            pending = pending[~ok]
        walks[:, t] = nxt
    return walks.astype(np.int32)


def _context_pairs(walks, window):
    """All (centre, context) pairs within `window` steps, both directions."""
    cs, xs = [], []
    for k in range(1, window + 1):
        a, b = walks[:, :-k].ravel(), walks[:, k:].ravel()
        cs += [a, b]
        xs += [b, a]
    return np.concatenate(cs), np.concatenate(xs)


def sgns(walks, n, dim=64, window=5, neg=5, epochs=2, lr0=0.025, batch=1024, seed=0):
    rng = np.random.default_rng(seed)
    cen, ctx = _context_pairs(walks, window)
    freq = np.bincount(walks.ravel(), minlength=n).astype(float) ** 0.75
    cdf = np.cumsum(freq / freq.sum())
    Win = ((rng.random((n, dim)) - 0.5) / dim).astype(np.float32)
    Wout = np.zeros((n, dim), dtype=np.float32)
    total = epochs * int(np.ceil(len(cen) / batch))
    step = 0
    for _ in range(epochs):
        perm = rng.permutation(len(cen))
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            u, v = cen[idx], ctx[idx]
            negs = np.minimum(np.searchsorted(cdf, rng.random((len(idx), neg))), n - 1)
            lr = lr0 * max(1e-4, 1 - step / total)
            step += 1
            hu, ov, on = Win[u], Wout[v], Wout[negs]
            g_pos = 1 / (1 + np.exp(-np.einsum("bd,bd->b", hu, ov))) - 1          # d loss / d score, label 1
            g_neg = 1 / (1 + np.exp(-np.einsum("bd,bkd->bk", hu, on)))            # label 0
            d_in = g_pos[:, None] * ov + np.einsum("bk,bkd->bd", g_neg, on)
            np.add.at(Win, u, -lr * d_in)
            np.add.at(Wout, v, -lr * g_pos[:, None] * hu)
            np.add.at(Wout, negs.ravel(), -lr * (g_neg[:, :, None] * hu[:, None, :]).reshape(-1, dim))
    return Win


def ppmi_svd(walks, n, dim=64, window=5, alpha=0.75, seed=0):
    c, x = _context_pairs(walks, window)
    C = sp.coo_matrix((np.ones(len(c), dtype=np.float32), (c, x)), shape=(n, n)).tocsr()
    rows = np.asarray(C.sum(1)).ravel()
    cols = np.asarray(C.sum(0)).ravel() ** alpha                              # context-distribution smoothing
    tot, ctot = rows.sum(), cols.sum()
    C = C.tocoo()
    pmi = np.log(C.data / tot) - np.log(rows[C.row] / tot) - np.log(cols[C.col] / ctot)
    M = sp.coo_matrix((np.maximum(pmi, 0), (C.row, C.col)), shape=(n, n)).tocsr()
    M.eliminate_zeros()
    U, S, _ = svds(M, k=dim, random_state=seed)
    return (U * np.sqrt(S)).astype(np.float32)
