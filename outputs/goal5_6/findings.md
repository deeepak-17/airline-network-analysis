# Goals 5 and 6: Link prediction and graph embeddings

All numbers come from the scripts in `goal5_6/` and are stored in the CSVs and logs in this folder.
Seeds are fixed (split seeds 42-46, embedding seeds 0-2). Run from the project root, in this order:

| Step | Command | Time | Writes |
|---|---|---|---|
| 1 | `.venv/bin/python goal5_6/goal5_link_prediction.py` | ~4 min | `split_*.csv`, `goal5_results_canonical.csv`, `goal5_results_mean_sd_over_5_splits.csv`, `goal5_roc.png` |
| 2 | `.venv/bin/python goal5_6/goal6_embeddings.py` | ~25 min | `emb_*.npy`, `goal6_pq_grid.csv`, `goal6_best_pq.json` |
| 3 | `.venv/bin/python goal5_6/goal6_node_tasks.py` | ~2 min | `goal6_node_tasks_summary.csv`, `goal6_confusion_matrices.png`, `goal6_embedding_projection_continent.png` |
| 4 | `.venv/bin/python goal5_6/goal6_link_prediction.py` | ~5 min | `goal5_vs_goal6_comparison_table.csv`, `goal6_lp_mean_sd_over_embedding_seeds.csv`, `goal6_lp_roc.png` |

Shared modules: `lp_common.py` (split, features, metrics) and `node2vec_np.py` (node2vec written in numpy, because gensim does not build on Python 3.14).

## Setup
- Undirected projection: 3,425 nodes, 19,256 edges, 8 components. Direction, airline and frequency are lost.
- 15% of edges held out as test positives (2,888). Test edges are never taken from a random spanning forest, so training keeps the same components. Bridge and leaf edges are therefore never tested, which makes the task easier than "predict any new route".
- Nested validation: a second 15% of training edges is held out; supervised models train on those pairs with features from the inner graph only, so training pairs are disjoint from test pairs. All negatives are true non-edges and mutually disjoint (asserted in code).
- Test sets are 1:1 balanced, with two negative schemes: "uniform" (random pairs) and "degree" (endpoints drawn in proportion to training degree; harder).
- Precision@k ties are broken randomly with a fixed seed. P@100 saturates at 1.0 under uniform negatives, so P@npos (k = 2,888) is the informative cut. Absolute precision is inflated by the 1:1 balance; compare methods with each other only.
- Embeddings: node2vec with exact biased walks and skip-gram with negative sampling (SGNS), dim 64, 10 walks, length 40, window 5, 3 epochs, p=2, q=0.5 (chosen on validation pairs only). PPMI+SVD on the same walks is reported as an ablation.
- Embeddings are learned on the training graph only. A "naive" protocol (classifier trained on edges the walks already saw) is reported next to the clean one.

## Goal 5: heuristics and supervised models (mean +/- sd over 5 splits)

| Method | Uniform AUC | Uniform P@npos | Degree-neg AUC | Degree-neg P@npos |
|---|---|---|---|---|
| Adamic-Adar | 0.9899 +/- 0.0007 | 0.9711 | 0.9220 +/- 0.0032 | 0.8458 |
| Resource allocation | 0.9898 +/- 0.0008 | 0.9711 | 0.9370 +/- 0.0022 | 0.8682 |
| Common neighbours | 0.9893 +/- 0.0006 | 0.9679 | 0.9114 +/- 0.0035 | 0.8310 |
| Jaccard | 0.9785 +/- 0.0009 | 0.9623 | 0.9300 +/- 0.0027 | 0.8594 |
| Preferential attachment | 0.9779 +/- 0.0011 | 0.9271 | 0.6996 +/- 0.0048 | 0.6433 |
| Geographic distance (non-graph baseline) | 0.9044 +/- 0.0024 | 0.8318 | 0.8969 +/- 0.0038 | 0.8359 |
| LogReg on structural features | 0.9953 +/- 0.0006 | 0.9717 | 0.9162 +/- 0.0063 | 0.8375 |
| RandomForest on structural features | 0.9944 +/- 0.0007 | 0.9720 | 0.9121 +/- 0.0098 | 0.8479 |
| HistGradBoost on structural features | 0.9945 +/- 0.0007 | 0.9697 | 0.9090 +/- 0.0100 | 0.8421 |

- Uniform negatives: the common-neighbour-style heuristics are indistinguishable. Preferential attachment is surprisingly strong (0.978).
- Degree-matched negatives: preferential attachment collapses to 0.700, so its uniform score mostly reflects "hubs connect to things" and uniform negatives overstate how easy the task is.
- The supervised models are marginally better on uniform AUC and no better on P@npos. On degree-matched negatives they lose to resource allocation (0.916 vs 0.937). A possible reason is that they were trained on uniform negatives; this was not tested.
- Geography alone gives AUC about 0.90 under both schemes. Topology beats it by about 0.09 under uniform negatives but only about 0.02-0.04 under degree-matched ones. A combined model was not tested.

## Goal 6a: node classification (embeddings on the training graph only, 5-fold CV, 3 seeds)

| Task | Model | Accuracy | F1 |
|---|---|---|---|
| Continent (6 classes, 3,147 labelled nodes; macro F1) | majority | 0.299 | 0.077 |
| | degree only | 0.314 | 0.137 |
| | node2vec LogReg | 0.968 | 0.966 |
| | node2vec RF | 0.970 | 0.966 |
| Hub (top 5% by full-graph degree; positive-class F1) | majority | 0.949 | 0.000 |
| | degree only | 0.995 | 0.950 |
| | node2vec LogReg | 0.960 | 0.502 |
| | node2vec RF | 0.950 | 0.038 |

- Continent labels come from the airports.dat timezone prefix plus a country list, not from the graph, so there is no label leakage. The 163 nodes missing from airports.dat and the ambiguous Atlantic/Indian/Arctic/Antarctica prefixes were dropped. The high score largely reflects walks staying within a continent: a structure-only embedding encodes geography.
- Hub status: embeddings do not beat a one-feature degree baseline, so the hypothesis that they capture hub status beyond degree is not supported. Thresholds and class weights were not tuned on this imbalanced label.
- A hub label defined from the same graph that was embedded is trivial (degree-only RF reaches F1 1.000); this leaky variant is reported only to show the leakage.

## Goal 6b: embedding link prediction vs Goal 5 (same split)
Goal 5 rows use the canonical split (seed 42), one run each. Goal 6 rows are the mean over 3 embedding seeds.

| Method | Uniform AUC | Uniform P@npos | Degree-neg AUC | Degree-neg P@npos |
|---|---|---|---|---|
| Goal 5 Adamic-Adar | 0.9897 | 0.9723 | 0.9258 | 0.8549 |
| Goal 5 resource allocation | 0.9896 | 0.9720 | 0.9394 | 0.8729 |
| Goal 5 LogReg structural | 0.9958 | 0.9730 | 0.9233 | 0.8511 |
| node2vec cosine (unsupervised) | 0.9182 +/- 0.0013 | 0.8599 | 0.9145 +/- 0.0008 | 0.8527 |
| node2vec LR, L1 edge feature | 0.9892 +/- 0.0005 | 0.9617 | 0.9484 +/- 0.0018 | 0.8837 |
| node2vec RF, Hadamard | 0.9844 +/- 0.0007 | 0.9443 | 0.9329 +/- 0.0014 | 0.8635 |
| node2vec LR, Hadamard | 0.9053 +/- 0.0060 | 0.8428 | 0.8659 +/- 0.0053 | 0.7907 |
| node2vec Hadamard + structural LR (hybrid) | 0.9965 +/- 0.0002 | 0.9767 | 0.9402 +/- 0.0014 | 0.8703 |
| PPMI-SVD cosine (ablation) | 0.9404 | 0.8922 | 0.9334 | 0.8755 |

- Uniform negatives: embeddings alone do not beat the heuristics. The best embedding-only classifier (L1) ties common neighbours at about 0.989; cosine is clearly weaker. The hybrid is nominally best (0.9965 vs 0.9958), a tie within noise.
- Degree-matched negatives: the L1 embedding classifier is best (0.9484 vs 0.9394 for resource allocation and 0.9233 for structural LogReg). This is the only setting where learned embeddings win, and it rests on one split, so treat it as suggestive. No ablation explains why.
- The edge operator matters more than the model: L1 with a linear classifier is strong, Hadamard with a linear classifier is poor.
- Seen-edge bias is small: for L1 under uniform negatives the naive protocol gives 0.9919 vs 0.9892 clean. The clean protocol is the defensible one.
- SGNS may be under-trained: PPMI-SVD cosine (0.940) beats SGNS cosine (0.918), and a quick check showed SGNS validation AUC still rising with more epochs. The node2vec numbers may understate the method.

## Limitations
- Undirected projection loses direction, airline and frequency information.
- The Goal 5 vs Goal 6 comparison uses a single canonical split; only Goal 5 alone was repeated over 5 splits.
- Test sets are 1:1 balanced and exclude bridge and leaf edges.
- p and q came from a 5-point grid and one seed; SGNS epochs and learning rate came from a quick check, not a systematic search.
- No slots, bilateral agreements or demand are modelled, so predicted routes are structurally plausible, not commercially viable.

Syllabus mapping: Goal 5 maps to CO4, CO5 and Unit 2 (link prediction); Goal 6 maps to CO4 and Unit 3 (graph representation learning).
