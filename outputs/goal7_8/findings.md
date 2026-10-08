# Goals 7 and 8: Disruption cascades and anomalous airports

All numbers below were produced by the scripts in `goal7_8/` and are stored in the CSVs and logs in this folder.
Run from the project root, in this order (seeds are fixed):

```
.venv/bin/python goal7_8/goal7a_resilience.py
.venv/bin/python goal7_8/goal7b_cascade.py
.venv/bin/python goal7_8/goal8_anomaly.py
.venv/bin/python goal7_8/goal8b_degree_controlled.py   # reads the CSV from goal8_anomaly.py
```

## Goal 7a: node-removal resilience (`goal7a_resilience.py`)
Undirected projection, removal in 1% batches, random removal averaged over 20 seeds. Intact LCC fraction 0.992.
Global efficiency is an estimate: 300 fixed sampled BFS sources, the same for every strategy, removed airports count as 0. Intact value 0.262.

Fraction of airports removed when the metric falls below 50% of its intact value:

| Strategy | LCC below 50% | Efficiency below 50% |
|---|---|---|
| Random (20 seeds) | 0.371 (sd 0.014) | 0.207 (sd 0.017) |
| Betweenness, adaptive | 0.037 | 0.020 |
| Betweenness, initial ranking | 0.059 | 0.021 |
| PageRank, initial ranking | 0.069 | 0.022 |
| Degree, initial ranking | 0.073 | 0.033 |
| Degree, adaptive | 0.078 | 0.031 |

- Targeted removal is far more damaging than random failure: about 4-8% of airports versus about 37% to break the LCC.
- Efficiency collapses before connectivity does (roughly 2-3% removed under targeted attack).
- Adaptive betweenness is the strongest attack, so removals reshuffle which airports are the bridges.
- Unexpected and not investigated: adaptive degree (0.078) is slightly weaker than static degree (0.073).
- Files: `resilience_curves.csv`, `resilience_summary.csv`, `resilience_random_all_seeds.csv`, `resilience_curves.png`.

## Goal 7b: Independent Cascade (`goal7b_cascade.py`)
- Model: failure spreads along directed routes u to v; each route passes it on once with probability p, with no recovery inside a cascade (equivalent to SIR with one-step recovery). Two variants: fixed p, and p_uv = 1-(1-p)^route_count (each airline route an independent channel).
- Setup: 20 seeds per class (top degree, top betweenness, top PageRank, random), 200 trials per seed, 11 values of p.
- Spectral radius of the adjacency matrix is 69.28, so the mean-field threshold is about 1/lambda_max = 0.0144 (a heuristic only).
- Smallest p at which the mean cascade reaches 5% of N: about 0.03 for hub seeds and 0.05 for random seeds (fixed p); 0.02 and 0.03 under weighted p. Growth is smooth, not a sharp transition (finite size), so this is threshold-like.
- Hubs beat random seeds: at p = 0.05 about 17-18% of N versus 5.6%; at p = 0.5, 73.6% versus 56.8%.
- The three hub classes are nearly indistinguishable (e.g. 0.5647, 0.5647, 0.5651 at p = 0.3) because the seed sets overlap heavily.
- Why random seeds do worse was not checked; low out-degree of many random airports is a suggested explanation only.
- Limitation: no capacity, rerouting or timing. This is a structural model, not an operational simulator.
- Files: `cascade_results.csv`, `cascade_thresholds.csv`, `cascade_seed_sets.csv`, `cascade_curves.png`.

## Goal 8: anomalous airports (`goal8_anomaly.py`, `goal8b_degree_controlled.py`)
Flags are for inspection only; the interpretations are hypotheses and were not verified against external data.

- Regression: log10(B) = 0.181 + 2.039*log10(k), R^2 = 0.345, on the 1,827 airports with betweenness above 0 (1,598 have zero).
- Betweenness/degree ratio, top flagged: Greenland (SFJ, JAV, JQA, UAK, GOH), Anchorage (ANC, degree 34, betweenness 424,761), Panama (DAV, BOC), Thunder Bay (YQT), Kindu (KND). They look like gateway airports bridging regional clusters. Three of the top 9 have degree 2 or 3, where the ratio is noisy.
- Residual z-score (degree at least 5): Greenland airports first, then Kodiak (ADQ) and remote airports in Brazil and Canada.
- Isolation Forest and LOF disagree (Spearman 0.24, top-15 lists share 0 airports); Isolation Forest and the residual list also share none. The plain Isolation Forest top-15 is mostly the largest hubs plus BET, SYD and AEP, which are extreme on every feature rather than unusual in combination.
- Degree-controlled Isolation Forest (features residualised on degree) surfaces BET, GEA, FAI, OME, HIR, WIL, PAC and AEP: high betweenness for their degree and core number well below what degree predicts. A plausible reading is regional gateway or spoke-consolidation roles.
- Some airports lack name/country in airports.dat (AGM, KUS, LKE, FBS, RCE) and show blank values.
- Files: `anomaly_all_airports.csv`, three top-15 CSVs, `anomaly_top15_isoforest_degree_controlled.csv`, `anomaly_plots.png`.

Syllabus mapping: Goal 7 maps to CO2, CO4 and Unit 2 (cascade and epidemic models); Goal 8 maps to Unit 3 (anomaly detection).
