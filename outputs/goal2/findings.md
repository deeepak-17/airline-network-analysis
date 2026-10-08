# Goal 2 - Growth model comparison (CO1, Unit 1)

Script: goal2/run_goal2.py. Re-run (about 7 min): `cd /Users/deepak/Downloads/sna_project && .venv/bin/python goal2/run_goal2.py`
Seeds: model seeds 0-9 (WS p-sweep uses seeds 0-2). Log of the run: outputs/goal2/run_log.txt.
Graph: undirected simple projection of the directed route graph (common/load_graph.py; direction and route multiplicity are lost; no self-loops).
n=3425, m=19256, mean degree 11.24, max degree 248, LCC = 99.18% of nodes. Path length is exact BFS on the LCC. Measured degree assortativity = -0.0065 (undirected projection), essentially zero, so the "disassortative" hypothesis is NOT supported at this level.

## Model matching (all in model_comparison.csv, model_metrics_per_seed.csv, ws_p_sweep.csv)
- ER: n=3425, p=2m/(n(n-1))=3.284e-3.
- BA: m=round(k/2)=6 -> 20514 edges (6.5% more than empirical; m is an integer, so a match is not exact).
- WS: k=12 (20550 edges). p chosen from a grid as the value whose transitivity is nearest the empirical one: p=0.3 (grid is coarse: 0.2 and 0.5 are the neighbours).
- Metrics are mean +/- sd over 10 seeds.

| metric | Empirical | ER | WS (p=0.3) | BA (m=6) |
|---|---|---|---|---|
| transitivity | 0.2483 | 0.0033 +/- 0.0002 | 0.2296 +/- 0.0032 | 0.0140 +/- 0.0005 |
| avg clustering | 0.4871 | 0.0033 +/- 0.0002 | 0.2378 +/- 0.0033 | 0.0180 +/- 0.0013 |
| avg path (LCC) | 4.103 | 3.633 +/- 0.010 | 3.953 +/- 0.008 | 3.191 +/- 0.016 |
| max degree | 248 | 25.4 +/- 1.2 | 19.3 +/- 0.7 | 251 +/- 38 |
| assortativity | -0.0065 | 0.001 +/- 0.006 | -0.020 +/- 0.006 | -0.033 +/- 0.005 |
| KS(degree) vs empirical | - | 0.644 | 0.744 | 0.677 |

(KS = two-sample KS between degree samples, mean over 10 seeds; sd of KS is 0.004/0.002/0.000 for ER/WS/BA.)

Observations:
- ER and WS cannot produce hubs (max degree about 19-25 vs 248). BA reproduces the maximum degree but not clustering (transitivity 18x too low) and has a minimum degree of 6, so its low-degree body is wrong. Thus BA's KS distance (0.68) is not better than ER's (0.64); all three KS values are large. BA's hub tail is right; the bulk is wrong.
- WS roughly matches transitivity and path length only because p was tuned for it, and it has no heavy tail. Its avg clustering (0.238) is still half the empirical 0.487, so even tuned WS does not match the clustering coefficient fully. WS p-sweep shows no single p gives both (see hist_and_clustering_vs_path.png).
- No model reproduces all of: heavy tail + high clustering + path length about 4.1. The empirical path length is longer than ER/BA at equal density, consistent with geographic/regional structure (inference; not tested directly here).
- Assortativity: empirical (-0.0065) is within about 1 sd of ER (0.001+/-0.006); BA gives -0.033. Assortativity does not discriminate the models.

## Degree-distribution fitting (powerlaw package, discrete; powerlaw_fit.csv, distribution_comparisons.csv, xmin_sensitivity.csv)
- Fitted xmin=2, alpha=1.738, KS D=0.032, n_tail=2657 of 3425. Alpha < 2 is unusually small; it signals the fit is being driven by a short, curved range, not a clean scale-free regime.
- Pairwise log-likelihood ratios (R normalised; positive favours first; p-values from the package; p printed as 0 is numerical underflow):
  - power law vs truncated power law: R=-9.41, p~0 -> truncated PL preferred.
  - power law vs lognormal: R=-6.40, p=1.6e-10 -> lognormal preferred.
  - truncated PL vs lognormal: R=+10.06, p=8e-24 -> truncated PL preferred.
  - truncated PL vs exponential: R=+20.3 -> truncated PL preferred.
  - truncated PL vs stretched exponential: R=+9.64, p=5e-22 -> truncated PL preferred.
  - Overall winner on likelihood: truncated power law (alpha=1.536, lambda=0.00593, i.e. cutoff scale 1/lambda about 169, inside the observed range up to 248). Pure power law is rejected in favour of it, and also beaten by lognormal and stretched exponential.
- Sensitivity (xmin in {1,2,3,5,10,20}): truncated PL beats pure PL and lognormal at all six xmin values. Exponential beats pure PL only at xmin=10 (not significant, p=0.21) and 20 (p=0.001), i.e. only deep in the tail where n_tail is 749/461; at small xmin pure PL beats exponential. The conclusion "not a pure power law; truncated PL best" is robust to xmin; the contest against exponential is xmin-dependent.
- Caveats: (1) the KS D of the truncated PL (0.068) and lognormal (0.069) is larger than the pure PL's (0.032), because the xmin was selected by minimising the pure-PL KS, so likelihood ratio and KS disagree on which is "closest"; likelihood ratio tests only say which candidate is relatively better, not that it is a good absolute fit. No goodness-of-fit bootstrap was run (not done). (2) The undirected projection merges in/out-edges. (3) Degrees are counts of distinct neighbours, not route frequency.

## Hypothesis verdict
- "Not a clean scale-free": SUPPORTED. Pure power law is significantly worse than truncated PL at every tested xmin, and BA fails clustering and the low-degree body.
- "Truncated PL fits better": SUPPORTED on likelihood ratios (against PL, lognormal, exponential, stretched exponential). But the truncation parameter is small (cutoff about 169 vs max 248), the lognormal is not far behind in KS terms, and no absolute goodness-of-fit test was done, so "truncated PL is the true mechanism" is not established.
- Why truncated? The data cannot identify the mechanism. Capacity/runway/slot limits and geography are plausible explanations for a finite cutoff and for the high clustering (regional airports linking to the same nearby hubs), and aging/saturation of preferential attachment is a possible account, but this study did not test them (no capacity, airport-age or time-resolved data; routes.dat is a single snapshot). These should be written as hypotheses, not findings. Preferential attachment is not "refuted": BA reproduces the hub scale (max degree 251 vs 248) but not clustering or low-degree nodes; a model adding geography/fitness/cutoff would be needed.

## Files (all in /Users/deepak/Downloads/sna_project/outputs/goal2/)
- model_comparison.csv (summary mean/std), model_metrics_per_seed.csv, ws_p_sweep.csv
- powerlaw_fit.csv, distribution_comparisons.csv, xmin_sensitivity.csv
- degree_ccdf_models_and_fits.png (CCDF empirical vs models, and tail fits), hist_and_clustering_vs_path.png
- run_log.txt (stdout of the exact run)
