# Goal 3 findings (all numbers from goal3/run_goal3.py; full log in run_log.txt)

Graph: 3425 airports, 37594 directed edges (self-loops dropped, duplicate airline routes collapsed; route_count = #airline routes). 163 nodes have no airports.dat metadata (blank names/country).

## Methods
- PageRank: networkx, alpha=0.85, directed, tol 1e-12; weighted variant uses route_count. Dangling nodes redistribute uniformly.
- Personalised PR: restart AND dangling mass both go uniformly to the 71 Indian airports (country=='India').
- Betweenness (for comparison only): APPROXIMATE, 500 sampled sources, seed 42.
- SimRank: FULL, all 3425 nodes, not a subgraph. Sparse-matrix form S <- C * W^T S W, diag reset to 1, C=0.8, W = column-normalised directed adjacency (in-neighbour SimRank). Converged to max-change < 1e-4 after 23 iterations. This is the matrix approximation of Jeh-Widom (diag forced), not the exact fixed point.

## PageRank vs degree
- Spearman(undirected degree, PR)=0.852, Kendall=0.707. Top-k overlap PR vs degree: 9/10, 12/15, 38/50, 79/100. So PR is largely a degree proxy here; divergence is a second-order effect.
- Weighted PR vs unweighted PR: Spearman 0.978, but top-15 changes (JFK, LHR, SIN, MIA, LAX enter; IST falls from #2 to #15; DXB, IAH, SYD, YYZ leave). Route-count weighting favours airports with many airlines per edge.
- PR vs approx. betweenness: Spearman 0.760; degree vs betweenness 0.764. Betweenness top-15 includes low-degree gateways (ANC, GRU, SEA) absent from PR/degree top lists.
- Top-15 PR: ATL, IST, ORD, DEN, DFW, DME, CDG, FRA, PEK, AMS, DXB, IAH, LAX, SYD, YYZ. SYD (undirected degree 83) is in the PR top-15 despite moderate degree.

## Does the divergence come from "who they connect to"? (hypothesis tested, partly REJECTED)
- log PR ~ log in-degree: slope 0.616 (OLS with covariates: 0.79), R^2=0.81.
- Hypothesis: residual PR is explained by high mean PR of in-neighbours. Result: Spearman(residual, log mean in-neighbour PR) = -0.545 (in_deg>=5), NEGATIVE, vs stratified-shuffle null -0.057 +/- 0.026. So the naive "connected to important airports" story is not supported; if anything it is reversed. Likewise, fraction of in-neighbours in PR top-50 correlates -0.30 with the residual. Adding log mean neighbour PR raises R^2 only 0.816 -> 0.889 with a negative coefficient (-0.449).
- Why: mass passed along an edge is PR(m)/outdeg(m). Hubs have huge out-degree, so each link from a hub carries little. The relevant quantity is neighbour EXCLUSIVITY (low out-degree of in-neighbours): Spearman(residual, mean log in-neighbour out-degree) = -0.749; Spearman(residual, mean per-edge share PR/outdeg of in-neighbours) = +0.714. OLS R^2: log in-deg + mean log nbr out-degree = 0.957. (Adding mean per-edge share gives R^2=0.997, but that is near-tautological because it is PR's defining equation; do not present it as a finding.)
- Structural reading: high-PR-for-its-degree airports (HIR, VLI, FAI, BET, ANC, PPT, WLG, CHC, ADL, CNS, NAN; deg 20-35, PR rank 44-250 vs degree rank 290-420) are regional gateways (Oceania, Alaska, Pacific) whose in-neighbours are small, low-out-degree airports that funnel their whole share into them, and the region is a small dense island so PR mass circulates. Low-PR-for-degree airports (SUF, SZG, SCQ, VRN, MAH, DJE, BFS, CWL...) are European leisure/low-cost airports whose in-neighbours are high-out-degree hubs/LCC bases that split mass widely. Do NOT label this "hub quality"; the evidence is about in-neighbour out-degree concentration. Caveat: unweighted, no passenger volumes; correlation, not causal.
- Rank-gap tables: rankgap_pr_better_than_degree.csv / rankgap_pr_worse_than_degree.csv (band: undirected degree 20-120, average-rank ties). An earlier band of degree 5-60 was dominated by massive degree ties (rank 1107) and was discarded.

## Personalised PageRank (India)
- Top-15 by PPR: BOM, DEL, CCU, BLR, HYD, GAU, MAA, DXB, SHJ, DOH, COK, SIN, IMF, BKK, SXR. Spearman(PR, PPR)=0.458; (degree, PPR)=0.711.
- Top non-Indian by PPR: DXB, SHJ, DOH, SIN, BKK, KUL, AUH, JED, RUH, MCT, IST, DMM, HKG, CMB, FRA. Largest lift PPR/PR among non-Indian airports with deg>=20: MCT 5.9x, SHJ 5.7x, CMB 5.5x, DMM 5.4x, KTM 4.8x. That is, India's gateways are Gulf + SE Asia + South Asia, while global PR top (ATL, ORD, DEN, DFW) vanish. Note: the walk restarts only at India with p=0.15 so PPR is strongly local by construction; "importance relative to India" means reachability within few hops.

## SimRank
- Raw top pairs (both in-degree 1) all equal C=0.8 trivially (shared single in-neighbour), e.g. EAR-MTJ, KMS-TKD; discarded as uninformative. Filtered to in-degree>=3 both: top pairs SLH-ZGU (0.400, Vanuatu), YCS-YUT (0.382, Canada), UAH-UAP (0.367, Fr. Polynesia), MTV-TOH, KKH-KPN, GEA-LIF, plus a New Zealand regional clique (NPL/TRG/ROT/BHE/NPE, 0.318). Mostly small airports sharing the same hub/feeder set.
- Nearest neighbours (in-degree>=3 candidates): DEL -> SXR, IXL, IXJ, BOM, IXZ, PAT, IDR, IXR (all India); BOM -> IXL, SXR, IXZ, IDR, IXR, RPR, DEL, IXJ; ATL -> PIE, PGD, SFB, CLT, DCA, DFW, LGA, DTW (all US); BLR -> IXZ, RPR, PNQ, HYD ... (all India); JFK, LHR neighbours are low-degree airports (Bahamas, Germany/Algeria/Romania spokes); SimRank values for hubs are tiny (<=0.05) and DEL's most similar airport is not BOM (0.033, rank 4), even though in-neighbour Jaccard(DEL,BOM)=0.57, because SimRank rewards neighbours that are themselves similar, and the scores decay with hub in-degree.
- Interpretation caution: SimRank is defined by neighbourhood structure, not distance, but empirically it correlates with geography: Spearman(SimRank, great-circle km) = -0.610 on 18,146 sampled non-zero pairs (n sampled from the first 20000 of 200000 random pairs; 97.7% of pairs have non-zero score). Cause: neighbourhoods are regional. Also hubs do not get "similar hub" matches; SimRank is dominated by degree-matched low-degree nodes. For hub-hub similarity use in-neighbour Jaccard or embeddings instead.
- Not converged to exact: stop tolerance 1e-4.

## So what / links to other goals
PR is largely degree-driven (rho 0.85), so rankings from Goals 1/3 agree for the very top, but the interesting residual is regional gateways whose importance stems from how concentrated their neighbours' outflow is, which matters for Goal 7 (targeted removal by PR vs betweenness will pick different airports: betweenness top has ANC/GRU/SEA; PR top has IST/DME/DEN) and for Goal 4 (SimRank neighbours fall within the same countries/communities).

CO mapping: CO2, CO3, Unit 1 (link analysis).

Files: centrality_table.csv, top15_*.csv, rank_correlations.csv, rankgap_*.csv, simrank_top_pairs.csv, simrank_nearest.csv, simrank.npy, degree_vs_pagerank.png, run_log.txt.

## Reproduce (single script produces every number, table and plot above)
Script: goal3/run_goal3.py (imports common/load_graph.py; fixed seeds: betweenness sampling seed=42, null shuffles and SimRank pair sampling numpy default_rng(42); no dependence on /tmp or scratchpad; runtime about 10 s).
Command, from /Users/deepak/Downloads/sna_project:
    .venv/bin/python goal3/run_goal3.py > outputs/goal3/run_log.txt 2>&1
Result-to-source map (all in goal3/run_goal3.py, section headers in the code):
- PageRank / weighted / PPR tables: top15_pagerank.csv, top15_pagerank_w.csv, top15_ppr_india.csv, top15_ppr_india_foreign.csv, top15_ppr_lift_deg20.csv, centrality_table.csv ("PageRank", "Personalised PR", "Feature table")
- Correlations and overlaps: rank_correlations.csv, run_log.txt ("Correlations")
- Rank gaps and neighbour-quality tests (shuffle null, OLS R^2): rankgap_*.csv, run_log.txt ("Rank gap + neighbour-quality test")
- Scatter: degree_vs_pagerank.png ("Scatter")
- SimRank: simrank.npy (full 3425x3425 matrix), simrank_top_pairs.csv, simrank_nearest.csv, run_log.txt ("SimRank")
