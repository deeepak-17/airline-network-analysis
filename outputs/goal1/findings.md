# Goal 1 - Structural characterisation (CO1, CO2, Unit 1)

Re-run: `cd /Users/deepak/Downloads/sna_project && .venv/bin/python goal1/run_goal1.py` (~2 min, seed 42).
Script: goal1/run_goal1.py. Every number below is in outputs/goal1/metrics.json (full console log: run_log.txt).
Graph: common/load_graph.py (routes.dat; airports as nodes; directed edge A->B if any airline flies A->B; route_count kept but unused here).

## Preprocessing audit (metrics.json: raw_*)
routes.dat has 67,663 rows; 0 missing source/dest, 1 self-loop (dropped), 0 exact duplicate rows -> 67,662 rows. Multiple airlines on the same A->B pair are collapsed into one unweighted directed edge. 163 of 3,425 nodes have no airports.dat metadata (names show "?"). Directed graph: n=3,425, m=37,594. Undirected projection (edge if either direction exists): m=19,256; loses direction and route multiplicity.

## Basic statistics
density 0.00321 (directed), 0.00328 (undirected). Mean degree 10.98 (in = out), 11.24 (undirected). Median total degree 3; 22.4% of airports have degree 1. Max total degree 248 (AMS).
8 weakly connected components (largest 3,397); 44 SCCs (largest 3,354, 35 singletons). Exact (all-source BFS) path lengths: undirected LCC avg 4.10, diameter 13; directed LSCC avg 4.13, diameter 14; 97.1% of ordered pairs are reachable.

## Degree distribution (degree_distributions.png, degree_histogram.png)
Heavily right-skewed (mean 11, median 3, max ~240). Not a clean straight line over the whole range in log-log; formal fit/model comparison (power law vs truncated/lognormal) is Goal 2's job and is NOT claimed here. In/out distributions are near-identical (max in 238, max out 239, both FRA).

## Centrality (top10_centralities.csv, centralities_all.csv, centrality_spearman.csv)
Betweenness: exact Brandes on the directed graph. Closeness: undirected largest component (directed networkx version also stored, wf_improved). Eigenvector: undirected LCC only (numpy solver cannot handle disconnected graph; nodes outside the LCC are NaN); directed LSCC variant also stored.
- Degree top-5: AMS 248, FRA 244, CDG 240, IST 236, ATL 217.
- Betweenness top-5: ANC, LAX, CDG, DXB, FRA. ANC and LAX rank above AMS/FRA: betweenness picks out inter-regional bridges (Alaska/cargo/transpacific), not the biggest degree.
- Closeness top-5: FRA, CDG, LHR, AMS, DXB. Eigenvector top-5: AMS, FRA, CDG, MUC, FCO (European-dense cluster).
- Spearman with degree: betweenness 0.774, closeness 0.687, eigenvector 0.718. Betweenness vs closeness 0.416. The measures agree only moderately, so "important airport" depends on definition.

## Assortativity (hypothesis: negative)
Undirected degree assortativity r = -0.0065. Directed variants: in-in -0.0100, in-out -0.0103, out-in -0.0105, out-out -0.0108. Slope of log <k_nn>(k) vs log k = -0.013 (knn_vs_k.png).
Result: the hypothesis of clear disassortativity is NOT supported. r is negative but indistinguishable from zero in practice (essentially neutral mixing). Caveat from a null model: 20 degree-preserving rewirings of the undirected graph give r = -0.097 +/- 0.004. So the observed network is MORE assortative than random wiring with the same degree sequence (difference ~ +0.09, ~20 null SDs). Interpretation: for a simple graph with this fat-tailed sequence, the structural expectation is mildly disassortative; the real network cancels that through hub-hub linking (see k-core). Likely mechanism: hubs link to other hubs (inter-hub trunk routes) AND to many low-degree spokes, which offset each other. Not verified causally here.

## Transitivity, clustering, reciprocity
Global transitivity (undirected, 3*triangles/connected triples) 0.248. Mean local clustering (undirected) 0.487 (0.628 over nodes with degree>=2; degree-1 nodes count as 0). Directed (Fagiolo) average clustering 0.469. Density 0.0033, i.e. clustering is ~two orders of magnitude above an Erdos-Renyi expectation (Goal 2 will test this with simulations). Local clustering vs degree: clustering_vs_degree.png.
Reciprocity: 97.56% of directed edges have a reverse edge (36,676 of 37,594; 918 one-way edges), 98.43% route-weighted. ER expectation is ~0.0032. Meaning: routes are almost always operated in both directions, so the directed graph is near-symmetric and the undirected projection loses little here; the directed/undirected distinction matters little for this dataset (consistent with nearly equal directed and undirected assortativity).

## Degeneracy / k-core (kcore_sizes.csv, innermost_core.csv, kcore.png)
Computed on the undirected projection. Maximum core number 31; the 31-core has 93 airports (shell sizes in kcore_sizes.csv; 1-core = all 3,425; 10-core 605). Innermost core is almost entirely European: Spain 11, UK 11, France 9, Germany 9, Italy 8, plus Greece, Turkey, Russia, Portugal etc.; non-European members include DXB, DOH, TLV, CMN, RAK, TUN and one US airport (value in innermost_core.csv). Its internal density is 0.528, so it is not a clique. Only 6 of the 10 highest-degree airports are in it (AMS, FRA, CDG, IST, MUC, DME); ATL, PEK, ORD are NOT, nor are LAX/ANC. Spearman core number vs degree 0.972, vs betweenness 0.676. Meaning: coreness measures dense mutual connectivity (European short-haul network), whereas betweenness measures bridging; large US/Chinese hubs have huge degree but sparser neighbourhoods. Caveat: the data-set coverage (OpenFlights) is uneven and Europe-heavy, which affects the k-core result.

## How assortativity (near zero) explains the rest of the structure
Mixed picture, not the textbook hub-and-spoke story. A strongly negative r would say hubs mostly feed degree-1/2 spokes; instead r ~ 0 and r above the rewired null tells us hubs also interconnect. That is consistent with (i) the high clustering (0.248 transitivity, 0.49 mean local) and a deep k-core of 93 densely linked European airports, (ii) a short average path (4.1 hops, diameter 13) with ~97% reachability, and (iii) betweenness/closeness leaders (LAX, ANC, DXB, FRA) being bridges between regional clusters rather than just the largest-degree nodes. The 22% degree-1 airports are still spokes of hubs, so both patterns coexist: a core-periphery structure (dense hub-to-hub core + spokes) rather than pure disassortative hub-and-spoke. This is an interpretation consistent with the numbers, not proven by them; community structure (Goal 4) and resilience (Goal 7) are the tests.

## Limitations
Edges unweighted (frequency/capacity ignored); OpenFlights data are not a complete or current schedule (Istanbul Airport and Berlin-Tegel both present, indicating mixed vintage); codeshare rows counted as routes; the closeness/eigenvector are on the undirected LCC only; no significance testing of degree-distribution shape here.
