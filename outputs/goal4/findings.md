# Goal 4 - Community detection (CO5, CO3, Unit 2: Evaluation of Community Detection Methods)

All numbers below come from the logs/CSVs in outputs/goal4/. Fixed seeds (Louvain/Leiden seeds 0-9; CPM deterministic; null shuffles seed 0). Re-run everything from the project root, in this order:

    .venv/bin/python goal4/run_disjoint.py      # Louvain + Leiden, stability, resolution sweep, weighted check
    .venv/bin/python goal4/run_overlap.py       # clique percolation k=3,4,5,6,8   (~35 s)
    .venv/bin/python goal4/verify_cpm.py        # sparse CPM == networkx CPM check
    .venv/bin/python goal4/run_interpret.py     # geography, alliances, plots

Support modules: goal4/common4.py (metrics: conductance, coverage, Shen EQ), goal4/geo_maps.py (hand-built country->continent), goal4/alliances.py (hand-built alliance list).

## Setup and decisions
- Graph: undirected PROJECTION of the shared directed graph (edge iff A->B or B->A), 3425 nodes, 19256 edges, 8 connected components, unweighted. Loses direction, reciprocity and route multiplicity. A weighted Louvain (weight = airline-route rows, both directions) is a sensitivity check only.
- Geography: 3262 of 3425 nodes have country metadata (163 airports are not in airports.dat; excluded from geo scores only). Continent comes from a hand-built country list; Russia split at lon 60E, Turkey->Asia, Cyprus->Europe, all US airports (incl. Hawaii) -> North America. Country-level mapping, so a few boundary cases are arbitrary.
- Alliances: hand-built IATA airline lists (goal4/alliances.py), union of members over ~2010-2024 because the OpenFlights snapshot is undated. NOT an official dataset. Rows of routes.dat (incl. codeshare rows) are classified Star/oneworld/SkyTeam/None.

## Disjoint methods (run_disjoint.py -> disjoint_stability.csv, disjoint_eval.csv, louvain_vs_leiden.json, leiden_resolution_sweep.csv, louvain_weighted_sensitivity.json)
| | Louvain (10 seeds) | Leiden (10 seeds) |
|---|---|---|
| Modularity Q mean +- std | 0.6633 +- 0.0019 | 0.6659 +- 0.0010 |
| #communities (min-max) | 26-32 (mean 28.8) | 25-30 (mean 26.7) |
| pairwise NMI across seeds (mean / min) | 0.900 / 0.838 | 0.926 / 0.878 |
| pairwise ARI mean | 0.845 | 0.881 |
Best-Q partitions: Louvain Q=0.6660, 27 communities; Leiden Q=0.6673, 26 communities; sizes median 17 / 14.5, max 681 / 672. Mean conductance 0.086 / 0.087; size-weighted 0.128 / 0.130; coverage 1.0. Louvain vs Leiden best partitions: NMI 0.965, ARI 0.966.
- Both are stable but not identical across seeds (NMI min 0.84-0.88): the modularity landscape is degenerate, so quote the range, not one partition. Leiden is slightly higher-Q and more stable; the difference (0.0026 in Q) is ~1-2 seed std, so do not over-claim it.
- Resolution (Leiden, RB-configuration, seed 0): gamma 0.25/0.5/1/2/4 -> 17/23/28/38/62 communities, Q(gamma=1)=0.577/0.632/0.664/0.627/0.530, mean conductance 0.032/0.069/0.102/0.249/0.375. Number of communities and conductance depend strongly on resolution; "26 communities" is a property of gamma=1, not of the network. Modularity has a resolution limit and tends to merge small groups.
- Weighted Louvain (5 seeds): weighted Q 0.658, 24-27 communities, NMI 0.908 vs unweighted best -> unweighted conclusions are reasonably robust to weighting.

## Overlapping method: clique percolation (run_overlap.py -> overlap_eval.csv, cpm_cliques.json, cpm_memberships.csv, run_overlap.log)
22780 maximal cliques, max size 24 (enumeration 0.3 s, so no restriction needed). networkx k_clique_communities took >2 min for k=3 (hub airports sit in thousands of cliques), so I used an exact sparse-matrix implementation (clique overlap via B*B^T, connected components); verify_cpm.py confirms identical output to networkx on 60 random cases (log: verify_cpm.log). k=3,4,5,6,8 all run.
| k | #comm | node coverage | largest comm | mean conductance | vol-weighted cond. | overlap nodes | max memberships | Shen EQ |
|---|---|---|---|---|---|---|---|---|
| 3 | 164 | 0.713 | 1960 | 0.668 | 0.505 | 94 | 8 | 0.0713 |
| 4 | 57 | 0.453 | 1373 | 0.688 | 0.533 | 71 | 5 | 0.0689 |
| 5 | 22 | 0.329 | 1071 | 0.742 | 0.563 | 51 | 3 | 0.0613 |
| 6 | 9 | 0.259 | 867 | 0.852 | 0.586 | 37 | 3 | 0.0551 |
| 8 | 7 | 0.173 | 572 | 0.812 | 0.550 | 30 | 2 | 0.0654 |
- Ordinary Newman modularity is undefined for overlapping covers; I report Shen et al. extended modularity EQ (equals Q for disjoint partitions - checked: EQ=0.6660/0.6673 equal the Louvain/Leiden Q). Limitation: uncovered nodes contribute nothing to EQ, so EQ of a partial cover is not strictly comparable with Q of a full partition; read it together with coverage.
- CPM is far worse on this graph by every criterion: EQ ~0.07 vs ~0.67, conductance 0.67-0.85 vs 0.09, and coverage only 17-71%. Overlap is small (mean memberships 1.04-1.06; 30-94 overlapping nodes). Reason (interpretation): the network is hub-and-spoke with many degree-1/2 spoke airports that are in no k-clique, while hubs form one giant percolating clique cluster (largest community 1373 of 1551 covered nodes at k=4). CPM finds the dense hub cores, not the regional partition. This is a statement about suitability for this graph, not that CPM is flawed.
- Size distribution at k=3: 164 communities, median size 3, i.e. many tiny triangle-communities plus one giant (community_eval.png, left).

## Geographic interpretation (geo_alignment.csv, leiden_community_profile.csv, leiden_continent_composition.csv, map_communities.png)
Hard-partition scores on 3262 nodes with country labels (purity null = same partition vs shuffled labels, mean of 50):
| method | NMI country | NMI continent | ARI continent | continent purity (null) | country purity (null) |
|---|---|---|---|---|---|
| Leiden | 0.633 | 0.717 | 0.568 | 0.880 (0.301) | 0.441 (0.172) |
| Louvain | 0.636 | 0.709 | 0.567 | 0.878 (0.302) | 0.446 (0.172) |
| CPM k=4 hardened*, 1529 nodes | 0.189 | 0.187 | 0.038 | 0.406 (0.332) | 0.238 (0.171) |
| Leiden on same 1529 nodes | 0.605 | 0.737 | 0.626 | 0.873 | 0.401 |
*each covered node assigned to its largest CPM community (loses overlap). CPM k=3,5,6,8 rows are in the CSV and are similar to k=4 (continent NMI 0.035-0.21; purity at/near null level for k>=5).
- Leiden/Louvain communities are strongly geographic: continent purity 0.88 vs 0.30 null. The hypothesis "communities correspond to geography" is supported at continent level; at country level it is only moderate (NMI 0.63) because communities are continent-/region-scale and merge several countries.
- Leiden top communities (profile CSV): North America (672, hubs ATL ORD DFW; 636 N.Am nodes + 27 S.Am), East/SE Asia-China (493, PEK PVG CAN; 477 Asia), Europe (509, AMS FRA CDG; 449 Europe + 48 Africa), South America (284), Oceania (276). Community 1 (510; IST DXB DOH JED) is Africa 228 + Asia 267: a Middle East / Africa / South Asia region built around Gulf and Istanbul hubs, so it is not a single continent (continent purity 0.54, and country purity 0.14 even though labelled "India" as the top country). Community 6 (185) is Russia (Asia 116, Europe 65). Alaska (ANC), Canada north, Greenland and Norway form small regional communities (hub-poor regional networks, low conductance 0.01-0.18).
- CPM communities (k>=4) do NOT map to geography: hardened purity is close to the shuffled null and the giant community spans every continent (map middle panel); the small ones are local clusters (Alaska/Canada, Pacific). So the CPM giant is a global hub backbone, not a regional group.

## Airline alliances (leiden_community_profile.csv, alliance_by_community.png, alliance_nmi.csv, run_interpret.log)
Global route-row shares: None 0.510, Star 0.172, SkyTeam 0.160, oneworld 0.157 (excluding codeshare rows: None 0.604, SkyTeam 0.138, Star 0.133, oneworld 0.126).
- Hypothesis "communities are alliance blocks" is NOT supported. NMI between Leiden community and each airport's dominant alliance = 0.103, versus NMI(continent, dominant alliance) = 0.041 and NMI(Leiden, continent) = 0.717. Alliance is weakly related to community structure, and what relation exists is probably mediated by geography (alliance carriers have regional strongholds), which this analysis does not separate causally.
- Per-community dominant alliance (share of intra-community route rows; enrichment = share / global share): North America -> oneworld 0.27 (enrichment 1.71; None 0.38), China/E.Asia -> SkyTeam 0.29 (1.82), Brazil/S.America -> oneworld 0.23 (1.46), Oceania -> oneworld 0.23 (1.47), Europe -> Star 0.13 (0.76, i.e. below global; None 0.67 because of LCCs such as FR, U2), Middle East/Africa -> Star 0.20 (1.16). In every community the non-alliance share is the largest (0.38-0.67 in the 12 largest), and small regional communities are 88-100% non-alliance. So alliances are visible as moderate enrichments at best; low-cost and regional carriers dominate the community-internal routes.
- Caveats: hand-built, time-merged membership list; codeshare rows double-count flights; the enrichment values are descriptive (no significance test performed).

## Evaluation summary (community_eval.png) and "so what"
- Disjoint modularity optimisers give high modularity (0.67), low conductance (~0.09 mean, 0.13 volume-weighted), full coverage, stable partitions (NMI ~0.9), and geographically meaningful regions; CPM gives low EQ, high conductance and partial coverage. For a hub-and-spoke network (cf. Goal 1 disassortativity - a Goal 1 result, not recomputed here) community = regional hub system, and clique-based definitions miss spokes.
- Conductance and modularity agree in ranking the methods, but conductance per community also shows which groups are well separated (Alaska 0.02, Oceania 0.10) vs leaky (Russia 0.29, Middle East/Africa 0.22).
- Limitations: modularity resolution dependence (sweep above); undirected projection; CPM coverage; alliance and continent mappings are hand-built; the Leiden/Louvain partitions are not unique (seed NMI 0.84-0.93).
- For Goal 5/6 reuse: partitions are in outputs/goal4/partition_leiden_best.csv, partition_louvain_best.csv, partitions_all_seeds.csv. If used as link-prediction features or embedding labels, compute them on the TRAIN graph only (these were computed on the full graph and would leak).
