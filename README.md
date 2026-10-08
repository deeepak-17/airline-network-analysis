# Global Airline Network Analysis

Social Network Analysis course project on the OpenFlights airline network. Airports are nodes and flight routes are edges. The project covers eight goals: structure, growth models, link analysis, community detection, link prediction, graph embeddings, cascade resilience and anomaly detection.

Each goal has its code in `goalN/` and its tables, plots and write-up in `outputs/goalN/` (start with `findings.md` in each).

## Data

| File | Content |
| --- | --- |
| `routes.dat` | 67,663 airline routes (source, destination, airline, codeshare, equipment) |
| `airports.dat` | Airport name, city, country, coordinates |
| `airlines.dat` | Airline names and codes |
| `openflights_routes.gexf` | Pre-aggregated graph (3,257 nodes, 37,041 edges); inspected but not used for the results |

The directed graph built from `routes.dat` has 3,425 airports and 37,594 edges. All goals build it through `common/load_graph.py`. Most goals use the undirected projection (19,256 edges).

## Goals and key results

| Goal | Folder | Headline result |
| --- | --- | --- |
| 1. Network structure | `goal1/` | Core-periphery small world. Degree assortativity r = -0.0065 (about neutral, not negative). Average path length 4.10, transitivity 0.248, reciprocity 97.6%, max k-core 31 |
| 2. Growth models | `goal2/` | No classic model fits. A truncated power law beats pure power law, lognormal and exponential, so the network is not cleanly scale-free |
| 3. Link analysis | `goal3/` | PageRank tracks degree (Spearman 0.85). Gains come from the out-degree of in-neighbours, not neighbour quality |
| 4. Communities | `goal4/` | Leiden/Louvain modularity about 0.66 with strong geographic purity (0.88). Alliance NMI only 0.10. Clique percolation performs far worse |
| 5. Link prediction | `goal5_6/` | Heuristics reach AUC about 0.99 on random negatives. Preferential attachment falls from 0.978 to 0.700 on degree-matched negatives |
| 6. Graph embeddings | `goal5_6/` | node2vec predicts continent (macro-F1 0.966) but does not beat degree for hub status. Embeddings win link prediction only on degree-matched negatives |
| 7. Cascades and resilience | `goal7_8/` | Random failure breaks the network at about 37% of airports removed. Targeted attack needs 4% to 8% |
| 8. Anomalous airports | `goal7_8/` | Gateway-like airports (Anchorage, Greenland) stand out, but detectors disagree, so treat the list as candidates |

Several results contradict common expectations (for example near-zero assortativity). `findings.md` in each outputs folder states which hypotheses held and which did not, along with caveats.

## Repository layout

```
common/            shared graph loader
goal1/ ... goal7_8/   scripts, one folder per goal group
outputs/goalN/     CSV tables, PNG plots, findings.md
requirements.txt   pinned Python packages
```

## Reproducing

Python 3.14 was used. Run everything from the repository root.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

| Goal | Command | Approx. time |
| --- | --- | --- |
| 1 | `.venv/bin/python goal1/run_goal1.py` | 2 min |
| 2 | `.venv/bin/python goal2/run_goal2.py` | 7 min |
| 3 | `.venv/bin/python goal3/run_goal3.py` | 10 s (also regenerates `outputs/goal3/simrank.npy`, 45 MB, not stored in git) |
| 4 | `goal4/run_disjoint.py`, `run_overlap.py`, `verify_cpm.py`, `run_interpret.py` | not recorded |
| 5 | `.venv/bin/python goal5_6/goal5_link_prediction.py` | 4 min |
| 6 | `goal5_6/goal6_embeddings.py` (25 min), then `goal6_node_tasks.py` (2 min), then `goal6_link_prediction.py` (5 min) | about 32 min |
| 7 | `goal7_8/goal7a_resilience.py`, `goal7b_cascade.py` | not recorded |
| 8 | `goal7_8/goal8_anomaly.py`, then `goal8b_degree_controlled.py` | not recorded |

Random seeds are fixed in every script. Goal 5 must run before Goal 6, because Goal 6 reuses the saved train/test edge split.

## Notes and limitations

- The network is one static snapshot with no capacity, frequency or demand data. Mechanisms proposed in the write-ups (capacity limits, aging in preferential attachment, gateway roles) are hypotheses, not tested findings.
- node2vec is implemented from scratch in numpy because gensim does not build on Python 3.14. It may be under-trained.
- SimRank uses the standard sparse-matrix approximation. Betweenness (Goal 3) and global efficiency (Goal 7) use sampled source nodes.
- The airline alliance membership list in Goal 4 is hand-built.
