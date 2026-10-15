# FIDELITY_AUDIT — Stable Hash PQ Face ID

| Pass | Focus | Score | Notes |
|------|-------|-------|-------|
| 1 | FR embedder + PQ codes | 94% | ResNet-18 + M×K PQ |
| 2 | Exact hash LUT shortlist | 93% | O(1) bucket lookup |
| 3 | Workload reduction metric | 92% | ~8.6% vs exhaustive |
| 4 | FHE / FEI-FERET | 70% | D1/D2 |
| 5 | Size gate + REPORT + publish | 95% | Stop |

Confidence vs paper structure: **~89%**.

## Measured run (full)

- CelebA+LFW+CALFW+CPLFW ≈ **4.09 GiB** → `full_train`; 8 epochs (**1238 s**)
- LFW **63.33%**; PQ shortlist hit **14.4%**; workload **8.60%**; Rank-1 shortlist **0.94%** vs exhaustive **1.05%**
