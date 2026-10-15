# REPORT — Stable Hash Generation for Privacy-Preserving Face Identification

**Protocol:** `full`

Lab twin of Osorio-Roig et al. (TBIOM): PQ stable hashes for O(1) shortlist retrieval;
plaintext cosine re-rank stands in for FHE shortlist comparison.

## Training

- Epochs: 8
- Train images: 141610
- Identities: 7124
- PQ: M=8 K=16 buckets=954
- Elapsed (s): 1237.5

![Training](outputs/figures/full/train_loss.svg)

## Verification

| Dataset | Accuracy | EER | N |
|---------|----------|-----|---|
| calfw | 53.50% | 46.50% | 6000 |
| cplfw | 54.85% | 45.15% | 6000 |
| lfw | 63.33% | 36.67% | 6000 |

![Verification](outputs/figures/full/verification.svg)

## Identification (PQ shortlist)

- Shortlist hit rate: **14.42%**
- Rank-1 (shortlist re-rank): **0.94%**
- Rank-1 (exhaustive): **1.05%**
- Workload vs exhaustive: **8.60%**
- Avg shortlist size: 261.7 / gallery 3044

![Identification](outputs/figures/full/identification.svg)

## Notes

- Paper uses FEI/FERET + FHE; this rebuild uses CelebA + LFW and plaintext re-rank.
- See `DEVIATIONS.md`.
