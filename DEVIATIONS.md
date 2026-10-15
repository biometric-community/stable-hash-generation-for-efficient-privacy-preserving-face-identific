# DEVIATIONS — Stable Hash PQ Face ID

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Datasets | FEI / FERET / LFW | CelebA + LFW/CALFW/CPLFW | Local public data |
| D2 | Protection | Fully homomorphic encryption | Plaintext cosine re-rank | No FHE stack in lab twin |
| D3 | Embeddings | Commercial / paper FR | CelebA-trained ResNet-18 | Local train |
| D4 | Stability | Multi-sample majority PQ | Single-pass PQ codes | Runtime simplification |
