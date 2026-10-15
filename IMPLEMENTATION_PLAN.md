# IMPLEMENTATION_PLAN — Stable Hash PQ Face ID

## Paper

PQ stable hashes index protected templates for O(1) shortlist; FHE compares only shortlisted refs.

## Lab twin

| Component | Choice |
|-----------|--------|
| Embedder | ResNet-18 CE on CelebA |
| Hash | Product quantization (M×K) |
| Index | Exact code → identity LUT |
| Match | Plaintext cosine re-rank (FHE stand-in) |
| Metrics | LFW utility; shortlist hit; workload %; Rank-1 |
