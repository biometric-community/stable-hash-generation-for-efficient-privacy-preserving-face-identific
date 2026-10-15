# Stable Hash Generation for Efficient Privacy-Preserving Face Identification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of **PQ stable-hash indexing** for face identification shortlist retrieval (FHE re-rank omitted; plaintext cosine stand-in).

**Repository:** https://github.com/biometric-community/stable-hash-generation-for-efficient-privacy-preserving-face-identific

**Paper:** [10.1109/TBIOM.2021.3100639](https://doi.org/10.1109/TBIOM.2021.3100639)

## Setup

```bash
bash scripts/setup_env.sh
```

## Train / Eval / Report

```bash
bash scripts/smoke.sh
bash scripts/train_full.sh
```

See [REPORT.md](./REPORT.md) and [DEVIATIONS.md](./DEVIATIONS.md).

## License

[CC BY 4.0](./LICENSE)
