"""Figures + REPORT.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sthash.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(hist, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([h["epoch"] for h in hist], [h["loss"] for h in hist], color=MULTI_SERIES_COLORS[0], linewidth=LINEWIDTH_MAIN)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE loss")
    ax.set_title("FR embedding training (CelebA)")
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_verify(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ver = eval_res.get("verification", {})
    keys = list(ver.keys())
    vals = [ver[k]["accuracy"] * 100 for k in keys]
    x = np.arange(len(keys))
    ax.bar(x, vals, color=MULTI_SERIES_COLORS[0], edgecolor="#002080")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Verification utility")
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_id(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ident = eval_res.get("identification", {}).get("celeba_val", {})
    names = ["shortlist hit", "rank1 shortlist", "rank1 exhaustive", "workload %"]
    vals = [
        ident.get("shortlist_hit_rate", 0) * 100,
        ident.get("rank1_shortlist", 0) * 100,
        ident.get("rank1_exhaustive", 0) * 100,
        ident.get("workload_pct", 0),
    ]
    x = np.arange(len(names))
    ax.bar(x, vals, color=MULTI_SERIES_COLORS[1], edgecolor="#002080")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("%")
    ax.set_title("PQ shortlist identification (CelebA val)")
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, fig_dir: Path, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    pq = train_s.get("pq", {})
    ident = eval_res.get("identification", {}).get("celeba_val", {})
    lines = [
        "# REPORT — Stable Hash Generation for Privacy-Preserving Face Identification",
        "",
        f"**Protocol:** `{protocol}`",
        "",
        "Lab twin of Osorio-Roig et al. (TBIOM): PQ stable hashes for O(1) shortlist retrieval;",
        "plaintext cosine re-rank stands in for FHE shortlist comparison.",
        "",
        "## Training",
        "",
        f"- Epochs: {train_s.get('epochs')}",
        f"- Train images: {train_s.get('n_train')}",
        f"- Identities: {train_s.get('num_classes')}",
        f"- PQ: M={pq.get('M')} K={pq.get('K')} buckets={pq.get('n_buckets')}",
        f"- Elapsed (s): {train_s.get('elapsed_sec', 0):.1f}",
        "",
        f"![Training]({fig_dir.as_posix()}/train_loss.svg)",
        "",
        "## Verification",
        "",
        "| Dataset | Accuracy | EER | N |",
        "|---------|----------|-----|---|",
    ]
    for k, v in sorted(eval_res.get("verification", {}).items()):
        lines.append(f"| {k} | {v['accuracy']*100:.2f}% | {v['eer']*100:.2f}% | {v['n_pairs']} |")
    lines += [
        "",
        f"![Verification]({fig_dir.as_posix()}/verification.svg)",
        "",
        "## Identification (PQ shortlist)",
        "",
        f"- Shortlist hit rate: **{ident.get('shortlist_hit_rate', 0)*100:.2f}%**",
        f"- Rank-1 (shortlist re-rank): **{ident.get('rank1_shortlist', 0)*100:.2f}%**",
        f"- Rank-1 (exhaustive): **{ident.get('rank1_exhaustive', 0)*100:.2f}%**",
        f"- Workload vs exhaustive: **{ident.get('workload_pct', 0):.2f}%**",
        f"- Avg shortlist size: {ident.get('avg_shortlist', 0):.1f} / gallery {ident.get('n_gallery', 0)}",
        "",
        f"![Identification]({fig_dir.as_posix()}/identification.svg)",
        "",
        "## Notes",
        "",
        "- Paper uses FEI/FERET + FHE; this rebuild uses CelebA + LFW and plaintext re-rank.",
        "- See `DEVIATIONS.md`.",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def generate(cfg: dict) -> None:
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval.json").read_text())
    plot_train(train_s["history"], fig_dir / "train_loss.svg")
    plot_verify(eval_res, fig_dir / "verification.svg")
    plot_id(eval_res, fig_dir / "identification.svg")
    write_report(cfg, train_s, eval_res, fig_dir, Path("REPORT.md"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    generate(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
