"""LFW utility + closed-set PQ shortlist identification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sthash.data.celeba import CelebAIdDataset, resolve_celeba_root
from sthash.data.pairs import build_calfw, build_cplfw, build_lfw
from sthash.hashing.pq import PQIndex
from sthash.models.fr_net import FaceEmbedder


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def best_acc_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_acc, best_eer = 0.0, 1.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        best_acc = max(best_acc, float((pred == labels).mean()))
        pos, neg = labels == 1, labels == 0
        if pos.any() and neg.any():
            fpr = float((pred[neg] == 1).mean())
            fnr = float((pred[pos] == 0).mean())
            best_eer = min(best_eer, 0.5 * (fpr + fnr))
    return best_acc, best_eer


@torch.no_grad()
def eval_pairs(model, ds, device, batch_size, nw, tag) -> Dict:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    scores, labels = [], []
    model.eval()
    for a, b, y in tqdm(loader, desc=tag, leave=False):
        ea = model.embed(a.to(device))
        eb = model.embed(b.to(device))
        scores.append((ea * eb).sum(dim=1).cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    acc, eer = best_acc_eer(s, y)
    print(f"[{tag}] acc={acc:.4f} eer={eer:.4f} n={len(y)}")
    return {"accuracy": acc, "eer": eer, "n_pairs": int(len(y))}


@torch.no_grad()
def eval_identification(model, index: PQIndex, gallery_embs, gallery_ids, probe_loader, device, tag) -> Dict:
    """Hash shortlist then cosine re-rank among shortlisted gallery rows."""
    model.eval()
    # map id → indices in gallery
    id_to_idx: Dict[int, List[int]] = {}
    for i, gid in enumerate(gallery_ids):
        id_to_idx.setdefault(int(gid), []).append(i)

    hits, rank1_hash, rank1_full, shortlist_sizes = [], [], [], []
    n_gallery = len(gallery_ids)
    for images, labels in tqdm(probe_loader, desc=tag, leave=False):
        e = model.embed(images.to(device)).cpu().numpy()
        labs = labels.numpy()
        codes = index.encode(e)
        for i in range(len(labs)):
            true_id = int(labs[i])
            cand_ids = index.lookup(codes[i])
            shortlist_sizes.append(len(cand_ids) if cand_ids else n_gallery)
            hit = true_id in cand_ids
            hits.append(hit)
            # full gallery rank-1
            sims_full = gallery_embs @ e[i]
            rank1_full.append(int(gallery_ids[sims_full.argmax()]) == true_id)
            # shortlist re-rank (fallback exhaustive if empty bucket)
            if cand_ids:
                idxs = []
                for cid in cand_ids:
                    idxs.extend(id_to_idx.get(cid, []))
                if idxs:
                    sims = gallery_embs[idxs] @ e[i]
                    pred = int(gallery_ids[idxs[int(sims.argmax())]])
                else:
                    pred = int(gallery_ids[sims_full.argmax()])
            else:
                pred = int(gallery_ids[sims_full.argmax()])
            rank1_hash.append(pred == true_id)

    workload = float(np.mean(shortlist_sizes) / max(n_gallery, 1) * 100)
    out = {
        "shortlist_hit_rate": float(np.mean(hits)),
        "rank1_shortlist": float(np.mean(rank1_hash)),
        "rank1_exhaustive": float(np.mean(rank1_full)),
        "workload_pct": workload,
        "avg_shortlist": float(np.mean(shortlist_sizes)),
        "n_probe": int(len(hits)),
        "n_gallery": int(n_gallery),
        "n_buckets": len(index.code_to_ids),
    }
    print(
        f"[{tag}] hit={out['shortlist_hit_rate']:.4f} r1_sl={out['rank1_shortlist']:.4f} "
        f"r1_ex={out['rank1_exhaustive']:.4f} workload={workload:.2f}%"
    )
    return out


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    ckpt = Path(cfg["paths"]["checkpoint_dir"]) / "last.pt"
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    emb_dim = int(blob.get("emb_dim", cfg["model"].get("emb_dim", 128)))
    model = FaceEmbedder(int(blob["num_classes"]), emb_dim=emb_dim, pretrained=False).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    lut_path = Path(cfg["paths"]["checkpoint_dir"]) / "pq_lut.json"
    lut = json.loads(lut_path.read_text()) if lut_path.is_file() else {}
    index = PQIndex(
        M=int(blob["pq_M"]),
        K=int(blob["pq_K"]),
        centroids=np.asarray(blob["pq_centroids"]),
        code_to_ids={k: list(v) for k, v in lut.items()},
    )

    image_size = int(cfg["model"].get("image_size", 112))
    bs = int(cfg["eval"].get("batch_size", 128))
    nw = int(cfg["train"].get("num_workers", 4))
    max_p = cfg["eval"].get("max_lfw_pairs")
    max_p = int(max_p) if max_p is not None else None

    results: Dict = {"verification": {}, "identification": {}}
    for name, builder in (("lfw", build_lfw), ("calfw", build_calfw), ("cplfw", build_cplfw)):
        try:
            ds = builder(cfg, image_size, max_p)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            continue
        results["verification"][name] = eval_pairs(model, ds, device, bs, nw, name)

    # CelebA val closed-set gallery/probe
    root = resolve_celeba_root(cfg)
    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    val_ds = CelebAIdDataset(
        root,
        split="val",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_n,
        seed=int(cfg["train"].get("seed", 42)),
        train=False,
    )
    by_id: Dict[int, List[int]] = {}
    for i in range(len(val_ds)):
        _, sid = val_ds.samples[i]
        by_id.setdefault(sid, []).append(i)
    gal_idx, pr_idx = [], []
    for sid, idxs in by_id.items():
        if len(idxs) < 2:
            continue
        gal_idx.append(idxs[0])
        pr_idx.extend(idxs[1:])
    max_gal = cfg["eval"].get("max_gallery")
    if max_gal is not None:
        gal_idx = gal_idx[: int(max_gal)]
        keep = {val_ds.samples[i][1] for i in gal_idx}
        pr_idx = [i for i in pr_idx if val_ds.samples[i][1] in keep]
        cap = int(max_gal) * 3
        if len(pr_idx) > cap:
            pr_idx = pr_idx[:cap]

    class _Sub(torch.utils.data.Dataset):
        def __init__(self, base, indices):
            self.base, self.indices = base, indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            return self.base[self.indices[i]]

    g_loader = DataLoader(_Sub(val_ds, gal_idx), batch_size=bs, shuffle=False, num_workers=nw)
    p_loader = DataLoader(_Sub(val_ds, pr_idx), batch_size=bs, shuffle=False, num_workers=nw)

    g_embs, g_labs = [], []
    with torch.no_grad():
        for images, labels in tqdm(g_loader, desc="gallery", leave=False):
            g_embs.append(model.embed(images.to(device)).detach().cpu().numpy())
            g_labs.append(labels.numpy())
    gallery_embs = np.concatenate(g_embs)
    gallery_ids = np.concatenate(g_labs)

    # Rebuild LUT from this gallery (eval-consistent) using saved centroids
    codes = index.encode(gallery_embs)
    lut2: Dict[str, List[int]] = {}
    for i, code in enumerate(codes):
        key = PQIndex.code_key(code)
        lut2.setdefault(key, []).append(int(gallery_ids[i]))
    index.code_to_ids = lut2

    results["identification"]["celeba_val"] = eval_identification(
        model, index, gallery_embs, gallery_ids, p_loader, device, "id_pq"
    )

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
