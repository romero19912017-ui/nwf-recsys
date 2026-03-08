# -*- coding: utf-8 -*-
"""MovieLens 100k: recommendations with NWF MatrixFactorEncoder.

Build item Field, recommend by user charge. HR@k, cold-start simulation.
Run: python movielens_100k.py [--epochs 15] [--save results/recsys.png]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import numpy as np

from nwf import Charge, Field
from nwf.recsys import MatrixFactorEncoder

if "--save" in sys.argv or os.environ.get("MPLBACKEND"):
    import matplotlib
    matplotlib.use("Agg")


def load_movielens_100k(data_dir: Path) -> tuple:
    """Download and load MovieLens 100k. Returns (user_idx, item_idx, ratings, n_users, n_items)."""
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = data_dir / "ml-100k.zip"
    if not zip_path.exists():
        print("Downloading MovieLens 100k...")
        urlretrieve(url, zip_path)
    if not (data_dir / "ml-100k" / "u.data").exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
    data_path = data_dir / "ml-100k" / "u.data"
    data = np.loadtxt(data_path, dtype=np.int64, usecols=(0, 1, 2))
    user_raw, item_raw, ratings = data[:, 0], data[:, 1], data[:, 2].astype(np.float32)
    user_u = np.unique(user_raw)
    item_u = np.unique(item_raw)
    u2i = {u: i for i, u in enumerate(user_u)}
    i2j = {m: j for j, m in enumerate(item_u)}
    user_idx = np.array([u2i[u] for u in user_raw])
    item_idx = np.array([i2j[m] for m in item_raw])
    return user_idx, item_idx, ratings, len(user_u), len(item_u)


def hit_rate_at_k(recs: np.ndarray, holdout: dict, k: int = 10) -> float:
    """HR@k: fraction of users with at least one hit in top-k."""
    hits = 0
    for u, items in holdout.items():
        if u >= len(recs):
            continue
        topk = set(recs[u, :k].astype(int))
        if topk & set(items):
            hits += 1
    return hits / max(len(holdout), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="MovieLens 100k with NWF")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    user_idx, item_idx, ratings, n_users, n_items = load_movielens_100k(data_dir)
    print(f"Loaded: {len(ratings)} ratings, {n_users} users, {n_items} items")

    rng = np.random.RandomState(args.seed)
    holdout = {}
    train_u, train_i, train_r = [], [], []
    for u in range(n_users):
        mask = user_idx == u
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        if len(idx) > 1:
            holdout[u] = [item_idx[idx[-1]]]
            train_u.extend(user_idx[idx[:-1]])
            train_i.extend(item_idx[idx[:-1]])
            train_r.extend(ratings[idx[:-1]])
    train_u = np.array(train_u)
    train_i = np.array(train_i)
    train_r = np.array(train_r)

    print("Training MatrixFactorEncoder...")
    enc = MatrixFactorEncoder(n_users, n_items, latent_dim=args.latent_dim)
    enc.fit(train_u, train_i, train_r, epochs=args.epochs, batch_size=args.batch_size, lr=1e-2)

    print("Building item Field...")
    field = Field()
    for j in range(n_items):
        z, s = enc.encode_item(j)
        field.add(Charge(z=z, sigma=s), labels=[j], ids=[j])

    print("Computing HR@k for k=1,5,10,20...")
    recs = []
    for u in range(n_users):
        zu, su = enc.encode_user(u)
        q = Charge(z=zu, sigma=su)
        _, _, labs = field.search(q, k=args.top_k)
        item_ids = np.array(labs[0]).astype(int)
        recs.append(item_ids)
    recs = np.array(recs)
    ks = [1, 5, 10, 20]
    hr_values = [hit_rate_at_k(recs, holdout, k=k) for k in ks]
    for k, hr in zip(ks, hr_values):
        print(f"HR@{k}: {hr:.3f}")
    hr10 = hr_values[2]

    print("Cold-start simulation: new user...")
    zu, su = enc.encode_user(0)
    q_new = Charge(z=zu + rng.randn(args.latent_dim) * 0.1, sigma=su)
    _, _, labs_new = field.search(q_new, k=5)
    top5 = np.array(labs_new[0]).astype(int)
    print(f"Top-5 for new user: {top5}")

    if args.save:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([str(k) for k in ks], hr_values, color="C0")
        ax.set_xlabel("k")
        ax.set_ylabel("HR@k")
        ax.set_title("MovieLens 100k: Hit Rate at k")
        ax.set_ylim(0, 1)
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {args.save}")
    print("Done.")


if __name__ == "__main__":
    main()
