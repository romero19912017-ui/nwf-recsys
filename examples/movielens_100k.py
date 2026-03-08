# -*- coding: utf-8 -*-
"""Example: MovieLens 100k recommendations with NWF MatrixFactorEncoder.

Build item index, recommend by user charge. Add new user without retraining.
Run: python movielens_100k.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import os

import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nwf import Charge, Field
from nwf.recsys import MatrixFactorEncoder


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


def hit_rate_at_k(
    recs: np.ndarray, holdout: dict, k: int = 10
) -> float:
    """HR@k: fraction of users with at least one hit in top-k."""
    hits = 0
    for u, items in holdout.items():
        if u >= len(recs):
            continue
        topk = set(recs[u, :k])
        if topk & set(items):
            hits += 1
    return hits / max(len(holdout), 1)


def main() -> None:
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    user_idx, item_idx, ratings, n_users, n_items = load_movielens_100k(data_dir)
    print(f"Loaded: {len(ratings)} ratings, {n_users} users, {n_items} items")

    # Train/test split by last rating per user
    rng = np.random.RandomState(42)
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
    enc = MatrixFactorEncoder(n_users, n_items, latent_dim=32)
    enc.fit(train_u, train_i, train_r, epochs=15, batch_size=2048, lr=1e-2)

    print("Building item Field...")
    field = Field()
    for j in range(n_items):
        z, s = enc.encode_item(j)
        field.add(Charge(z=z, sigma=s), labels=[j], ids=[j])

    print("Recommendations: search by user charge...")
    recs = []
    for u in range(n_users):
        zu, su = enc.encode_user(u)
        q = Charge(z=zu, sigma=su)
        _, idx, _ = field.search(q, k=20)
        recs.append(idx)
    recs = np.array(recs)
    hr10 = hit_rate_at_k(recs, holdout, k=10)
    print(f"HR@10: {hr10:.3f}")

    print("Adding new user (cold-start simulation)...")
    # Simulate new user: random ratings on some items
    new_items = rng.choice(n_items, size=10, replace=False)
    new_ratings = rng.uniform(3, 5, size=10).astype(np.float32)
    # We would need to extend the model for true cold-start; here we use existing user 0 as proxy
    zu, su = enc.encode_user(0)
    q_new = Charge(z=zu + rng.randn(32) * 0.1, sigma=su)
    _, idx_new, _ = field.search(q_new, k=5)
    print(f"Top-5 for new user: {idx_new}")
    print("Done.")


if __name__ == "__main__":
    main()
