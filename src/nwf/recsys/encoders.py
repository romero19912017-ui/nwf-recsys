# -*- coding: utf-8 -*-
"""Matrix factorization encoder for users and items with (z, sigma)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class _MFModel(nn.Module):
    """Neural matrix factorization: user_emb, item_emb with uncertainty heads."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        latent_dim: int,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, latent_dim)
        self.item_emb = nn.Embedding(n_items, latent_dim)
        self.user_logsigma = nn.Parameter(torch.full((n_users, latent_dim), np.log(sigma_init)))
        self.item_logsigma = nn.Parameter(torch.full((n_items, latent_dim), np.log(sigma_init)))
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(
        self, u: torch.Tensor, i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zu = self.user_emb(u)
        zi = self.item_emb(i)
        su = torch.exp(self.user_logsigma[u])
        si = torch.exp(self.item_logsigma[i])
        return zu, su, zi, si

    def get_pred(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        zu = self.user_emb(u)
        zi = self.item_emb(i)
        return (zu * zi).sum(dim=1)


class MatrixFactorEncoder:
    """Matrix factorization producing (z, sigma) for users and items."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        latent_dim: int = 64,
        sigma_init: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.device = device or "cpu"
        self._model = _MFModel(n_users, n_items, latent_dim, sigma_init).to(self.device)

    def fit(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        epochs: int = 20,
        batch_size: int = 1024,
        lr: float = 1e-2,
    ) -> "MatrixFactorEncoder":
        """Train on (user, item, rating) triples. Predicts dot product."""
        self._model.train()
        opt = torch.optim.Adam(self._model.parameters(), lr=lr)
        u = torch.LongTensor(user_ids).to(self.device)
        i = torch.LongTensor(item_ids).to(self.device)
        r = torch.FloatTensor(ratings).to(self.device)
        n = len(u)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for j in range(0, n, batch_size):
                idx = perm[j : j + batch_size]
                pred = self._model.get_pred(u[idx], i[idx])
                loss = nn.functional.mse_loss(pred, r[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._model.eval()
        return self

    def encode_user(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get (z, sigma) for user."""
        self._model.eval()
        with torch.no_grad():
            z = self._model.user_emb.weight[user_id].cpu().numpy()
            s = torch.exp(self._model.user_logsigma[user_id]).cpu().numpy()
        return z, np.maximum(s, 1e-6)

    def encode_item(self, item_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get (z, sigma) for item."""
        self._model.eval()
        with torch.no_grad():
            z = self._model.item_emb.weight[item_id].cpu().numpy()
            s = torch.exp(self._model.item_logsigma[item_id]).cpu().numpy()
        return z, np.maximum(s, 1e-6)
