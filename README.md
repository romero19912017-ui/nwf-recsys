# nwf-recsys

[![PyPI version](https://badge.fury.io/py/nwf-recsys.svg)](https://pypi.org/project/nwf-recsys/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## NWF for Recommendation Systems

`nwf-recsys` provides matrix factorization encoders that produce **semantic charges** `(z, sigma)` for users and items. Recommendations are made by searching the item index with the user charge; supports incremental addition of users and items.

### Features

- **MatrixFactorEncoder** — neural matrix factorization with uncertainty for users and items
- **Output (z, sigma)** — compatible with `nwf-core` Field and Mahalanobis search
- **Recommendations** — build item Field, search by user charge for top-k items
- **Incremental** — add new items to index without retraining; cold-start via content (planned)
- **Metrics** — HR@k, NDCG@k

---

## Installation

```bash
pip install nwf-core nwf-recsys
```

Requires: `nwf-core>=0.2.3`, `torch`, `numpy`, `scikit-learn`.

---

## Quick Start

```python
from nwf import Charge, Field
from nwf.recsys import MatrixFactorEncoder
import numpy as np

enc = MatrixFactorEncoder(n_users=1000, n_items=1700, latent_dim=64)
enc.fit(user_ids, item_ids, ratings, epochs=20)

# Build item index
field = Field()
for j in range(n_items):
    z, s = enc.encode_item(j)
    field.add(Charge(z=z, sigma=s), labels=[j], ids=[j])

# Recommend for user
z_u, s_u = enc.encode_user(user_id)
q = Charge(z=z_u, sigma=s_u)
distances, indices, labels = field.search(q, k=10)
```

---

## API

### MatrixFactorEncoder

| Parameter | Description |
|-----------|-------------|
| `n_users` | Number of users (0..n_users-1) |
| `n_items` | Number of items (0..n_items-1) |
| `latent_dim` | Embedding dimension |
| `sigma_init` | Initial sigma (default 0.5) |

| Method | Description |
|--------|-------------|
| `fit(user_ids, item_ids, ratings, epochs, batch_size, lr)` | Train on rating triples |
| `encode_user(user_id)` | Returns `(z, sigma)` for user |
| `encode_item(item_id)` | Returns `(z, sigma)` for item |

---

## Example: MovieLens 100k

```bash
cd examples && python movielens_100k.py
```

- Downloads MovieLens 100k automatically
- Trains MatrixFactorEncoder (15 epochs)
- Builds item Field, computes HR@10
- Simulates cold-start for new user

---

## License

MIT

---

# nwf-recsys (Русский)

## NWF для рекомендательных систем

`nwf-recsys` предоставляет матричную факторизацию с выходом **семантических зарядов** `(z, sigma)` для пользователей и товаров. Рекомендации: поиск в индексе товаров по заряду пользователя.

### Компоненты

- **MatrixFactorEncoder** — нейросетевая матричная факторизация с неопределённостью
- **Выход (z, sigma)** — совместим с Field и поиском по Махаланобису
- **Рекомендации** — поиск top-k товаров по заряду пользователя
- **Инкрементальность** — добавление новых товаров в индекс без переобучения

### Установка

```bash
pip install nwf-core nwf-recsys
```

### Пример

```python
from nwf.recsys import MatrixFactorEncoder
from nwf import Charge, Field

enc = MatrixFactorEncoder(n_users=943, n_items=1682, latent_dim=32)
enc.fit(user_ids, item_ids, ratings)
z_u, s_u = enc.encode_user(user_id)
field.search(Charge(z=z_u, sigma=s_u), k=10)
```

### Лицензия

MIT
