import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/recipients.csv")
X = pd.DataFrame({
    "age": df["age"],
    "female": (df["sex"] == "F").astype(float),
    "household": df["household_size"],
    "displaced": df["displaced"].astype(float),
    "log_income": np.log1p(df["income"]),
})
y = df["medical_need"].astype(float).values
train_idx, _ = train_test_split(np.arange(len(df)), test_size=0.3, random_state=0, stratify=y)
mu, sd = X.iloc[train_idx].mean(), X.iloc[train_idx].std()
Xb = np.hstack([((X - mu) / sd).values, np.ones((len(X), 1))])

C = 1.0
ROUNDS = 200

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def private_update(w, idx, sigma, rng):
    g = (sigmoid(Xb[idx] @ w) - y[idx])[:, None] * Xb[idx]
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    g = g * np.minimum(1.0, C / (norms + 1e-12))
    return (g.sum(axis=0) + rng.normal(0, sigma * C, size=w.shape)) / len(idx)

def alignment(g, feats):
    return np.abs(feats @ g) / (np.linalg.norm(feats, axis=1) * np.linalg.norm(g) + 1e-12)

# Fixed public trajectory of global models (non-private training path)
w = np.zeros(Xb.shape[1])
W = []
for _ in range(ROUNDS):
    W.append(w.copy())
    grad = Xb[train_idx].T @ (sigmoid(Xb[train_idx] @ w) - y[train_idx]) / len(train_idx)
    w = w - 0.5 * grad

rng = np.random.default_rng(0)
pool = rng.choice(train_idx, 400, replace=False)
checkpoints = [1, 10, 50, 200]

print("sigma | batch | attack AUC after R rounds: " + " / ".join(f"R={r}" for r in checkpoints))
for sigma in [1, 4]:
    for B in [1, 5, 20]:
        res = {r: [] for r in checkpoints}
        for _ in range(150):
            members = rng.choice(pool, B, replace=False)
            labels = np.isin(pool, members)
            U = np.array([private_update(W[t], members, sigma, rng) for t in range(ROUNDS)])
            for r in checkpoints:
                scores = alignment(U[:r].mean(axis=0), Xb[pool])
                res[r].append(roc_auc_score(labels, scores))
        cells = " / ".join(f"{np.mean(res[r]):.3f}" for r in checkpoints)
        print(f"{sigma:5} | {B:5d} | {cells}")
