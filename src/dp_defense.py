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
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.3, random_state=0, stratify=y)
mu, sd = X.iloc[train_idx].mean(), X.iloc[train_idx].std()
Xb = np.hstack([((X - mu) / sd).values, np.ones((len(X), 1))])

orgs = {"A": ["D01", "D02", "D03", "D04"],
        "B": ["D05", "D06", "D07", "D08"],
        "C": ["D09", "D10", "D11", "D12"]}
org_idx = {k: train_idx[df["district"].iloc[train_idx].isin(v).values] for k, v in orgs.items()}

C = 1.0  # clipping norm

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def private_update(w, idx, sigma, rng):
    g = (sigmoid(Xb[idx] @ w) - y[idx])[:, None] * Xb[idx]        # per-person gradients
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    g = g * np.minimum(1.0, C / (norms + 1e-12))                   # clip each person
    total = g.sum(axis=0) + rng.normal(0, sigma * C, size=w.shape) # sum + Gaussian noise
    return total / len(idx)

def alignment(g, feats):
    return np.abs(feats @ g) / (np.linalg.norm(feats, axis=1) * np.linalg.norm(g) + 1e-12)

def sample_org(idx, n, rng):
    while True:
        s = rng.choice(idx, n, replace=False)
        if 3 <= y[s].sum() <= n - 3:
            return s

rng = np.random.default_rng(0)
pool = rng.choice(train_idx, 400, replace=False)
w0 = np.zeros(Xb.shape[1])
draws = [[sample_org(i, 20, rng) for i in org_idx.values()] for _ in range(20)]

print("sigma | attack AUC B=1 | B=5 | B=20 | FL test AUC, 20 rows/org (std over 20 draws)")
for sigma in [0, 0.5, 1, 2, 4]:
    attack = []
    for B in [1, 5, 20]:
        aucs = []
        for _ in range(150):
            members = rng.choice(pool, B, replace=False)
            scores = alignment(private_update(w0, members, sigma, rng), Xb[pool])
            aucs.append(roc_auc_score(np.isin(pool, members), scores))
        attack.append(np.mean(aucs))

    fl = []
    for samples in draws:
        w = np.zeros(Xb.shape[1])
        for _ in range(200):
            ups = [private_update(w, s, sigma, rng) for s in samples]
            w = w - 0.5 * np.mean(ups, axis=0)
        fl.append(roc_auc_score(y[test_idx], Xb[test_idx] @ w))
    print(f"{sigma:5} | {attack[0]:14.3f} | {attack[1]:.3f} | {attack[2]:.3f} | {np.mean(fl):.3f} ({np.std(fl):.3f})")
