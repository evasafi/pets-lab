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

C = 1.0
ROUNDS = 200
LR = 0.5

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def private_update(w, idx, sigma, rng):
    g = (sigmoid(Xb[idx] @ w) - y[idx])[:, None] * Xb[idx]
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    g = g * np.minimum(1.0, C / (norms + 1e-12))
    return (g.sum(axis=0) + rng.normal(0, sigma * C, size=w.shape)) / len(idx)

def local_gd(idx):
    w = np.zeros(Xb.shape[1])
    for _ in range(ROUNDS):
        w = w - LR * Xb[idx].T @ (sigmoid(Xb[idx] @ w) - y[idx]) / len(idx)
    return w

def dp_fl(samples, sigma, rng):
    w = np.zeros(Xb.shape[1])
    for _ in range(ROUNDS):
        w = w - LR * np.mean([private_update(w, s, sigma, rng) for s in samples], axis=0)
    return w

def auc(w):
    return roc_auc_score(y[test_idx], Xb[test_idx] @ w)

def sample_org(idx, n, rng):
    while True:
        s = rng.choice(idx, n, replace=False)
        if 3 <= y[s].sum() <= n - 3:
            return s

rng = np.random.default_rng(0)
cols = ["local-only", "DP-FL sigma 0", "DP-FL sigma 4", "DP-FL sigma 56.6"]
print("rows/org | " + " | ".join(cols) + "   (mean AUC, std in brackets, 20 draws)")
print("epsilon over 200 rounds: sigma 0 = none, sigma 4 = about 21-22, sigma 56.6 = about 1")
for n in [20, 150, 500]:
    res = {c: [] for c in cols}
    for _ in range(20):
        samples = [sample_org(i, n, rng) for i in org_idx.values()]
        res["local-only"].append(np.mean([auc(local_gd(s)) for s in samples]))
        res["DP-FL sigma 0"].append(auc(dp_fl(samples, 0.0, rng)))
        res["DP-FL sigma 4"].append(auc(dp_fl(samples, 4.0, rng)))
        res["DP-FL sigma 56.6"].append(auc(dp_fl(samples, 56.6, rng)))
    print(f"{n:8d} | " + " | ".join(f"{np.mean(res[c]):.3f} ({np.std(res[c]):.3f})" for c in cols))
