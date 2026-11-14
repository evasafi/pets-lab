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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def local_train(w, idx, epochs=5, lr=0.5):
    w = w.copy()
    for _ in range(epochs):
        grad = Xb[idx].T @ (sigmoid(Xb[idx] @ w) - y[idx]) / len(idx)
        w -= lr * grad
    return w

def test_auc(w):
    return roc_auc_score(y[test_idx], Xb[test_idx] @ w)

def sample_org(idx, n, rng):
    while True:  # need a few of each class so training is meaningful
        s = rng.choice(idx, n, replace=False)
        if 3 <= y[s].sum() <= n - 3:
            return s

rng = np.random.default_rng(0)
zero = np.zeros(Xb.shape[1])
rounds = 40
print("rows/org | local-only mean (std) | FedAvg mean (std)")
for n in [20, 50, 150, 500]:
    local_res, fed_res = [], []
    for _ in range(20):
        samples = {k: sample_org(i, n, rng) for k, i in org_idx.items()}
        local_res.append(np.mean([test_auc(local_train(zero, s, epochs=rounds * 5)) for s in samples.values()]))
        w = zero
        for _ in range(rounds):
            w = np.mean([local_train(w, s) for s in samples.values()], axis=0)
        fed_res.append(test_auc(w))
    print(f"{n:8d} | {np.mean(local_res):.3f} ({np.std(local_res):.3f})        | {np.mean(fed_res):.3f} ({np.std(fed_res):.3f})")
