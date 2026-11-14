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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

rng = np.random.default_rng(0)
pool = rng.choice(train_idx, 400, replace=False)  # attacker's candidate pool
w = np.zeros(Xb.shape[1])                          # first round, fresh model

def update(idx):
    return Xb[idx].T @ (sigmoid(Xb[idx] @ w) - y[idx]) / len(idx)

def alignment(g, feats):
    return np.abs(feats @ g) / (np.linalg.norm(feats, axis=1) * np.linalg.norm(g) + 1e-12)

print("batch | membership AUC | top-1 hit rate | chance top-1")
for B in [1, 2, 5, 20, 100]:
    aucs, hits = [], []
    for _ in range(200):
        members = rng.choice(pool, B, replace=False)
        scores = alignment(update(members), Xb[pool])
        labels = np.isin(pool, members)
        aucs.append(roc_auc_score(labels, scores))
        hits.append(labels[np.argmax(scores)])
    print(f"{B:5d} | {np.mean(aucs):14.3f} | {np.mean(hits):14.1%} | {B / len(pool):.1%}")
