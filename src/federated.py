import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
Xs = ((X - mu) / sd).values
Xb = np.hstack([Xs, np.ones((len(Xs), 1))])  # bias column

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

def test_auc(scores):
    return roc_auc_score(y[test_idx], scores)

# 1. Centralized (pooled data)
central = LogisticRegression(max_iter=1000).fit(Xs[train_idx], y[train_idx])
print(f"Centralized AUC:  {test_auc(central.decision_function(Xs[test_idx])):.3f}")

# 2. Each org alone
for k, idx in org_idx.items():
    m = LogisticRegression(max_iter=1000).fit(Xs[idx], y[idx])
    print(f"Local only, org {k} ({len(idx)} rows): AUC {test_auc(m.decision_function(Xs[test_idx])):.3f}")

# 3. FedAvg
w = np.zeros(Xb.shape[1])
rounds = 40
sizes = np.array([len(i) for i in org_idx.values()])
for _ in range(rounds):
    updates = [local_train(w, idx) for idx in org_idx.values()]
    w = np.average(updates, axis=0, weights=sizes)
print(f"FedAvg AUC ({rounds} rounds): {test_auc(Xb[test_idx] @ w):.3f}")
print(f"Sent per org per round: {len(w)} numbers (weights), no raw rows")
