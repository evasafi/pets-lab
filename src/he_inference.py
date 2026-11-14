import time
import numpy as np
import pandas as pd
import tenseal as ts
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

# Server side: a trained model in plaintext
model = LogisticRegression(max_iter=1000).fit(Xs[train_idx], y[train_idx])
w = model.coef_[0].tolist()
b = float(model.intercept_[0])

# Client side: encryption context (the secret key stays with the client)
ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
ctx.global_scale = 2**40
ctx.generate_galois_keys()  # needed for the dot product

idx = test_idx[:200]
plain = Xs[idx] @ np.array(w) + b

enc_scores = []
t0 = time.perf_counter()
for row in Xs[idx]:
    v = ts.ckks_vector(ctx, row.tolist())  # client encrypts features
    r = v.dot(w) + b                        # server computes on ciphertext
    enc_scores.append(r.decrypt()[0])       # client decrypts the score
elapsed = time.perf_counter() - t0

err = np.abs(np.array(enc_scores) - plain)
print(f"Predictions: {len(idx)}")
print(f"Error vs plaintext: mean {err.mean():.2e}, max {err.max():.2e}")
print(f"Time per prediction: {elapsed / len(idx) * 1000:.1f} ms")
print(f"Ciphertext size: {len(v.serialize()) / 1024:.0f} KB (raw features: {len(w) * 8} bytes)")
print(f"AUC plaintext: {roc_auc_score(y[idx], plain):.3f} | AUC encrypted: {roc_auc_score(y[idx], enc_scores):.3f}")
