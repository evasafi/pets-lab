import secrets
import numpy as np
import pandas as pd

P = 2**61 - 1
SCALE = 1000  # fixed-point: store noisy counts as integers

def share(value, n):
    shares = [secrets.randbelow(P) for _ in range(n - 1)]
    shares.append((value - sum(shares)) % P)
    return shares

df = pd.read_csv("data/recipients.csv")
orgs = {"A": ["D01", "D02", "D03", "D04"],
        "B": ["D05", "D06", "D07", "D08"],
        "C": ["D09", "D10", "D11", "D12"]}
names = list(orgs)
n = len(names)
eps = 1.0
rng = np.random.default_rng(7)

local = {k: int((df["district"].isin(v) & df["medical_need"]).sum()) for k, v in orgs.items()}
print("Collusion without noise: B and C recover A =",
      sum(local.values()) - local["B"] - local["C"], "(true:", local["A"], ")")

# Each org adds a noise piece; the n pieces sum to one Laplace(1/eps) draw
piece = {k: rng.gamma(1 / n, 1 / eps) - rng.gamma(1 / n, 1 / eps) for k in names}
noisy_local = {k: local[k] + piece[k] for k in names}

# Secure sum of the noisy values (fixed-point integers)
enc = {k: int(round(noisy_local[k] * SCALE)) % P for k in names}
all_shares = {k: share(enc[k], n) for k in names}
partial = [sum(all_shares[k][j] for k in names) % P for j in range(n)]
total = sum(partial) % P
if total > P // 2:
    total -= P
print("Noisy secure total:", round(total / SCALE, 2), "| true total:", sum(local.values()))

# How much does noise protect A, with and without collusion?
T = 20000
pieces = rng.gamma(1 / n, 1 / eps, (T, n)) - rng.gamma(1 / n, 1 / eps, (T, n))
print(f"Outsider view, mean abs error on the total: {np.abs(pieces.sum(axis=1)).mean():.2f}")
print(f"B and C colluding, mean abs error on A's count: {np.abs(pieces[:, 0]).mean():.2f}")
