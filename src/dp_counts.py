import numpy as np
import pandas as pd

df = pd.read_csv("data/recipients.csv")
rng = np.random.default_rng(0)

sizes = df.groupby("district").size()
true = df[df["medical_need"]].groupby("district").size().reindex(sizes.index, fill_value=0)

# One example release at epsilon = 1
noisy = (true + rng.laplace(0, 1 / 1.0, len(true))).round().clip(lower=0).astype(int)
print(pd.DataFrame({"people": sizes, "true": true, "noisy_eps1": noisy}))

# Relative error across many trials, for several epsilons
trials = 500
out = {}
for eps in [0.1, 0.5, 1.0, 5.0]:
    noise = rng.laplace(0, 1 / eps, size=(trials, len(true)))
    mean_abs_err = np.abs(noise).mean(axis=0)
    out[f"eps={eps}"] = (mean_abs_err / true.values).round(2)

table = pd.DataFrame(out, index=true.index)
table.insert(0, "true_count", true)
print()
print("Relative error (mean abs error / true count):")
print(table)
