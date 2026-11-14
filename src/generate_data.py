import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 5000

age = np.clip(rng.normal(34, 16, n), 0, 90).astype(int)
sex = rng.choice(["F", "M"], n)
districts = [f"D{i:02d}" for i in range(1, 13)]
district = rng.choice(districts, n, p=rng.dirichlet(np.ones(12) * 2))
household_size = np.clip(rng.poisson(4, n) + 1, 1, 12)
displaced = rng.random(n) < 0.4

logit = -3 + 0.03 * age + 0.8 * displaced
medical_need = rng.random(n) < 1 / (1 + np.exp(-logit))
income = np.clip(rng.lognormal(6.0, 0.6, n) * (1 - 0.3 * displaced), 0, None).round(0)

df = pd.DataFrame({
    "person_id": np.arange(n),
    "age": age,
    "sex": sex,
    "district": district,
    "household_size": household_size,
    "displaced": displaced,
    "income": income,
    "medical_need": medical_need,
})
df.to_csv("data/recipients.csv", index=False)

print(df.head())
print("medical_need rate:", round(df["medical_need"].mean(), 3))
print(df["district"].value_counts().sort_index())
