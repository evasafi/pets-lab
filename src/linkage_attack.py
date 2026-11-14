import pandas as pd

df = pd.read_csv("data/recipients.csv")
qi = ["age", "sex", "district", "household_size"]

# Step 1: the naive release, with the ID column dropped
released = df.drop(columns=["person_id"])

# Step 2: how unique are people on the quasi-identifiers?
group_size = released.groupby(qi)["age"].transform("size")
print("Smallest group size (k-anonymity of the release):", group_size.min())
print("Records unique on quasi-identifiers:", (group_size == 1).sum(),
      f"({(group_size == 1).mean():.1%})")

# Step 3: attacker has a named side list for 500 people
aux = df.sample(500, random_state=1)[["person_id"] + qi].copy()
aux["name"] = "name_" + aux["person_id"].astype(str)
aux = aux.drop(columns=["person_id"])

# Step 4: join the side list to the released data
merged = aux.merge(released, on=qi)
per_person = merged.groupby("name").agg(
    n_matches=("medical_need", "size"),
    n_values=("medical_need", "nunique"),
)

reidentified = (per_person["n_matches"] == 1).sum()
exposed = (per_person["n_values"] == 1).sum()
print(f"Re-identified exactly: {reidentified} of 500 ({reidentified / 500:.1%})")
print(f"Medical status exposed (unique or all matches agree): {exposed} of 500 ({exposed / 500:.1%})")
