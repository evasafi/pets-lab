import numpy as np
import pandas as pd

df = pd.read_csv("data/recipients.csv")
base_qi = ["sex", "district", "household_size"]

print("age band | min k | unique records | re-identified | positives exposed with certainty")
for b in [1, 5, 10]:
    d = df.copy()
    d["age_band"] = d["age"] // b * b
    qi = ["age_band"] + base_qi
    released = d.drop(columns=["person_id"])
    group_size = released.groupby(qi)["age_band"].transform("size")

    aux = d.sample(500, random_state=1)[qi + ["medical_need"]].rename(columns={"medical_need": "truth"})
    aux["name"] = np.arange(500)

    merged = aux.drop(columns="truth").merge(released[qi + ["medical_need"]], on=qi)
    per = merged.groupby("name").agg(n=("medical_need", "size"), n_pos=("medical_need", "sum"))
    per = per.join(aux.set_index("name")["truth"])

    reid = (per["n"] == 1).mean()
    pos = per[per["truth"]]
    exposed = (pos["n_pos"] == pos["n"]).mean()
    print(f"{b:8d} | {group_size.min():5d} | {(group_size == 1).mean():13.1%} | {reid:12.1%} | "
          f"{exposed:.1%} of {len(pos)} positives")
