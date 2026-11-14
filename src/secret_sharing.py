import secrets
import pandas as pd

P = 2**61 - 1  # a large prime; all arithmetic is modulo P

def share(value, n):
    shares = [secrets.randbelow(P) for _ in range(n - 1)]
    shares.append((value - sum(shares)) % P)
    return shares

df = pd.read_csv("data/recipients.csv")
orgs = {
    "A": ["D01", "D02", "D03", "D04"],
    "B": ["D05", "D06", "D07", "D08"],
    "C": ["D09", "D10", "D11", "D12"],
}
names = list(orgs)
n = len(names)

# Each org's private local count of medical need
local = {k: int((df["district"].isin(v) & df["medical_need"]).sum()) for k, v in orgs.items()}
print("Private local counts:", local)

# Each org splits its count into n shares; share j goes to org j
all_shares = {k: share(local[k], n) for k in names}

# Same value shared twice gives different shares
print("A's count shared twice:", share(local["A"], n)[0], share(local["A"], n)[0])
print("The one share of A's count that org B sees:", all_shares["A"][1])

# Each org sums the shares it received and publishes only that sum
partial = [sum(all_shares[k][j] for k in names) % P for j in range(n)]
total = sum(partial) % P

print("Secure total:", total)
print("True total:  ", int(df["medical_need"][df["district"].isin(sum(orgs.values(), []))].sum()))
assert total == sum(local.values())
print("Match: OK")
