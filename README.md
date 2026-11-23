# PETs Lab

A small learning lab on privacy-enhancing technologies (PETs). One synthetic problem, several techniques, each tested against an attack, with the costs and the caveats written down. This is a learning project, not a library and not a security audit.

> All experiments use one synthetic dataset (5,000 simulated aid recipients, single data seed) and are small-scale. The results illustrate trade-offs and are not findings about real data.

## The problem

5,000 synthetic aid recipients (age, sex, district, household size, displacement, income, and a sensitive attribute, medical_need). Three organizations each hold the records of four districts. They want aggregate statistics and a shared model without exposing individuals. Districts have deliberately uneven sizes (35 to 979 people), because small groups are where privacy techniques cost the most.

## Setup (Windows PowerShell, Python 3.12)

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python src\generate_data.py

Python 3.12 is used because TenSEAL (homomorphic encryption) had a prebuilt wheel for it. Other versions may work but were not tested.

## Run order

    python src\linkage_attack.py        # re-identification on a naive release
    python src\linkage_attack_v2.py     # fairer metric, age generalization
    python src\dp_counts.py             # Laplace mechanism on district counts
    python src\secret_sharing.py        # additive secret sharing, secure sum
    python src\secure_dp_sum.py         # secure sum + distributed noise, collusion
    python src\federated.py             # FedAvg vs centralized vs local-only
    python src\federated_small.py       # when does FL help (small orgs)
    python src\he_inference.py          # CKKS private inference (TenSEAL)
    python src\gradient_leakage.py      # what an FL server can learn from one update
    python src\dp_defense.py            # clipping + Gaussian noise on updates
    python src\multiround_attack.py     # averaging noisy updates over rounds
    python src\epsilon_accounting.py    # epsilon from scratch (RDP)
    python src\epsilon_crosscheck.py    # cross-check with dp-accounting (RDP and PLD)
    python src\dp_utility.py            # accuracy cost of private FL

NOTES.md is the full research log, with every number and caveat. COMPARISON.md is the summary table.

## Tests

A small test suite covers the additive secret-sharing function (`share`), checking that shares sum to the original value modulo the prime, that the share count is correct, and that shares are random.

```powershell
pip install pytest
python -m pytest tests -v
```

## Results in brief

- Removing the ID column is weak protection: 51.4% of an attacker's 500 named targets were re-identified. Coarsening age to 10-year bands cut that to 8.2%, but the smallest group still had size 1, so it is not k-anonymity.
- Laplace noise on counts has the same scale for every group, so the relative error is roughly 1 / (epsilon x count). A district with 5 true cases had 19% error at epsilon 1; one with 167 had about 6% at epsilon 0.1.
- Secret sharing computed the correct joint total without revealing any organization's count, but the output itself leaks: if two organizations collude, the third's count follows from the total. Adding distributed noise helps, and under collusion roughly half the noise protects the remaining party.
- Federated learning gave no gain when organizations had 500 or more rows each (AUC about 0.65 for every method). It helped when data was scarce: 0.602 vs 0.562 at 20 rows per organization.
- Homomorphic encryption (CKKS) added about 5e-7 error and about 10 ms per prediction, but a ciphertext was 324 KB for 40 bytes of features.
- A federated learning server that sees one plain update from a batch of one person recovers that person's features exactly. The attack fades as batch size grows (AUC 0.53 at batch 100).
- Clipping plus Gaussian noise defeats the attack on a single update (near chance at sigma 2 to 4), but averaging 200 rounds restores it at batch 1 (AUC 0.98 at sigma 1, 0.80 at sigma 4).
- Epsilon accounting (delta 1e-5): sigma 4 costs about 1.0 for one round but about 22 over 200 rounds. Epsilon around 1 over 200 rounds needs sigma around 57, and at that noise level private FL did not beat local-only training at any tested organization size.

## What I would not conclude

- That any technique is "safe" or "unsafe" in general. These are results on one synthetic dataset.
- That federated learning does not help. Organizations here are statistically near-identical and the model is tiny.
- That the accuracy numbers transfer to real data. The signal in the synthetic data is weak (AUC ceiling about 0.66).

## Limits

- Synthetic data, one data seed, small experiments (20 to 200 trials per cell); differences of about 0.01 AUC are not meaningful.
- Teaching implementations: the Laplace mechanism, secret sharing and FedAvg are written from scratch with numpy; the random number generator used for noise is not cryptographically secure; the HE parameters are common example values that were not audited.
- Epsilon assumes add/remove-one-person adjacency and no subsampling. Feature scaling used pooled statistics computed without noise, which sits outside the accounting, so the guarantee does not strictly cover the full pipeline.
- The attacks are simple ones, so their success rates are lower bounds on the risk.

## Not tested

Non-identical organizations; HE packing many people per ciphertext; effective epsilon of the secure-sum noise under collusion; secure aggregation as a defense against gradient leakage; leakage for larger models; tuning (learning rate, fewer rounds) and intermediate epsilon values for private FL.

## License

MIT. See LICENSE.
