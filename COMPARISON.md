# PET comparison (synthetic data, single seed, small experiments)

All numbers come from the scripts in src/ on 5,000 synthetic recipients. "Not measured" means we did not test it.

| Technique | Protects whom from whom | Does not protect | Measured in this lab |
|---|---|---|---|
| ID removal only (baseline) | Nothing meaningful | Anyone with a side list of quasi-identifiers | 51.4% re-identified; 54.3% of positives exposed with certainty |
| Age generalization (5 / 10 year bands) | Partly, against linkage | Still min group size 1; homogeneous groups still leak | Re-identified 15.4% / 8.2%; positives exposed 21.0% / 14.8%; utility cost not measured |
| Differential privacy (Laplace counts) | Individuals, from anyone who sees the released statistics | Data during computation (a curator sees raw data); org-level totals | Noise scale 1/eps regardless of group size. Relative error about 1/(eps * count): D01 (count 5) 206% at eps 0.1 and 19% at eps 1; D03 (167) 6% at eps 0.1 |
| Secret sharing (secure sum) | Each org's input, from the other orgs, during computation | The output: if B and C collude, 875 - 402 - 178 = 295 recovers A exactly | Correct total (875). Runtime and communication not measured |
| Secure sum + distributed noise | Both of the above, without a trusted noise-adder | Weaker under collusion: with 2 of 3 colluding, error on A's count is 0.46 vs 0.98 on the total | Effective epsilon under collusion not computed. Simplified fixed-point model |
| Federated learning (FedAvg) | Raw rows stay at each org | Weight updates leak: a server seeing one org's update recovers a single record exactly (membership AUC 1.000 at batch 1), fading to 0.526 at batch 100 | No gain at 500+ rows/org; AUC gain over local-only of 0.040 / 0.037 / 0.023 / 0.006 at 20 / 50 / 150 / 500 rows. Orgs near-identical, so favorable to FL |
| Homomorphic encryption (CKKS inference) | A person's features and score, from the computing server | The model (server holds it in plaintext); query count and timing | Error 5e-07; 9.9 ms per prediction; 324 KB ciphertext vs 40 bytes raw. Parameters not audited |
| Clipping + Gaussian noise on FL updates (DP-SGD style) | Individuals in a batch, from the server seeing the updates | Cumulative leakage over many rounds (not tested; noise may average out). No epsilon computed | Single-update attack AUC at batch 1: 1.000 (sigma 0) to 0.825 / 0.644 / 0.554 / 0.555 (sigma 0.5 / 1 / 2 / 4); batch 5: 0.701 to 0.505; batch 20: 0.589 to 0.526. FL test AUC with 20 rows/org stayed 0.608 to 0.614 (std over draws about 0.03) |

## How they combine
- DP protects outputs; secret sharing, FL and HE protect the process. They address different threats, so real systems stack them (for example FL + secure aggregation + DP noise)
- No single technique above is enough alone: secret sharing leaks through its output, FL leaks through updates, HE says nothing about what the result reveals, DP needs a trusted party unless combined with the others

## Not tested yet
- Secure aggregation as a defense against gradient leakage; leakage for larger models; multi-round averaging attack on noisy updates; formal epsilon accounting
- Paired significance test for the FL results
- Non-IID organizations
- HE packing (many people per ciphertext)
- Effective epsilon of secure-sum noise under collusion
- Secret sharing runtime and communication
- DP on income means and DP model training (see the dp-fairness-adult repo for DP-SGD)

## Limits of this lab
- Synthetic data with weak signal; one seed; small experiments; teaching implementations, not audited libraries. Not evidence about real deployments
