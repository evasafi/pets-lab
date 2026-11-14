import numpy as np

orders = np.concatenate([1 + np.arange(1, 100) / 10.0, np.arange(12, 64), [128.0, 256.0, 512.0]])

def epsilon(sigma, rounds, delta=1e-5):
    rdp = rounds * orders / (2 * sigma ** 2)  # Gaussian mechanism, composed additively
    eps = rdp - (np.log(delta) + np.log(orders)) / (orders - 1) + np.log((orders - 1) / orders)
    return float(np.min(eps))

sigmas = [0.5, 1, 2, 4, 4 * np.sqrt(200)]
rounds_list = [1, 10, 50, 200]

print("epsilon at delta = 1e-5 (add/remove-one-person, no subsampling)")
print("sigma  | " + " | ".join(f"R={r:>3}" for r in rounds_list))
for s in sigmas:
    print(f"{s:6.1f} | " + " | ".join(f"{epsilon(s, r):6.2f}" for r in rounds_list))
