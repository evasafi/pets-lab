import numpy as np
from dp_accounting import dp_event
from dp_accounting.rdp import rdp_privacy_accountant as rdp

delta = 1e-5
sigmas = [0.5, 1, 2, 4, 4 * np.sqrt(200)]
rounds_list = [1, 10, 50, 200]

def eps_rdp(sigma, rounds):
    acc = rdp.RdpAccountant()
    acc.compose(dp_event.SelfComposedDpEvent(dp_event.GaussianDpEvent(sigma), rounds))
    return acc.get_epsilon(delta)

print("dp-accounting RDP accountant, epsilon at delta = 1e-5")
print("sigma  | " + " | ".join(f"R={r:>3}" for r in rounds_list))
for s in sigmas:
    print(f"{s:6.1f} | " + " | ".join(f"{eps_rdp(s, r):6.2f}" for r in rounds_list))

try:
    from dp_accounting.pld import pld_privacy_accountant as pld

    def eps_pld(sigma, rounds):
        acc = pld.PLDAccountant()
        acc.compose(dp_event.SelfComposedDpEvent(dp_event.GaussianDpEvent(sigma), rounds))
        return acc.get_epsilon(delta)

    print()
    print("dp-accounting PLD accountant (tighter), epsilon at delta = 1e-5")
    print("sigma  | " + " | ".join(f"R={r:>3}" for r in rounds_list))
    for s in [1, 2, 4, 4 * np.sqrt(200)]:
        print(f"{s:6.1f} | " + " | ".join(f"{eps_pld(s, r):6.2f}" for r in rounds_list))
except Exception as e:
    print("PLD part failed:", repr(e))
