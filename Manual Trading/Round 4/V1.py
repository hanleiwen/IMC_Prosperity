import numpy as np

# Define the 20 “Seal or No Seal” cases: (label, multiplier, contestants)
cases = [
    ("A1", 80, 6), ("A2", 50, 4), ("A3", 83, 7), ("A4", 31, 2), ("A5", 60, 4),
    ("B1", 89, 8), ("B2", 10, 1), ("B3", 37, 3), ("B4", 70, 4), ("B5", 90, 10),
    ("C1", 17, 1), ("C2", 40, 3), ("C3", 73, 4), ("C4", 100, 15), ("C5", 20, 2),
    ("D1", 41, 3), ("D2", 79, 5), ("D3", 23, 2), ("D4", 47, 3), ("D5", 30, 2),
]

labels = [c[0] for c in cases]
M = np.array([c[1] for c in cases], dtype=float)
H = np.array([c[2] for c in cases], dtype=float)


def simulate_with_costs(exponent=1, base=10000,
                        delta=0.005, delta2=0.0003, delta3=0.0002,
                        top_n=3):
    # Costs for selecting the 1st, 2nd, and 3rd pick:
    costs = [0, 50000, 100000]

    # 1) Compute prior from (M/H)^exponent
    ratios = (M / H) ** exponent
    p_prior = ratios / ratios.sum()

    # 2) Compute expected payoffs under static prior
    payoffs = base * M / (H + 100 * p_prior)

    # 3) Identify top_n static picks
    order = np.argsort(payoffs)[::-1]
    top_static = order[:top_n]

    # 4) Simulate congestion adjustment
    p_mod = p_prior.copy()
    for i, idx in enumerate(top_static):
        if i == 0:
            p_mod[idx] += delta
        elif i == 1:
            p_mod[idx] += delta2
        elif i == 2:
            p_mod[idx] += delta3
    p_mod /= p_mod.sum()
    payoffs_mod = base * M / (H + 100 * p_mod)
    order_mod = np.argsort(payoffs_mod)[::-1]
    top_mod = order_mod[:top_n]

    # 5) Compute revenues and net profits for picks
    static_revenues = payoffs[top_static]
    static_costs = costs[:len(top_static)]
    static_nets = static_revenues - static_costs
    static_cum_nets = np.cumsum(static_nets)

    mod_revenues = payoffs_mod[top_mod]
    mod_costs = costs[:len(top_mod)]
    mod_nets = mod_revenues - mod_costs
    mod_cum_nets = np.cumsum(mod_nets)

    # Print results
    print(f"\n=== Exponent = {exponent} ===")
    print("Static picks (revenue, cost, net):")
    for rank, idx in enumerate(top_static, 1):
        print(
            f" {rank}. {labels[idx]}: rev {static_revenues[rank - 1]:.2f}, cost {static_costs[rank - 1]}, net {static_nets[rank - 1]:.2f}")
    print("Static cumulative net by k:")
    for k, net in enumerate(static_cum_nets, 1):
        print(f"  k={k}: cum_net {net:.2f}")

    print("\nAfter congestion adjustment:")
    print("Adjusted picks (revenue, cost, net):")
    for rank, idx in enumerate(top_mod, 1):
        print(
            f" {rank}. {labels[idx]}: rev {mod_revenues[rank - 1]:.2f}, cost {mod_costs[rank - 1]}, net {mod_nets[rank - 1]:.2f}")
    print("Adjusted cumulative net by k:")
    for k, net in enumerate(mod_cum_nets, 1):
        print(f"  k={k}: cum_net {net:.2f}")

    return {
        "static": {
            "indices": top_static,
            "revenues": static_revenues,
            "costs": static_costs,
            "nets": static_nets,
            "cum_nets": static_cum_nets
        },
        "adjusted": {
            "indices": top_mod,
            "revenues": mod_revenues,
            "costs": mod_costs,
            "nets": mod_nets,
            "cum_nets": mod_cum_nets
        }
    }


if __name__ == "__main__":
    simulate_with_costs(exponent=1)
    simulate_with_costs(exponent=7)
