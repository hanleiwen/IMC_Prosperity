import numpy as np

# Define the container parameters, each container is represented as (multiplier, hunters)
spots = [
    (10, 1),
    (80, 6),
    (37, 3),
    (90, 10),
    (31, 2),
    (17, 1),
    (50, 4),
    (20, 2),
    (73, 4),
    (89, 8)
]

def simulate_using_prior(exponent=1, base=10000, delta=0.05):
    """
    1. Construct a prior distribution p based on the M/H ratio of each container:
         p_j = ( (M/H)_j^exponent ) / (sum over all j (M/H)_j^exponent )
       If exponent=1, then it is proportional to M/H; if exponent>1, 
       the advantage of higher ratios is amplified.
    2. Compute the expected payoff for each container using the fixed prior distribution:
         payoff_j = base * M_j / (H_j + 100 * p_j)
    3. Return the prior distribution, the payoff array, and the index of the container with the highest payoff.
    4. Then incorporate a congestion factor: assume everyone chooses the container with the 
       highest theoretical payoff, so add delta to that container's prior probability, renormalize, 
       and recalculate the payoffs to obtain the new optimal container.
    """
    # Calculate the M/H ratio for each container
    ratios = np.array([m / h for m, h in spots])
    # Apply the exponential factor (for example, exponent=1 means the original ratio; exponent>1 amplifies the advantage)
    ratios_exp = ratios ** exponent
    p_prior = ratios_exp / ratios_exp.sum()  # Normalize

    # Compute the expected payoff for each container (using the fixed p_prior, without dynamic congestion update)
    payoffs = np.zeros(len(spots))
    for j, (m, h) in enumerate(spots):
        denom = h + 100 * p_prior[j]
        payoffs[j] = base * m / denom if denom != 0 else float('inf')

    best_choice = np.argmax(payoffs)

    print("=== Using Prior Simulation ===")
    print("Exponent =", exponent)
    print("Prior distribution (p_prior):", p_prior)
    print("Calculated payoffs:", payoffs)
    print("Optimal container (if everyone uses p) is Container", best_choice)
    print()

    # Simulate congestion effect: assume everyone chooses the container with the highest payoff,
    # then that container's actual selection probability is increased by delta (congestion factor), then renormalized.
    p_modified = p_prior.copy()
    p_modified[best_choice] += delta
    p_modified /= p_modified.sum()

    # Recalculate payoffs based on the modified distribution.
    new_payoffs = np.zeros(len(spots))
    for j, (m, h) in enumerate(spots):
        denom = h + 100 * p_modified[j]
        new_payoffs[j] = base * m / denom if denom != 0 else float('inf')
    new_best_choice = np.argmax(new_payoffs)

    print("After incorporating congestion factor delta =", delta)
    print("Modified distribution (p_modified):", p_modified)
    print("New calculated payoffs:", new_payoffs)
    print("New optimal container is Container", new_best_choice)
    print()

    return p_prior, payoffs, best_choice, p_modified, new_payoffs, new_best_choice


def sequential_simulation(num_players=10000, base=10000, delta=0.05):
    """
    Simulate num_players players sequentially choosing one container each.
    Each player sees the updated distribution from previous players' choices.

    Method:
      - Initially, set chosen_counts to all 0 and total_picks = 0.
      - For each player i:
            Compute the real-time score for each container j:
              fraction_j = chosen_counts[j] / total_picks, or set fraction_j = 0 if total_picks is 0.
              payoff_j = base * M_j / (H_j + 100 * fraction_j)
            The player chooses the container with the highest payoff, then that container's count is incremented by 1,
            and total_picks is increased by 1.
      - Finally, compute the final distribution for each container (chosen_counts / total_picks).
    The function returns the final statistics, and then incorporates a congestion factor:
    Increase the final distribution of the most selected container by delta, renormalize, and recalculate payoffs.
    """
    num_spots = len(spots)
    chosen_counts = np.zeros(num_spots, dtype=int)
    total_picks = 0

    # Simulate each player sequentially choosing one container.
    for i in range(num_players):
        payoffs = np.zeros(num_spots)
        for j, (m, h) in enumerate(spots):
            if total_picks == 0:
                fraction = 0  # Initially, no congestion info is available.
            else:
                fraction = chosen_counts[j] / total_picks
            denom = h + 100 * fraction
            payoffs[j] = base * m / denom if denom != 0 else float('inf')
        best_choice = np.argmax(payoffs)
        chosen_counts[best_choice] += 1
        total_picks += 1

    final_distribution = chosen_counts / total_picks

    print("=== Sequential Simulation ===")
    print("After", num_players, "players have chosen one container each:")
    print("Chosen counts:", chosen_counts)
    print("Final distribution:", final_distribution)
    print("Container with highest selection (optimal by count) is Container", np.argmax(chosen_counts))
    print()

    # Incorporate the congestion factor: increase the final distribution of the container
    # with the highest selection by delta, then renormalize.
    final_distribution_mod = final_distribution.copy()
    idx_max = np.argmax(chosen_counts)
    final_distribution_mod[idx_max] += delta
    final_distribution_mod /= final_distribution_mod.sum()

    # Recalculate payoffs based on the modified distribution.
    new_payoffs = np.zeros(num_spots)
    for j, (m, h) in enumerate(spots):
        denom = h + 100 * final_distribution_mod[j]
        new_payoffs[j] = base * m / denom if denom != 0 else float('inf')
    new_best_choice = np.argmax(new_payoffs)

    print("After incorporating congestion factor delta =", delta)
    print("Modified distribution:", final_distribution_mod)
    print("New calculated payoffs:", new_payoffs)
    print("New optimal container after congestion is Container", new_best_choice)
    print()

    return chosen_counts, final_distribution, final_distribution_mod, new_payoffs, new_best_choice


# Call the sequential simulation (each player chooses one container) and print optimal selection info
sequential_simulation(num_players=10000, base=10000, delta=0)
simulate_using_prior(exponent=1, base=10000, delta=0)