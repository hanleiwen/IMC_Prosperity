import numpy as np

# Define container parameters; each container is represented as (multiplier, hunters)
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

def simulate_using_prior(exponent=1, base=10000, delta=0.005, delta2=0.0003, delta3=0.0002):
    """
    1. Construct the prior distribution p using the ratios (M/H):
         p_j = ((M/H)_j^exponent) / (sum over j ((M/H)_j^exponent))
       When exponent = 1 the distribution is proportional to M/H; if exponent > 1, it amplifies the advantage of higher ratios.
    2. Calculate the expected payoff for each container using the fixed prior:
         payoff_j = base * M_j / (H_j + 100 * p_j)
    3. Return the prior distribution, payoff array and index of the container with the highest payoff.
    4. Then, simulate congestion: assume that everyone chooses the container with the highest payoff; 
       add an extra delta to its prior probability, add delta2 to the second-best, and delta3 to the third-best container,
       renormalize the distribution, and recalculate the payoffs to determine the new optimal container.
    """
    # Calculate ratios M/H
    ratios = np.array([m / h for m, h in spots])
    # Apply the exponent factor (exponent=1 means original ratio; exponent>1 amplifies the advantage)
    ratios_exp = ratios ** exponent
    p_prior = ratios_exp / ratios_exp.sum()  # Normalize the prior distribution

    # Calculate the expected payoffs for each container (using the fixed p_prior, without dynamic congestion update)
    payoffs = np.zeros(len(spots))
    for j, (m, h) in enumerate(spots):
        denom = h + 100 * p_prior[j]
        payoffs[j] = base * m / denom if denom != 0 else float('inf')

    best_choice = np.argmax(payoffs)

    print("=== Using Prior Simulation ===")
    print("Exponent =", exponent)
    print("Prior distribution (p_prior):", p_prior)
    print("Calculated payoffs:", payoffs)
    print("Optimal container (if everyone uses p) is Container", best_choice, "with parameters", spots[best_choice])
    print()
    # Add: Show top 3 best containers before congestion
    sorted_indices = np.argsort(payoffs)[::-1]
    top3 = sorted_indices[:3]
    print("Top 3 containers before congestion:")
    for rank, idx in enumerate(top3, 1):
        m, h = spots[idx]
        print(f"  Rank {rank}: Container {idx} with Multiplier={m}, Hunters={h}, Payoff={payoffs[idx]:.2f}")
    print()
    # Simulate congestion: assume everyone chooses the container with the highest payoff,
    # then increase its selection fraction by an extra amount. 
    # Also, add extra amounts for the second-best and third-best containers.
    sorted_indices = np.argsort(payoffs)[::-1]  # Sorted indices in descending order
    best_choice   = sorted_indices[0]
    second_choice = sorted_indices[1]
    third_choice  = sorted_indices[2]

    # Modify the prior distribution: add delta to the best container, delta2 to the second, and delta3 to the third.
    p_modified = p_prior.copy()
    p_modified[best_choice]   += delta
    p_modified[second_choice] += delta2
    p_modified[third_choice]  += delta3
    # Recalculate the payoffs using the modified distribution
    new_payoffs = np.zeros(len(spots))
    for j, (m, h) in enumerate(spots):
        denom = h + 100 * p_modified[j]
        new_payoffs[j] = base * m / denom if denom != 0 else float('inf')
    new_best_choice = np.argmax(new_payoffs)
    new_sorted_indices = np.argsort(new_payoffs)[::-1]
    new_top3 = new_sorted_indices[:3]
    print("Top 3 containers after congestion adjustment:")
    for rank, idx in enumerate(new_top3, 1):
        m, h = spots[idx]
        print(f"  Rank {rank}: Container {idx} with Multiplier={m}, Hunters={h}, Payoff={new_payoffs[idx]:.2f}")
    print()
    print("After incorporating congestion factors delta =", delta, ", delta2 =", delta2, ", delta3 =", delta3)
    print("Modified distribution (p_modified):", p_modified)
    print("New calculated payoffs:", new_payoffs)
    print("New optimal container is Container", new_best_choice, "with parameters", spots[new_best_choice])
    print()

    return p_prior, payoffs, best_choice, p_modified, new_payoffs, new_best_choice, top3

'''
def sequential_simulation(num_players=10000, base=10000, delta=0.05):
    """
    Simulate num_players making a choice one after another, where each player chooses exactly one container.
    Each player sees the current distribution of previous choices (i.e., updated chosen_counts).
    
    Procedure:
      - Initially, set chosen_counts to zero for all containers, and total_picks = 0.
      - For each player i:
            For each container j, compute the current fraction:
              fraction_j = chosen_counts[j] / total_picks  (if total_picks is 0, set fraction_j = 0)
            Then compute:
              payoff_j = base * M_j / (H_j + 100 * fraction_j)
            The player chooses the container with the highest payoff, increment that container's count,
            and increase total_picks by 1.
      - Finally, compute the final distribution (chosen_counts / total_picks).
    Then, incorporate a congestion factor: increase the proportion of the most selected container by delta,
    renormalize the distribution, and recalculate the payoffs.
    """
    num_spots = len(spots)
    chosen_counts = np.zeros(num_spots, dtype=int)
    total_picks = 0

    # Simulate each player sequentially choosing a container
    for i in range(num_players):
        payoffs = np.zeros(num_spots)
        for j, (m, h) in enumerate(spots):
            if total_picks == 0:
                fraction = 0  # Initially, no congestion information is available
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
    print("Container with highest selection:", np.argmax(chosen_counts))
    print()

    # Incorporate congestion: Increase the proportion of the container with the highest count by delta, then renormalize.
    final_distribution_mod = final_distribution.copy()
    idx_max = np.argmax(chosen_counts)
    final_distribution_mod[idx_max] += delta
    final_distribution_mod /= final_distribution_mod.sum()

    # Recalculate the payoffs using the modified distribution
    new_payoffs = np.zeros(num_spots)
    for j, (m, h) in enumerate(spots):
        denom = h + 100 * final_distribution_mod[j]
        new_payoffs[j] = base * m / denom if denom != 0 else float('inf')
    new_best_choice = np.argmax(new_payoffs)

    print("After incorporating congestion factor delta =", delta)
    print("Modified distribution:", final_distribution_mod)
    print("New calculated payoffs:", new_payoffs)
    print("New optimal container is Container", new_best_choice)
    print()

    return chosen_counts, final_distribution, final_distribution_mod, new_payoffs, new_best_choice
'''
# Run simulations
def simulate_game_theory_choice(top3, payoffs, num_players=10000, temperature=1.0):
    """
    Each player selects one of the top 3 containers using a softmax function based on payoff.
    'temperature' controls randomness: lower = greedier.
    """
    payoff_values = np.array([payoffs[i] for i in top3])
    exp_vals = np.exp(payoff_values / temperature)
    probs = exp_vals / exp_vals.sum()

    choices = np.random.choice(top3, size=num_players, p=probs)
    counts = {i: (choices == i).sum() for i in top3}

    print("=== Game-Theoretic Choice Simulation ===")
    print(f"Temperature = {temperature}")
    print(f"Softmax Probabilities: {[f'{probs[i]:.3f}' for i in range(3)]}")
    for i in top3:
        print(f"Container {i}: chosen by {counts[i]} players, payoff = {payoffs[i]:.2f}")
    print()

    return counts
p_prior, payoffs, best_choice, p_mod, new_payoffs, new_best_choice, top3 = simulate_using_prior()

simulate_using_prior(exponent=2, base=10000, delta=0.05, delta2=0.03, delta3=0.02)

