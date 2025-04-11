import numpy as np


def multi_round_one_pick(num_teams=10000, num_iterations=5):
    """
    Runs a picking process for 'num_iterations' rounds.
    In each round, 'num_teams' players each pick exactly one spot (container).
    Everyone sees the updated distribution in real time (sequential picking).

    - spots: the list of (multiplier, hunters) for each container
    - chosen_counts: array tracking how many total picks each container has so far
    - We do not reset chosen_counts between rounds, so the distribution keeps evolving.
    """

    # 1) Our 10 containers: (multiplier, hunters)
    spots = [
        (10, 1),  # Spot 0
        (80, 6),  # Spot 1
        (37, 3),  # Spot 2
        (90, 10),  # Spot 3
        (31, 2),  # Spot 4
        (17, 1),  # Spot 5
        (50, 4),  # Spot 6
        (20, 2),  # Spot 7
        (73, 4),  # Spot 8
        (89, 8)  # Spot 9
    ]

    # 2) Initialize how many total times each container has been chosen
    chosen_counts = np.zeros(len(spots), dtype=int)

    # We'll keep track of total picks so far (start at 0).
    # Each round adds num_teams picks (one per player).
    total_picks_so_far = 0

    for round_idx in range(num_iterations):
        print(f"\n=== Starting round {round_idx + 1} ===")

        # 3) In this round, each of the 'num_teams' players picks exactly once
        for i in range(num_teams):
            # Compute payoff for each container j
            payoffs = []
            for j, (multiplier, hunters) in enumerate(spots):
                # fraction of picks that have gone to j so far
                if total_picks_so_far == 0:
                    frac_j = 0.0
                else:
                    frac_j = chosen_counts[j] / total_picks_so_far

                denom = hunters + 100 * frac_j

                if denom == 0:
                    payoff_j = float('inf')
                else:
                    payoff_j = 10000 * multiplier / denom

                payoffs.append(payoff_j)

            # Find the container with the best payoff
            best_spot = np.argmax(payoffs)
            # Update
            chosen_counts[best_spot] += 1
            total_picks_so_far += 1

        # 4) After finishing this round, print distribution
        distribution = chosen_counts / total_picks_so_far
        print("Chosen counts so far:", chosen_counts)
        print("Distribution so far:", distribution)


# Example usage:
if __name__ == "__main__":
    multi_round_one_pick(num_teams=10000, num_iterations=4)