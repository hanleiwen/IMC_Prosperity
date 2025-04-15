import numpy as np
import scipy.stats
import math
import collections

#########################################
# 1. Custom Sea Turtle Reserve Price Distribution
#########################################
class sea_turtle_price_gen(scipy.stats.rv_continuous):
    """
    Custom distribution for sea turtle reserve prices:
      - Sea turtle reserve prices occur only in two intervals: [160, 200] and [250, 320];
      - They are uniformly distributed within these intervals, with a total length of 
        (200 - 160) + (320 - 250) = 40 + 70 = 110,
      - Thus, the PDF within the support is 1/110, and 0 outside;
      - The CDF is defined piecewise as:
          * For x < 160: F(x) = 0;
          * For 160 ≤ x ≤ 200: F(x) = (x – 160)/110;
          * For 200 < x < 250: F(x) = 40/110 (remains constant);
          * For 250 ≤ x ≤ 320: F(x) = [40 + (x – 250)]/110;
          * For x > 320: F(x) = 1.
    """
    def _pdf(self, x):
        if (160 <= x <= 200) or (250 <= x <= 320):
            return 1 / 110
        else:
            return 0.0

    def _cdf(self, x):
        if x < 160:
            return 0.0
        elif 160 <= x <= 200:
            return (x - 160) / 110
        elif 200 < x < 250:
            return 40 / 110
        elif 250 <= x <= 320:
            return (40 + (x - 250)) / 110
        else:
            return 1.0

    def _ppf(self, u):
        # To support vectorized operations, convert u to a numpy array
        u = np.asarray(u)
        out = np.empty_like(u, dtype=float)

        # For x in [160,200] corresponding to u in [0, 40/110]
        cond1 = (u <= 40 / 110)
        out[cond1] = 160 + 110 * u[cond1]

        # For x in [250,320] corresponding to u in (40/110, 1]
        cond2 = (u > 40 / 110) & (u <= 1.0)
        out[cond2] = 250 + 110 * (u[cond2] - (40 / 110))

        return out


#########################################
# 2. Objective Function: Calculate Total Profit for Two Bids
#########################################
def objective_two_bids(low, high, reserve_prices):
    """
    Calculate the total profit:
      - For a sea turtle with reserve price ≤ low, the first bid is used, profit = (320 - low);
      - For a sea turtle with reserve price in (low, high], the second bid is used, profit = (320 - high);
      - Otherwise, no transaction occurs (profit = 0).

    Parameters:
       low: The first bid (low bid)
       high: The second bid (must satisfy high ≥ low)
       reserve_prices: An array of sea turtle reserve prices

    Returns the total profit summed over all sea turtles.
    """
    profit_first = (320 - low) * (reserve_prices <= low)
    profit_second = (320 - high) * ((reserve_prices > low) & (reserve_prices <= high))
    total_profit = profit_first.sum() + profit_second.sum()
    return total_profit


#########################################
# 3. Exhaustive Search for the Optimal (low, high) Pair
#########################################
def maximize_two_bids(reserve_prices):
    """
    When low and high are integers within [160, 320] (with low ≤ high),
    use grid search to find the bid pair (low, high) that maximizes objective_two_bids.

    Returns the optimal (low, high) pair and the corresponding total profit.
    """
    best_profit = -np.inf
    best_pair = None
    for low in range(160, 321):
        for high in range(low, 321):
            profit = objective_two_bids(low, high, reserve_prices)
            if profit > best_profit:
                best_profit = profit
                best_pair = (low, high)
    return best_pair, best_profit


#########################################
# 4. Run Multiple Simulations to Gather Optimal Bid Pair Frequencies
#########################################
def run_simulation_two_bids(num_experiments=100, sample_size=5000):
    """
    Perform multiple experiments:
      - In each experiment, sample 'sample_size' sea turtle reserve prices from the custom distribution;
      - Use grid search to find the optimal (low, high) bid pair that maximizes total profit.
      
    Returns a list of the optimal (low, high) pairs from each experiment.
    """
    distribution = sea_turtle_price_gen(a=160, b=320)
    best_pairs = []
    for _ in range(num_experiments):
        reserves = distribution.rvs(size=sample_size)
        pair, profit = maximize_two_bids(reserves)
        best_pairs.append(pair)
    return best_pairs


#########################################
# 5. Main Entry Point
#########################################
if __name__ == "__main__":
    # For example, perform 100 experiments with a sample size of 5000 reserve prices each
    simulation_results = run_simulation_two_bids(num_experiments=100, sample_size=5000)

    # Count the frequency of the optimal (low, high) pair found in the experiments
    freq = collections.Counter(simulation_results)

    print("Optimal (low, high) pairs frequencies over 100 experiments:")
    for pair, count in sorted(freq.items()):
        print(f"{pair}: {count}")