import pandas as pd

# Parameters
params = {
    "default_spread_mean": 250,
    "default_spread_std": 50,
    "spread_std_window": 50,
    "zscore_threshold": 7,
    "target_position": 40
}

# Load CSV
df = pd.read_csv("./round-2-island-data-bottle/prices_round_2_day_-1.csv")  # <-- replace with actual filename

# Ensure timestamps are sorted per product
df.sort_values(by=["product", "timestamp"], inplace=True)

# Calculate spread = ask_price_1 - bid_price_1
df["spread"] = df["ask_price_1"] - df["bid_price_1"]

# Container for results
processed = []

# Process each product independently
for product, group in df.groupby("product"):
    group = group.copy()

    # Rolling mean and std of spread
    group["rolling_mean"] = group["spread"].rolling(window=params["spread_std_window"], min_periods=1).mean()
    group["rolling_std"] = group["spread"].rolling(window=params["spread_std_window"], min_periods=1).std()

    # Fill missing values with defaults
    group["rolling_mean"].fillna(params["default_spread_mean"], inplace=True)
    group["rolling_std"].fillna(params["default_spread_std"], inplace=True)

    # Compute z-score
    group["zscore"] = (group["spread"] - group["rolling_mean"]) / group["rolling_std"]

    # Decide position
    group["position"] = 0
    group.loc[group["zscore"] > params["zscore_threshold"], "position"] = -params["target_position"]
    group.loc[group["zscore"] < -params["zscore_threshold"], "position"] = params["target_position"]

    processed.append(group)

# Combine all products back together
result_df = pd.concat(processed).sort_values(by=["product", "timestamp"])

# Preview output
print(result_df[["timestamp", "product", "spread", "zscore", "position"]])

# Optional: Save to CSV
# result_df.to_csv("spread_analysis_output.csv", index=False)
