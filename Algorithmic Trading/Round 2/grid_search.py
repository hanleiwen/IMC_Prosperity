# run_grid_search.py

import json
import subprocess
import itertools
import os
import re
from datetime import datetime
from tqdm import tqdm

# ---------- CONFIG ----------
TRADER_TEMPLATE_PATH = "./trader_test_file.py"
DATASET_ID = "2"
PRODUCT = "RAINFOREST_RESIN"

PARAM_GRID = {
    "fair_value": [10000],
    "take_width": [0, 1, 2],
    "clear_width": [0, 1, 2],
    "disregard_edge": [0, 1, 2],
    "join_edge": [0, 1, 2],
    "default_edge": [0, 1, 2],
    "soft_position_limit": [0, 10],
}

OUTPUT_DIR = "grid_search_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- HELPER FUNCTIONS ----------

def generate_param_combinations(grid):
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def inject_params_into_bot(template_path, output_path, new_rf_config):
    with open(template_path, "r") as f:
        lines = f.readlines()

    injected = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("PARAMS = {"):
            injected = True
            new_lines.append("PARAMS = {\n")
            new_lines.append(f"    Product.RAINFOREST_RESIN: {json.dumps(new_rf_config)},  # Injected\n")
        elif injected and line.strip().startswith("Product.SQUID_INK"):
            new_lines.append("    " + line)
            injected = False
        elif not injected:
            new_lines.append(line)

    with open(output_path, "w") as f:
        f.writelines(new_lines)

def run_backtest(bot_path, dataset_id):
    result = subprocess.run(
        ["prosperity3bt", bot_path, dataset_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout

def extract_total_profit(output):
    match = re.search(r"Total profit:\s*([0-9,]+)", output)
    if match:
        return int(match.group(1).replace(",", ""))
    return None

# ---------- MAIN ----------

def main():
    param_combinations = generate_param_combinations(PARAM_GRID)

    best_config = None
    best_score = float("-inf")
    log_path = os.path.join(OUTPUT_DIR, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    with open(log_path, "w") as log_file:
        log_file.write("Grid Search Log\n" + "=" * 50 + "\n")

        for i, config in enumerate(tqdm(param_combinations, desc="Grid Search")):
            tmp_bot_path = os.path.join(OUTPUT_DIR, f"trader_config_{i}.py")
            inject_params_into_bot(TRADER_TEMPLATE_PATH, tmp_bot_path, config)

            output = run_backtest(tmp_bot_path, DATASET_ID)

            score = extract_total_profit(output)

            log_file.write(f"\nConfig {i}: {json.dumps(config)}\n")
            log_file.write(f"Total Profit: {score}\n")

            if score is not None and score > best_score:
                best_score = score
                best_config = config

    print("\n✅ BEST CONFIGURATION:")
    print(json.dumps(best_config, indent=2))
    print("TOTAL PROFIT:", best_score)

if __name__ == "__main__":
    main()
