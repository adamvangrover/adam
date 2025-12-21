# scripts/analyze_simulation_results.py

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from core.world_simulation.data_manager import DataManager


def main():
    parser = argparse.ArgumentParser(description="Analyze the results of the world simulation.")
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="The number of simulation runs to analyze.",
    )
    args = parser.parse_args()

    data_manager = DataManager()
    all_data = data_manager.load_all_data(args.runs)

    # Example analysis: Plot the average GDP growth over time
    gdp_growth = all_data.explode("state").reset_index()
    gdp_growth["gdp"] = gdp_growth["state"].apply(lambda x: x["economic_indicators"]["gdp_growth"])
    gdp_growth_by_step = gdp_growth.groupby("level_0")["gdp"].mean()

    plt.figure(figsize=(10, 6))
    gdp_growth_by_step.plot()
    plt.title("Average GDP Growth Over Time")
    plt.xlabel("Step")
    plt.ylabel("GDP Growth")
    plt.grid(True)
    plt.show()

    # Example analysis: Plot the distribution of the final stock prices
    final_states = all_data.groupby("run_id").last()
    final_stock_prices = final_states["state"].apply(lambda x: pd.Series(x["stock_prices"])).unstack()

    plt.figure(figsize=(10, 6))
    final_stock_prices.hist(bins=20)
    plt.suptitle("Distribution of Final Stock Prices")
    plt.show()

if __name__ == "__main__":
    main()
