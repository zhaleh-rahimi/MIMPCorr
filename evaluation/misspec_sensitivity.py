# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:24:21 2025

@author: Zhaleh
"""

from data_prep.generate_varma_process import varma_data_generator
import pandas as pd
from joblib import Parallel, delayed
import time
from evaluation.evaluate_varma_order_policy import evaluate_varma_order_policy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.linalg import LinAlgError


def misspec_sensitivity_single_run(run_id):

    sigma_base = 1
    min_y = 10
    T= 500
    test_size=100
    train_size = T -test_size
    true_p=4
    true_q=1
    # # Uncomment to run for 2 items
    k = 2
    cost_params = {
        "holding_cost": [[10, 10]],
        "shortage_cost": [[50, 50]]
    }
    max_rho = {(1, 0): 0.8, (1, 1): 0.8, (2, 0): 0.8, (2, 1): 0.8,
               (3, 0): 0.76, (3, 1): 0.76, (4, 0): 0.73, (4, 1): 0.73}
    alpha = {(1, 0): 0.3, (1, 1): 0.3, (2, 0): 0.3, (2, 1): 0.3,
             (3, 0): 0.29, (3, 1): 0.29, (4, 0): 0.29, (4, 1): 0.29}
    # size_list=[30, 80, 100, 200, 300, 400, 500]
    
    # Uncomment to run for 4 items
    # k = 4
    # cost_params = {
    #     "holding_cost": [[10, 10, 10, 10]],
    #     "shortage_cost": [[10, 10, 10,  10]]
    # }
    # max_rho = {(1, 0): 0.8, (1, 1): 0.8, (2, 0): 0.7, (2, 1): 0.7,
    #            (3, 0): 0.7, (3, 1): 0.7, (4, 0): 0.65, (4, 1): 0.65}
    # alpha = {(1, 0): 0.3, (1, 1): 0.3, (2, 0): 0.89, (2, 1): 0.89,
    #          (3, 0): 0.78, (3, 1): 0.78, (4, 0): 0.7, (4, 1): 0.7}
    # size_list=[80, 100, 200, 300, 400, 1000,2000]
    
    improvement_results = []
    config = {
                "time_steps": T,
                "num_products": k,
                "model_order": [true_p, true_q],
                "min_demand": min_y,
                "max_rho": max_rho[(true_p, true_q)],
                "alpha": alpha[(true_p, true_q)],
                "train_size": train_size,
                "test_size": test_size
            }
    # Simulate data
    varma_generator = varma_data_generator(config=config, seed=run_id)
    data_fit, data_gen = varma_generator.generate_scenarios()
    title = f'Items={k}, p={true_p}, q={true_q}, High Dependence'
    df = {title: data_fit[title]}
    # Iterate    
    for p in range(1, 5):
        for q in range(3):        

            # Iterate over cost items
            for cost_idx in range(len(cost_params['holding_cost'])):
                try:
                    costs = {key: values[cost_idx]
                                for key, values in cost_params.items() if len(values) > cost_idx}
                    percentage_improvement, cost, _, forecast_performance,h_cost,s_cost = evaluate_varma_order_policy(
                        df, costs, [p, q], data_gen, min_y, train_size,test_size)

                    improvement_results.append([k,train_size, true_p, true_q , p, q,
                                                np.mean(
                                                    forecast_performance["VARMA"][title]['mape']),
                                                np.mean(
                                                    forecast_performance["VARMA"][title]['rmse']),
                                                cost["VARMA"][title],
                                                percentage_improvement[title]])
                except LinAlgError:
                    print("LU decomposition error occurred! Skipping this iteration and continuing.")
    improvement_results = pd.DataFrame(improvement_results, columns=["Items","Train_Size", "true_p", "true_q","p", "q", "MAPE", "RMSE",
                                                                     "total_cost", "percentage_improvement"])
    plot_error(improvement_results)

    return improvement_results

# batch run


def misspec_batch_run(n_run):
    # Start timer
    start_time = time.time()
    # Run the tasks in parallel
    results = Parallel(n_jobs=-1)(delayed(misspec_sensitivity_single_run)(run_id)
                                  for run_id in range(n_run))

    # End timer
    end_time = time.time()

    # Flatten the list of results
    improvement_results = pd.concat(results, ignore_index=True)

    # # List of columns to average
    # columns_to_average = ["MAPE", "RMSE", "total_cost", "percentage_improvement"]

    # # Compute the average for each column
    # average_results = improvement_results[columns_to_average].mean()
    # average_results= pd.DataFrame(improvement_results, columns=["Train_Size", "p", "q", "MAPE", "RMSE",
    #                                                                  "total_cost","percentage_improvement"])
    # write summary table to an output file
    filename = f"misspec_sens({n_run}).csv"
    summary_tbl_path = f"outputs/csv/{filename}"
    improvement_results.to_csv(summary_tbl_path)

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.5f} seconds")

    plot_error(improvement_results)

# plot error against traning size


def plot_error(df):
    k=df["Items"][0]
    # Line plots for MAPE and RMSE vs Training Size
    plt.figure(figsize=(12, 5))
    for metric in ["MAPE", "RMSE", "total_cost", "percentage_improvement"]:
        agg = (
            df.groupby(["p", "q"], as_index=False)
            .agg(mean=(f"{metric}", "mean"))
        )

        # Pivot for heatmap
        pivot = agg.pivot(index="p", columns="q", values="mean").sort_index(ascending=True)
        data = pivot.values

        # Create heatmap
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=180)
        im = ax.imshow(data)

        # Ticks / labels
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_xticklabels(pivot.columns.tolist())
        ax.set_yticklabels(pivot.index.tolist())
        ax.set_xlabel("q")
        ax.set_ylabel("p")
        ax.set_title(f"Mean {metric} by (p, q) - Items={k}")

        # Annotate each cell with the value (rounded)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center")

        # Highlight the best (minimum) cell
        min_idx = np.unravel_index(np.nanargmin(data), data.shape)
        rect = plt.Rectangle((min_idx[1]-0.5, min_idx[0]-0.5), 1, 1, fill=False, linewidth=2)
        ax.add_patch(rect)

        plt.tight_layout()
        out_path = f"outputs/figures/misspec-{metric}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.show()
   
