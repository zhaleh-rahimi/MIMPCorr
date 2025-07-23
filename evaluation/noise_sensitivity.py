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
from util.stat_tests import relative_noise_dispersion as rnd
import gc
from numpy.linalg import LinAlgError


def noise_sensitivity_single_run(run_id):

    steps = 200
    k = 2

    model_order = [4, 1]
    min_y = 10

    cost_params = {
        "holding_cost": [[1, 1]],
        "shortage_cost": [[1, 1]]
    }
    max_rho = {(1, 0): 0.8, (1, 1): 0.8, (2, 0): 0.8, (2, 1): 0.8,
               (3, 0): 0.76, (3, 1): 0.76, (4, 0): 0.73, (4, 1): 0.73}
    alpha = {(1, 0): 0.3, (1, 1): 0.3, (2, 0): 0.29, (2, 1): 0.29,
             (3, 0): 0.29, (3, 1): 0.29, (4, 0): 0.30, (4, 1): 0.30}

    improvement_results = []

    # Iterate
    title = 'Items=2, p=4, q=1, High Dependence'
    cov_rng = [(x * 0.1 * min_y)**2 for x in range(1, 6)]

    for cov_noise in cov_rng:
        sigma_u = np.eye(k)*cov_noise
        mu_Y = np.ones(k)*min_y
        rnd_u, _ = rnd(mu_Y, sigma_u)

        # # Simulate data
        # varma_generator = varma_data_generator(
        #     steps, k, cov_noise, model_order[0], model_order[1], min_y, max_rho, alpha
        # )
        # data_fit, data_gen = varma_generator.generate_scenarios()
        # df = {title: data_fit[title]}
        for p in range(1, 5):
            for q in range(2):
                try:
                    config = {
                        "time_steps": steps,
                        "num_products": k,
                        "model_order": [p, q],
                        "min_demand": min_y,
                        "noise_level": cov_noise,
                        "max_rho": max_rho[(p, q)],
                        "alpha": alpha[(p, q)],
                        'model_order': [p, q]
                    }
                    # Simulate data
                    varma_generator = varma_data_generator(config=config, seed=run_id)
                    
                    data_fit, data_gen = varma_generator.generate_scenarios()

                    title = f'Items=2, p={p}, q={q}, High Dependence'
                    df = {title: data_fit[title]}

                    # Iterate over cost items
                    for cost_idx in range(len(cost_params['holding_cost'])):
                        costs = {key: values[cost_idx]
                                 for key, values in cost_params.items() if len(values) > cost_idx}
                        percentage_improvement, cost, _, forecast_performance,_,_ = evaluate_varma_order_policy(
                            df, costs, [p, q], data_gen, min_y, 100)

                        improvement_results.append([rnd_u, p, q,

                                                   np.mean(
                                                       forecast_performance["VARMA"][title]['mape']),
                                                   np.mean(
                                                       forecast_performance["VARMA"][title]['rmse']),
                                                   cost["VARMA"][title],
                                                    percentage_improvement[title]])
                except LinAlgError:
                    print("LU decomposition error occurred! Skipping this iteration and continuing.")

    improvement_results = pd.DataFrame(improvement_results, columns=["RND", "p", "q", "MAPE", "RMSE",
                                                                     "total_cost", "percentage_improvement"])

    return improvement_results


# batch run
def noise_level_sensitivity_batch_run(n_run):
    # Start timer
    start_time = time.time()
    # Run the tasks in parallel
    results = Parallel(n_jobs=-1)(delayed(noise_sensitivity_single_run)(run_id)
                                  for run_id in range(n_run))

    # End timer
    end_time = time.time()

    # Flatten the list of results
    improvement_results = pd.concat(results, ignore_index=True)

    # Compute the average of MAPE, RMSE, total_cost, and improvement_results
    # grouped by p, q, and Train_Size
    average_results = improvement_results.groupby(
        ["RND", "p", "q"])[["MAPE", "RMSE", "total_cost", "percentage_improvement"]].mean().reset_index()
    average_results = pd.DataFrame(average_results, columns=["RND", "p", "q", "MAPE", "RMSE",
                                                                    "total_cost", "percentage_improvement"])
    # write summary table to an output file
    filename = f"noise_level_sense({n_run}).csv"
    summary_tbl_path = f"outputs/csv/{filename}"
    improvement_results.to_csv(summary_tbl_path)

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.5f} seconds")

    plot_error(average_results)
    plot_error(improvement_results)

    del results  # Release memory
    gc.collect()  # Force garbage collection

# plot error against traning size


def plot_error(results_df):

    # Line plots for Noise Level   
    for metric in ["MAPE", "RMSE", "total_cost", "percentage_improvement"]:
        plt.figure()
        sns.lineplot(data=results_df, x="RND", y=metric, hue="p", style="q", marker="o", ci=None, palette="deep")
        # plt.title(f"{metric.capitalize()} vs. Noise Level Ratio")
        plt.xlabel("Relative Noise Dispersion")
        plt.ylabel(metric)
        plt.legend(title="(p, q)", loc="best", bbox_to_anchor=(1, 1))
        plt.grid(color='gray', linestyle='-', linewidth=0.1)
        plt.savefig(f'outputs/figures/{metric}-rnd.pdf', format="pdf")
        plt.show()

   
