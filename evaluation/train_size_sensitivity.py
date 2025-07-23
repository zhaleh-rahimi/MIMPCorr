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


def train_size_sensitivity_single_run(run_id):

     
    sigma_base = 1
    min_y = 10
   
    # # Uncomment to run for 2 items
    # k = 2
    # cost_params = {
    #     "holding_cost": [[10, 10]],
    #     "shortage_cost": [[10, 10]]
    # }
    # max_rho = {(1, 0): 0.8, (1, 1): 0.8, (2, 0): 0.8, (2, 1): 0.8,
    #            (3, 0): 0.76, (3, 1): 0.76, (4, 0): 0.73, (4, 1): 0.73}
    # alpha = {(1, 0): 0.3, (1, 1): 0.3, (2, 0): 0.3, (2, 1): 0.3,
    #          (3, 0): 0.29, (3, 1): 0.29, (4, 0): 0.29, (4, 1): 0.29}
    # size_list=[30, 80, 100, 200, 300, 400, 500]
    
    # Uncomment to run for 4 items
    k = 4
    cost_params = {
        "holding_cost": [[10, 10,10, 10]],
        "shortage_cost": [[10, 10,10, 10]]
    }
    max_rho = {(1, 0): 0.8, (1, 1): 0.8, (2, 0): 0.7, (2, 1): 0.7,
               (3, 0): 0.7, (3, 1): 0.7, (4, 0): 0.65, (4, 1): 0.65}
    alpha = {(1, 0): 0.3, (1, 1): 0.3, (2, 0): 0.89, (2, 1): 0.89,
             (3, 0): 0.78, (3, 1): 0.78, (4, 0): 0.7, (4, 1): 0.7}
    size_list=[80, 100, 200, 300, 400, 1000,2000]
    
    improvement_results = []
    
    test_size=100
    # Iterate    
    for train_size in size_list:
        steps = train_size+test_size
        for p in range(1, 5):
            for q in range(2):
                # Simulate data
                varma_generator = varma_data_generator(
                    steps, k, sigma_base, p, q, min_y, max_rho[(p,q)], alpha[(p,q)]
                )
                data_fit, data_gen = varma_generator.generate_scenarios()
                title = f'Items={k}, p={p}, q={q}, High Dependence'
                df = {title: data_fit[title]}
                # Iterate over cost items
                for cost_idx in range(len(cost_params['holding_cost'])):
                    try:
                        costs = {key: values[cost_idx]
                                 for key, values in cost_params.items() if len(values) > cost_idx}
                        percentage_improvement, cost, _, forecast_performance,h_cost,s_cost = evaluate_varma_order_policy(
                            df, costs, [p, q], data_gen, min_y, train_size,test_size)

                        improvement_results.append([k,train_size, p, q,
                                                   np.mean(
                                                       forecast_performance["VARMA"][title]['mape']),
                                                   np.mean(
                                                       forecast_performance["VARMA"][title]['rmse']),
                                                   cost["VARMA"][title],
                                                   percentage_improvement[title]])
                    except LinAlgError:
                        print("LU decomposition error occurred! Skipping this iteration and continuing.")
    improvement_results = pd.DataFrame(improvement_results, columns=["Items","Train_Size", "p", "q", "MAPE", "RMSE",
                                                                     "total_cost", "percentage_improvement"])
    plot_error(improvement_results)

    return improvement_results

# batch run


def train_size_sensitivity_batch_run(n_run):
    # Start timer
    start_time = time.time()
    # Run the tasks in parallel
    results = Parallel(n_jobs=-1)(delayed(train_size_sensitivity_single_run)(run_id)
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
    filename = f"train_size_sens({n_run}).csv"
    summary_tbl_path = f"data/output/{filename}"
    improvement_results.to_csv(summary_tbl_path)

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.5f} seconds")

    plot_error(improvement_results)

# plot error against traning size


def plot_error(results_df):
    k=results_df["Items"][0]
    # Line plots for MAPE and RMSE vs Training Size
    plt.figure(figsize=(12, 5))
    for metric in ["MAPE", "RMSE", "total_cost", "percentage_improvement"]:
        sns.lineplot(data=results_df, x="Train_Size", y=metric, hue="p", style="q", marker="o",ci=None, palette="deep")
        # plt.title(f"{metric.capitalize()} vs. Training Size")
        plt.xlabel("Train Size")
        plt.ylabel(metric)
        plt.legend(title="(p, q)", loc="best", bbox_to_anchor=(1, 1))
        plt.grid(color='gray', linestyle='-', linewidth=0.1)
        plt.savefig(f'data/figures/{k}Items-{metric}-train-size.pdf', format="pdf")
        plt.show()

   
