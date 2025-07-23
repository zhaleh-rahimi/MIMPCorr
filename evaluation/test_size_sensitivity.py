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
from util.helper_func import summary_table
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.linalg import LinAlgError


def test_size_sensitivity_single_run(run_id):

    steps = 500
    k = 2
    sigma_base = 1
    model_order = [4, 1]
    min_y = 10

    cost_params = {
        "holding_cost": [[1, 1]],
        "shortage_cost": [[1, 1]]
    }
    # max_rho = 0.6
    # alpha = 0.8
    
    max_rho = {(1, 0): 0.8, (1, 1): 0.8, (2, 0): 0.75, (2, 1): 0.75,
               (3, 0): 0.8, (3, 1): 0.73, (4, 0): 0.6, (4, 1): 0.5}
    alpha = {(1, 0): 0.3, (1, 1): 0.3, (2, 0): 0.2, (2, 1): 0.3,
             (3, 0): 0.3, (3, 1): 0.25, (4, 0): 0.8, (4, 1): -0.8}
    # # Simulate data
    # varma_generator = varma_data_generator(
    #     steps, k, sigma_base, model_order[0], model_order[1], min_y, max_rho, alpha
    # )
    # data_fit, data_gen = varma_generator.generate_scenarios()

    improvement_results = []

    # # Iterate
    # title = 'Items=2, p=4, q=1, High Dependence'
    # df = {title: data_fit[title]}
    for test_size in [20,50,100]:
        for p in range(1, 5):
            for q in range(2):
                try:
                    # Simulate data
                    varma_generator = varma_data_generator(
                        steps, k, sigma_base, p, q, min_y, max_rho[(p, q)], alpha[(p, q)]
                    )
                    data_fit, data_gen = varma_generator.generate_scenarios()
                    title = f'Items=2, p={p}, q={q}, High Dependence'
                    df = {title: data_fit[title]}
                    # Iterate over cost items
                    for cost_idx in range(len(cost_params['holding_cost'])):
                        
                        costs = {key: values[cost_idx]
                                 for key, values in cost_params.items() if len(values) > cost_idx}
                        percentage_improvement, cost, _, forecast_performance,hcost,scost = evaluate_varma_order_policy(
                            df, costs, [p, q], data_gen, min_y, 300,test_size)
    
                        improvement_results.append([test_size, p, q,
                                                   np.mean(
                                                       forecast_performance["VARMA"][title]['mape']),
                                                   np.mean(
                                                       forecast_performance["VARMA"][title]['rmse']),
                                                   cost["VARMA"][title],
                                                   hcost["VARMA"][title],
                                                   scost["VARMA"][title],
                                                   cost["ARIMA"][title],
                                                   hcost["ARIMA"][title],
                                                   scost["ARIMA"][title],
                                                   cost["VARMA_known"][title],
                                                   hcost["VARMA_known"][title],
                                                   scost["VARMA_known"][title],
                                                   percentage_improvement[title]])
                except LinAlgError:
                        print("LU decomposition error occurred! Skipping this iteration and continuing.")
    improvement_results = pd.DataFrame(improvement_results, columns=["Test_Size", "p", "q", "MAPE", "RMSE",
                                                                     "total_cost_varma","holding_cost_varma","shortage_cost_varma", 
                                                                     "total_cost_arma","holding_cost_arma","shortage_cost_arma",
                                                                     "total_cost_varma_true","holding_cost_varma_true","shortage_cost_varma_true",
                                                                     "percentage_improvement"])
   
    return improvement_results

# batch run


def test_size_sensitivity_batch_run(n_run):
    # Start timer
    start_time = time.time()
    # Run the tasks in parallel
    results = Parallel(n_jobs=-1)(delayed(test_size_sensitivity_single_run)(run_id)
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
    filename = f"test_size_sens({n_run}).csv"
    summary_tbl_path = f"data/output/{filename}"
    improvement_results.to_csv(summary_tbl_path)

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.5f} seconds")

    plot_error(improvement_results, 'varma')
    plot_error(improvement_results, 'varma_true')

# plot error against traning size


def plot_error(results_df, model_type):

    # Line plots for MAPE and RMSE vs Training Size
    plt.figure(figsize=(12, 5))
    for metric in ["MAPE", "RMSE", f"total_cost_{model_type}",f"holding_cost_{model_type}",f"shortage_cost_{model_type}", "percentage_improvement"]:
        sns.lineplot(data=results_df, x="Test_Size", y=metric, hue="p", style="q", marker="o")
        # plt.title(f"{metric.capitalize()} vs. Training Size")
        plt.xlabel("Test Size")
        plt.ylabel(metric)
        plt.legend(title="(p, q)", loc="best", bbox_to_anchor=(1, 1))
        plt.savefig(f'{metric}-test-size.pdf', format="pdf")
        plt.show()

    