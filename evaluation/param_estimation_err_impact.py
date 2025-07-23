# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 10:56:55 2025

@author: Zhaleh
"""
from data_prep.generate_varma_process import varma_data_generator
import pandas as pd
import json
from joblib import Parallel, delayed
import time
import os
from evaluation.evaluate_varma_order_policy import evaluate_varma_order_policy
from util.helper_func import summary_table, full_summary_table
from numpy.linalg import LinAlgError
import numpy as np


def delta_c_est_single_run(run_id, config):
    """function for a single run to compute the percentage imporvement of VARMA true
    multi-period cost compared to VARMA estimated models"""
    """Run a scenario based on a configuration file."""

    model_order = config["model_order"]
    min_y = config["min_demand"]
    cost_params = config["cost_params"]
    

    try:
        # Simulate data
        varma_generator = varma_data_generator(config=config, seed=run_id)
        data_fit, data_gen = varma_generator.generate_scenarios()

        improvement_results = []

        # Iterate over dependency items
        for title in data_fit.keys():
            df = {title: data_fit[title]}
            # Iterate over cost items
            for cost_idx in range(len(cost_params['holding_cost'])):
                costs = {key: values[cost_idx]
                         for key, values in cost_params.items() if len(values) > cost_idx}
                _, cost, perc_diff , _ , _,_ = evaluate_varma_order_policy(
                    df, costs, model_order, data_gen, min_y)

                holding_cost = cost_params['holding_cost'][cost_idx]
                shortage_cost = cost_params['shortage_cost'][cost_idx]

                improvement_results.append([title, holding_cost[0], shortage_cost[0],
                                            cost['VARMA'][title],cost['VARMA_known'][title], np.abs(perc_diff[title])])
    except LinAlgError:
        print("LU decomposition error occurred! Skipping this iteration and continuing.")
    return improvement_results


def run_config_directory(func, directory_path, n_run):
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        config_file = os.path.join(directory_path, filename)  # Get full file path

        run_config_file(func, config_file, n_run)


def run_config_file(func, file_path, n_run):
    print(f"Reading file: {file_path}")
    with open(file_path, 'r') as f:
        config = json.load(f)

    # Start timer
    start_time = time.time()
    # Run the tasks in parallel
    results = Parallel(n_jobs=-1)(delayed(func)(run_id, config)
                                  for run_id in range(n_run))

    # End timer
    end_time = time.time()

    # Flatten the list of results
    improvement_results = [item for sublist in results for item in sublist]

    # Convert to DataFrame
    improvement_results = pd.DataFrame(improvement_results, columns=[
        'title', 'holding_unit_cost', 'shortage_unit_cost', 'cost_varma_est',
        'cost_varma_true','abs_percent_diff'
    ])
    # Summarize the results
    summary = summarize_overall_impact(improvement_results)

    sentence = (
    f"Across {summary['n_simulations']} simulations,\n "
    f"the average absolute cost difference between true and estimated VARMA forecasts \n"
    f"was only {summary['average_abs_percent_diff']}%, \n"
    f"with over {summary['percent_within_threshold']}% of simulations staying within\n "
    f"a {summary['threshold_used']}% cost deviation. \n"
    f"This suggests that estimation errors in VARMA parameters have limited impact on inventory performance.\n"
    )

    print("\n Report Summary:\n", sentence)


    # Generate summary table- fitted varma
    summary_tbl = full_summary_table(
        improvement_results, ['title', 'holding_unit_cost', 'shortage_unit_cost'])

    # write summary table to an output file
    filename = f"{func.__name__}(x{n_run}runs)-Items={config['num_products']}-p={config['model_order'][0]}-q={config['model_order'][1]}.csv"
    summary_tbl_path = f"outputs/csv/{filename}"
    summary_tbl.to_csv(summary_tbl_path)

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.5f} seconds")


def summarize_overall_impact(df, threshold=5.0):
    """
    Extracts:
    - average absolute cost difference
    - % of simulations within a threshold
    - total number of simulations
    """

    avg_abs_diff = df['abs_percent_diff'].mean()
    within_thresh = (df['abs_percent_diff'] <= threshold).mean() * 100
    total_runs = len(df)

    print("Overall Summary:")
    print(f"- Number of simulations: {total_runs}")
    print(f"- Average absolute % cost difference: {avg_abs_diff:.2f}%")
    print(f"- Proportion within {threshold}%: {within_thresh:.1f}%")

    return {
        "average_abs_percent_diff": round(avg_abs_diff, 2),
        "percent_within_threshold": round(within_thresh, 1),
        "n_simulations": total_runs,
        "threshold_used": threshold
    }
