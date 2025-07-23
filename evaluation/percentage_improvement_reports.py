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
from util.helper_func import summary_table
from numpy.linalg import LinAlgError
import numpy as np

def delta_c_gain_single_run(run_id, config):
    """function for a single run to compute the percentage imporvement of VARMA
    multi-period cost compared to independent ARMA models"""
    """Run a scenario based on a configuration file."""

    steps = config["time_steps"]
    k = config["num_products"]
    sigma_base = 1  # config["noise_level"]
    model_order = config["model_order"]
    min_y = config["min_demand"]
    # dependency = config["dependency"]
    cost_params = config["cost_params"]
    max_rho = config['max_rho']
    alpha = config['alpha']
    try:
        # Simulate data
        varma_generator = varma_data_generator(
            steps, k, sigma_base, model_order[0], model_order[1], min_y, max_rho, alpha
        )
        data_fit, data_gen = varma_generator.generate_scenarios()

        improvement_results = []

        # Iterate over dependency items
        for title in data_fit.keys():
            print(title)
            df = {title: data_fit[title]}
            # Iterate over cost items
            for cost_idx in range(len(cost_params['holding_cost'])):
                costs = {key: values[cost_idx]
                         for key, values in cost_params.items() if len(values) > cost_idx}
                percentage_improvement, _,  _, _,_,_ = evaluate_varma_order_policy(
                    df, costs, model_order, data_gen, min_y)

                holding_cost = cost_params['holding_cost'][cost_idx]
                shortage_cost = cost_params['shortage_cost'][cost_idx]

                improvement_results.append([title, holding_cost[0], shortage_cost[0],
                                            percentage_improvement[title]])
    except LinAlgError:
        print("LU decomposition error occurred! Skipping this iteration and continuing.")
    return improvement_results


def delta_c_est_single_run(run_id, config):
    """function for a single run to compute the percentage imporvement of VARMA true
    multi-period cost compared to VARMA estimated models"""
    """Run a scenario based on a configuration file."""

    steps = config["time_steps"]

    k = config["num_products"]
    sigma_base = 1  # config["noise_level"]
    model_order = config["model_order"]
    min_y = config["min_demand"]
    # dependency = config["dependency"]
    cost_params = config["cost_params"]
    max_rho = config['max_rho']
    alpha = config['alpha']

    try:
        # Simulate data
        varma_generator = varma_data_generator(
            steps, k, sigma_base, model_order[0], model_order[1], min_y, max_rho, alpha
        )
        data_fit, data_gen = varma_generator.generate_scenarios()

        improvement_results = []

        # Iterate over dependency items
        for title in data_fit.keys():
            df = {title: data_fit[title]}
            # Iterate over cost items
            for cost_idx in range(len(cost_params['holding_cost'])):
                costs = {key: values[cost_idx]
                         for key, values in cost_params.items() if len(values) > cost_idx}
                _, _,  percentage_improvement, _ , _ , _ = evaluate_varma_order_policy(
                    df, costs, model_order, data_gen, min_y)

                holding_cost = cost_params['holding_cost'][cost_idx]
                shortage_cost = cost_params['shortage_cost'][cost_idx]

                improvement_results.append([title, holding_cost[0], shortage_cost[0],
                                            percentage_improvement[title]])
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
        'title', 'holding_unit_cost', 'shortage_unit_cost', 'percentage_improvement'
    ])

    # Generate summary table- fitted varma
    summary_tbl = summary_table(
        improvement_results, ['title', 'holding_unit_cost', 'shortage_unit_cost'], 'percentage_improvement')

    # write summary table to an output file
    filename = f"{func.__name__}(x{n_run}runs)-Items={config['num_products']}-p={config['model_order'][0]}-q={config['model_order'][1]}.csv"
    summary_tbl_path = f"data/output/{filename}"
    summary_tbl.to_csv(summary_tbl_path)

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.5f} seconds")
