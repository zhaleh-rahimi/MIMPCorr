# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:18:18 2025

@author: Zhaleh
"""
from evaluation.percentage_improvement_reports import run_config_directory, run_config_file, delta_c_gain_single_run
from evaluation.param_estimation_err_impact import delta_c_est_single_run, run_config_directory as run_dir_est ,run_config_file as run_file_est
from evaluation.cost_reports import compute_cost_config_directory, compute_cost_config_file
from util.helper_func import create_table_summary_delta_c
from data_prep.generate_varma_process import varma_data_generator
from evaluation.evaluate_varma_order_policy import evaluate_varma_order_policy
from evaluation.train_size_sensitivity import train_size_sensitivity_batch_run, train_size_sensitivity_single_run
from evaluation.noise_sensitivity import noise_level_sensitivity_batch_run
from evaluation.test_size_sensitivity import test_size_sensitivity_batch_run, test_size_sensitivity_single_run
from evaluation.sub_cost_analysis import cost_analysis_batch_run, cost_analysis_single_run
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX

def load_data(file_path=None):
    
    """ Load the actual demand data from a CSV file"""
    data = pd.read_csv(file_path, index_col=0, engine='python')
    print(data.shape)
    return data

def find_best_varma_order(data, max_p=5, max_q=3):
    """
    Find the best VARMA order based on AIC.
    """
    best_aic = float('inf')
    best_order = [1, 0]
    
    for p in range(1,max_p + 1):
        for q in range(max_q + 1):
            try:
                model = VARMAX(data, order=(p, q))
                fitted_model = model.fit(disp=False)
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_order = [p, q]
            except Exception as e:
                print(f"Error fitting VARMA({p},{q}): {e}")
    print(f"Best VARMA order: {best_order} with AIC: {best_aic}")
    return best_order   

def inventory_optimization_single_run():
    """function for a single run to compute multi-period cost"""
    """Run a scenario based on a configuration file."""
    
    data = np.array(load_data(file_path='inputs/global_weekly_demand_group1.csv'))
    steps = 500
    k = 3
    min_y = np.mean(data)  # Minimum demand threshold for the VARMA process
    model_order = find_best_varma_order(data-min_y)
   

    costs = {
        "holding_cost": [1, 1,1],
        "shortage_cost": [1, 1,1]
    }
    # max_rho = 0.73
    # alpha = 0.25
    # sigma_base = 1
    # # Simulate data
    # varma_generator = varma_data_generator(
    #     steps, k, sigma_base, model_order[0], model_order[1], min_y, max_rho, alpha
    # )
    # data_fit, data_gen = varma_generator.generate_scenarios()
    # scenario_results = varma_generator.get_scenario_results()

    # key = f"Items={k}, p={model_order[0]}, q={model_order[1]}, High Dependence"
    # df = {key: data_fit[key]}
   
    df = {"Items=3 p=4, q=1, High Dependence": data}
    delta_c_gain, cost, delta_c_est, forecast_performance,h_cost,s_cost = evaluate_varma_order_policy(
        df, costs, model_order, df, min_y)

    return delta_c_gain, cost,  forecast_performance


# Usage
if __name__ == "__main__":

    # Single run 
    # deltaC_gain, tcost, forecast_performance = inventory_optimization_single_run()
    # print("Delta C Gain:", deltaC_gain)
    # print("Total Cost:", tcost)
    # print("Forecast Performance:", forecast_performance)
    # Batch run of evaluation
    n_run = 200

    # Directory/File path
    directory_path = "inputs/json"

    file_path = "inputs/json/2items_4-1_1_10.json"

    # Delta C gain (VARMA vs ARMA inventory performance)
    # run_config_directory(delta_c_gain_single_run, directory_path, n_run)
    # run_config_file(delta_c_gain_single_run, file_path, n_run)
    
    # Delta C est (VARMA true true vs VARMA estimated inventory performance)
    # run_dir_est(delta_c_est_single_run, directory_path, n_run)
    run_file_est(delta_c_est_single_run, file_path, n_run)

    # noise sensitivity analysis
    # noise_results = noise_level_sensitivity_batch_run(n_run)

    # Train size sensitivity analysis
    # train_size_sensitivity_batch_run(n_run)
    
    # # Test size sensitivity analysis
    # test_size_sensitivity_batch_run(n_run)
    
    # costs analysis
    # cost_analysis_batch_run(n_run)

    