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
from evaluation.controlled_est_error import estimation_error_batch_run, estimation_error_single_run
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
import argparse


def main(experiment_type, directory_path = "inputs/json", file_path ="inputs/json/2items_2-0_1_10.json", n_run=200):
    try:
        if experiment_type == "delta_c_gain":
            run_config_directory(delta_c_gain_single_run, directory_path, n_run)
        if experiment_type == "practical_err_impact":
            run_config_directory(delta_c_gain_single_run, directory_path, n_run)
        if experiment_type == "noise_sensitivity":
            # noise sensitivity analysis
            noise_level_sensitivity_batch_run(n_run)
        if experiment_type == "train_size_sensitivity":
            # Train size sensitivity analysis
            train_size_sensitivity_batch_run(n_run)
        if experiment_type == "cost_analysis":            
            # costs analysis
            cost_analysis_batch_run(n_run)
        if experiment_type == "controlled_err_impact":
            # Controlled estimation error impact analysis
            estimation_error_batch_run(n_run)          
            
            
    except Exception as e:
        print(e)

# Usage
if __name__ == "__main__":

    # Single run 
    # deltaC_gain, tcost, forecast_performance = inventory_optimization_single_run()
    # print("Delta C Gain:", deltaC_gain)
    # print("Total Cost:", tcost)
    # print("Forecast Performance:", forecast_performance)
    # Batch run of evaluation
    n_run = 3

    # Directory/File path
    directory_path = "inputs/json"

    file_path = "inputs/json/2items_4-1_1_10.json"

    # Delta C gain (VARMA vs ARMA inventory performance)
    # run_config_directory(delta_c_gain_single_run, directory_path, n_run)
    # run_config_file(delta_c_gain_single_run, file_path, n_run)
    
    # Delta C est (VARMA true true vs VARMA estimated inventory performance)
    # run_dir_est(delta_c_est_single_run, directory_path, n_run)
    # run_file_est(delta_c_est_single_run, file_path, n_run)

    # noise sensitivity analysis
    # noise_results = noise_level_sensitivity_batch_run(n_run)

    # Train size sensitivity analysis
    # train_size_sensitivity_batch_run(n_run)
    
    # # Test size sensitivity analysis
    # test_size_sensitivity_batch_run(n_run)
    
    # costs analysis
    # cost_analysis_batch_run(n_run)

    # Controlled estimation error impact analysis
    estimation_error_batch_run(n_runs=200)