# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:18:18 2025

@author: Zhaleh
"""
from evaluation.percentage_improvement_reports import run_config_directory, run_config_file, delta_c_gain_single_run
from evaluation.param_estimation_err_impact import delta_c_est_single_run, run_config_directory as run_dir_est ,run_config_file as run_file_est
from evaluation.misspec_sensitivity import misspec_batch_run
from evaluation.controlled_est_error import estimation_error_batch_run
from evaluation.sub_cost_analysis import cost_analysis_batch_run
import argparse


# Usage
if __name__ == "__main__":

    n_run = 200

    # Directory/File path
    directory_path = "inputs/json"

    file_path = "inputs/json/2items_4-1_1_10.json"

    # Delta C gain (VARMA vs ARMA inventory performance)
    # run_config_directory(delta_c_gain_single_run, directory_path, n_run)
    # run_config_file(delta_c_gain_single_run, file_path, n_run)
    
    # Delta C est (VARMA true true vs VARMA estimated inventory performance)
    # run_dir_est(delta_c_est_single_run, directory_path, n_run)
    # run_file_est(delta_c_est_single_run, file_path, n_run)

    
    # Controlled estimation error impact analysis
    # estimation_error_batch_run(1000)

    #misspec sensitivity analysis
    # misspec_batch_run(n_run)

    #plot unit and cost analysis
    cost_analysis_batch_run(n_run)