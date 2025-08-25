# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:18:18 2025

@author: Zhaleh
"""
from evaluation.percentage_improvement_reports import run_config_directory, run_config_file, delta_c_gain_single_run
from evaluation.param_estimation_err_impact import delta_c_est_single_run, run_config_directory as run_dir_est ,run_config_file as run_file_est
from evaluation.armaEst_vs_varmaTrue import armaEst_varmaTrue_single_run, run_config_directory as run_dir_arma_est, run_config_file as run_file_arma_est
from evaluation.misspec_sensitivity import misspec_batch_run
from evaluation.train_size_sensitivity import train_size_sensitivity_batch_run
from evaluation.noise_sensitivity import noise_level_sensitivity_batch_run
from evaluation.sub_cost_analysis import cost_analysis_batch_run
from evaluation.controlled_est_error import estimation_error_batch_run

import argparse


def main(experiment_type, directory_path:str, file_path:str , n_run=200):
    try:
        if experiment_type == "delta_c_gain":
            # Delta C gain (VARMA vs ARMA inventory performance)
            run_config_directory(delta_c_gain_single_run, directory_path, n_run)
        if experiment_type == "practical_err_impact":
            # Delta C est (VARMA true vs VARMA estimated inventory performance)
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
        if experiment_type == "misspec_sensitivity":
            # misspec sensitivity analysis
            misspec_batch_run(n_run) 
              
            
    except Exception as e:
        raise RuntimeError(f"Error in running {experiment_type} with {n_run} runs: {e}")    

# Usage
if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Run evaluation experiments.")
    # parser.add_argument("--experiment_type", type=str, required=True, choices=[
    #     "delta_c_gain",
    #     "practical_err_impact",
    #     "noise_sensitivity",
    #     "train_size_sensitivity",
    #     "cost_analysis",
    #     "controlled_err_impact",
    #     "misspec_sensitivity"
    # ], help="Type of experiment to run.")
    
    # # Optional arguments for directory and file paths
    # parser.add_argument("--directory_path", type=str, default=None, help="Directory path for input JSON files.")
    # parser.add_argument("--file_path", type=str, default="inputs/json/2items_1-0_1_10.json", help="File path for input JSON file.")
    # parser.add_argument("--n_run", type=int, default=200, help="Number of runs for the experiment.")        

    # Single run 
    # deltaC_gain, tcost, forecast_performance = inventory_optimization_single_run()
    # print("Delta C Gain:", deltaC_gain)
    # print("Total Cost:", tcost)
    # print("Forecast Performance:", forecast_performance)
    # Batch run of evaluation
    n_run = 200

    # Directory/File path
    directory_path = "inputs/json"

    file_path = "inputs/json/6items_2-1_1_10.json"

    # Delta C gain (VARMA vs ARMA inventory performance)
    # run_config_directory(delta_c_gain_single_run, directory_path, n_run)
    # run_config_file(delta_c_gain_single_run, file_path, n_run)
    
    # Delta C est (VARMA true true vs VARMA estimated inventory performance)
    # run_dir_est(delta_c_est_single_run, directory_path, n_run)
    # run_file_est(delta_c_est_single_run, file_path, n_run)

    # Delta C est (VARMA true true vs ARMA estimated inventory performance)
    # run_dir_arma_est(armaEst_varmaTrue_single_run, directory_path, n_run)
    # run_file_arma_est(armaEst_varmaTrue_single_run, file_path, n_run)


    # noise sensitivity analysis
    # noise_results = noise_level_sensitivity_batch_run(n_run)

    # Train size sensitivity analysis
    # train_size_sensitivity_batch_run(n_run)
    
    # # Test size sensitivity analysis
    # test_size_sensitivity_batch_run(n_run)
    
    # costs analysis
    # cost_analysis_batch_run(n_run)

    # Controlled estimation error impact analysis
    # estimation_error_batch_run(n_run)


    #misspec sensitivity analysis
    misspec_batch_run(n_run)