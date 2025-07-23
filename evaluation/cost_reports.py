# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 10:57:43 2025

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

def cost_single_run(run_id, config):
    """function for a single run to compute multi-period cost"""
    """Run a scenario based on a configuration file."""

    # Simulate data
    varma_generator = varma_data_generator(config = config,seed=run_id)
    data_fit, data_gen = varma_generator.generate_scenarios()
    # scenario_results = varma_generator.get_scenario_results()
    
    # Extract configuration parameters
    model_order = config['model_order']
    min_y = config['min_demand']         
    k = config['num_products']

    # Create a dictionary to hold the generated data
    key = f"Items={k}, p={model_order[0]}, q={model_order[1]}, High Dependence"
    df = {key: data_fit[key]}
   
    # Iterate over cost items    
    costs = {
        "holding_cost": [1,1,1,1],
        "shortage_cost": [1,1,1,1]
    }
    _, tc,  _, _,_,_ = evaluate_varma_order_policy(df, costs, model_order,data_gen, min_y)
    
    return tc

def compute_cost_config_file(file_path, n_run):
    print(f"Reading file: {file_path}")
    with open(file_path, 'r') as f:
        config = json.load(f)

    
    # Start timer
    start_time = time.time()
    # Run the tasks in parallel
    results = Parallel(n_jobs=-1)(delayed(cost_single_run)(run_id, config)
                                  for run_id in range(n_run))    
  
    # End timer
    end_time = time.time()

    # Flatten the list of results
    costs = [
    {'Model': key, 'Scenario': item1, 'Cost_mp': item2}
    for run in results
    for key, value in run.items()
    for item1, item2 in value.items()]
      
    # Convert to DataFrame
    costs = pd.DataFrame(costs)

    # Generate summary table- fitted varma
    cost_tbl = summary_table(
        costs, ['Model','Scenario'], 'Cost_mp') 
    # write summary table to an output file
    filename = f"cost_tbl_fit({n_run})-Items={config['num_products']}-p={config['model_order'][0]}-q={config['model_order'][1]}.csv"
    cost_tbl_path = f"outputs/csv/{filename}"
    cost_tbl.to_csv(cost_tbl_path)
 

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.5f} seconds")
    

def compute_cost_config_directory(directory_path, n_run):
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        config_file = os.path.join(directory_path, filename)  # Get full file path
        compute_cost_config_file(config_file, n_run)

  
    