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
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter,defaultdict

def cost_analysis_single_run(run_id):

    steps = 500
    k = 2
    sigma_base = 1
    model_order = [4, 1]
    min_y = 10
    train_size=400
    test_size=100
       
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
    h_units, s_units=defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(dict))
    for p in range(4, 5):
        for q in range(1,2):
            try:
                config = {
                    'time_steps': steps,
                    'num_products': k,
                    'model_order': [p, q],
                    'min_demand': min_y,
                    'train_size': train_size,
                    'test_size': test_size,
                    'max_rho': max_rho[(p, q)],
                    'alpha': alpha[(p, q)],
                    "sigma_base": sigma_base
                }
                # Simulate data
                varma_generator = varma_data_generator(config=config, seed=run_id)
                data_fit, data_gen = varma_generator.generate_scenarios()
                title = f'Items=2, p={p}, q={q}, High Dependence'
                df = {title: data_fit[title]}
                # Iterate over cost items
                for s in [1,5,10,20,30,50]:
                    for h in [10]:
                        costs = {"holding_cost" : [h,h],"shortage_cost": [s,s]}
                        percentage_improvement, cost, _, forecast_performance,hcost,scost = evaluate_varma_order_policy(
                            df, costs, [p, q], data_gen, min_y, train_size,test_size)
                        
                        for outer_key, inner_dict in hcost.items():
                            for inner_key, value in inner_dict.items():
                                h_units[outer_key][inner_key] = value / h
                        for outer_key, inner_dict in scost.items():
                            for inner_key, value in inner_dict.items():
                                s_units[outer_key][inner_key] = value / s
                        
                        improvement_results.append([s/h,s,h, p, q,
                                                   np.mean(
                                                       forecast_performance["VARMA"][title]['mape']),
                                                   np.mean(
                                                       forecast_performance["VARMA"][title]['rmse']),
                                                   cost["VARMA"][title],
                                                   hcost["VARMA"][title],
                                                   h_units["VARMA"][title],
                                                   scost["VARMA"][title],
                                                   s_units["VARMA"][title],
                                                   cost["ARIMA"][title],
                                                   hcost["ARIMA"][title],
                                                   h_units["ARIMA"][title],
                                                   scost["ARIMA"][title],
                                                   s_units["ARIMA"][title],
                                                   cost["VARMA_known"][title],
                                                   hcost["VARMA_known"][title],
                                                   h_units["VARMA_known"][title],
                                                   scost["VARMA_known"][title],
                                                   s_units["VARMA_known"][title],
                                                   percentage_improvement[title]])
            except LinAlgError:
                    print("LU decomposition error occurred! Skipping this iteration and continuing.")
    improvement_results = pd.DataFrame(improvement_results, columns=["s/h","s","h", "p", "q", "MAPE", "RMSE",
                                                                     "total_cost_varma","holding_cost_varma","holding_units_varma","shortage_cost_varma","shortage_units_varma", 
                                                                     "total_cost_arma","holding_cost_arma","holding_units_arma","shortage_cost_arma","shortage_units_arma",
                                                                     "total_cost_varma_true","holding_cost_varma_true","holding_units_varma_true","shortage_cost_varma_true","shortage_units_varma_true",
                                                                     "percentage_improvement"])
    

    return improvement_results

# batch run


def cost_analysis_batch_run(n_run):
    # Start timer
    start_time = time.time()
    # Run the tasks in parallel
    results = Parallel(n_jobs=-1)(delayed(cost_analysis_single_run)(run_id)
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
    filename = f"cost_analysis_({n_run}).csv"
    summary_tbl_path = f"output/csv/{filename}"
    improvement_results.to_csv(summary_tbl_path)

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.5f} seconds")

    # Plot for VARMA model
    # plot_costs(improvement_results)

    plot(improvement_results, 'varma', 'arma')
    plot(improvement_results, 'varma', 'varma_true')



# Assuming improvement_results DataFrame is already created

def plot_costs(improvement_results):
 
    
    # Grouping by p and q
    grouped = improvement_results.groupby(['p', 'q'])
    
    # Loop through each group and plot
    for (p_value, q_value), group in grouped:
        # Set up the figure for each p, q combination
        plt.figure(figsize=(10, 6))
        
        # Plot for VARMA
        plt.subplot(1, 2, 1)  # (rows, columns, position)
        sns.lineplot(data=group, x="h", y="total_cost_varma", label="Total Cost (VARMA)", marker="o")
        sns.lineplot(data=group, x="h", y="holding_cost_varma", label="Holding Cost (VARMA)", marker="^")
        sns.lineplot(data=group, x="h", y="shortage_cost_varma", label="Shortage Cost (VARMA)", marker="s")
        
        # Plot for ARMA
        plt.subplot(1, 2, 2)  # (rows, columns, position)
        sns.lineplot(data=group, x="h", y="total_cost_arma", label="Total Cost (ARMA)", marker="o")
        sns.lineplot(data=group, x="h", y="holding_cost_arma", label="Holding Cost (ARMA)", marker="^")
        sns.lineplot(data=group, x="h", y="shortage_cost_arma", label="Shortage Cost (ARMA)", marker="s")
        
        # Add titles, labels, and legends for both plots
        plt.suptitle(f'Costs for s= 10, p ={str(p_value)}, q={str(q_value)}')
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust for the suptitle
        plt.show()
    
   
def plot(improvement_results, model1, model2):
   

    # Grouping by p and q
    grouped = improvement_results.groupby(['p', 'q'])
    
    # Loop through each group and plot
    for (p_value, q_value), group in grouped:
        plt.figure(figsize=(10, 6))  # Create a new figure for each (p, q)
    
        # Plot Total Cost
        sns.lineplot(data=group, x="s/h", y=f"total_cost_{model1}", label=f"Total Cost ({model1})", linestyle="-", color='black', ci=None)
        sns.lineplot(data=group, x="s/h", y=f"total_cost_{model2}", label=f"Total Cost ({model2})", marker="o", linestyle="--", color='black', ci=None)
    
        # Plot Holding Cost
        sns.lineplot(data=group, x="s/h", y=f"holding_cost_{model1}", label=f"Holding Cost ({model1})",  linestyle="-", color='blue', ci=None)
        sns.lineplot(data=group, x="s/h", y=f"holding_cost_{model2}", label=f"Holding Cost ({model2})", marker="*", linestyle="--", color='blue', ci=None)
    
        # Plot Shortage Cost
        sns.lineplot(data=group, x="s/h", y=f"shortage_cost_{model1}", label=f"Shortage Cost ({model1})", linestyle="-", color='red', ci=None)
        sns.lineplot(data=group, x="s/h", y=f"shortage_cost_{model2}", label=f"Shortage Cost ({model2})", marker="s", linestyle="--", color='red', ci=None)
    
        # Title and labels
        plt.title(f'Costs for p={p_value}, q={q_value}')
        plt.xlabel('Unit Costs Ratio (s/h)')
        plt.ylabel('Cost')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'data/figures/p={p_value}, q={q_value}-cost.pdf', format="pdf")
        plt.show()  # Display the plot
    
    
    # Loop through each group and plot
    for (p_value, q_value), group in grouped:
        plt.figure(figsize=(10, 6))  # Create a new figure for each (p, q)
    
      
           
        # Plot Holding units
        sns.lineplot(data=group, x="s/h", y=f"holding_units_{model1}", label=f"Holding Units ({model1})",  linestyle="-", color='blue', ci=None)
        sns.lineplot(data=group, x="s/h", y=f"holding_units_{model2}", label=f"Holding Units ({model2})", marker="*", linestyle="--", color='blue', ci=None)
    
        # Plot Shortage Cost
        sns.lineplot(data=group, x="s/h", y=f"shortage_units_{model1}", label=f"Shortage Units ({model1})", linestyle="-", color='red', ci=None)
        sns.lineplot(data=group, x="s/h", y=f"shortage_units_{model2}", label=f"Shortage Units ({model2})", marker="s", linestyle="--", color='red', ci=None)
    
        # Title and labels
        plt.title(f'Units for p={p_value}, q={q_value}')
        plt.xlabel('Unit Costs Ratio (s/h)')
        plt.ylabel('Units')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'data/figures/p={p_value}, q={q_value}-units.pdf', format="pdf")
        plt.show()  # Display the plot