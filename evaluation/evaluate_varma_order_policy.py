# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:18:18 2025

@author: Zhaleh
"""
from util.helper_func import calculate_improvement_percentage
from forecast_models.fit_arima import fit_arima
from forecast_models.fit_varma import fit_varma
from inventory_models.policy_optimization import InventoryOptimizer
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from collections import defaultdict
from sklearn.covariance import LedoitWolf

def evaluate_varma_order_policy(data, cost_params, model_order, data_gen, min_demand=10, train_size=None, test_size=100):
    """Read scenario config file, generate data and run a multi-item multi-period inventory
    optimization and evaluate the process"""
    delta_c_gain = {}
    delta_c_est = {}
    policy = defaultdict(lambda: defaultdict(dict))
    cost = defaultdict(lambda: defaultdict(dict))
    s_cost, h_cost=defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(dict))
    forecast_performance = defaultdict(lambda: defaultdict(dict))

    for scenario, params in data.items():
        # print(f'{scenario}\n')

        # Simulated data for each scenario of strength
        data = pd.DataFrame(
            params, columns=[f"Item{i+1}" for i in range(params.shape[1])])
        
        if train_size is None:            
            split_size = round(len(data)*0.8)
            train, test = data.iloc[:split_size, :], data.iloc[split_size:, :]
        else:
            if (train_size + test_size) < len(data):            
                train, test = data.iloc[:train_size, :], data.iloc[train_size:train_size + test_size, :]
            else:
                train, test = data.iloc[:train_size, :], data.iloc[train_size:, :]

        df_gen = pd.DataFrame(
            data_gen[scenario], columns=[f"Item{i+1}" for i in range(params.shape[1])])

        # Extract values if matches are found
        p = model_order[0]
        q = model_order[1]

        predictions, fitted = {}, {}


        # Model fitting and prediction
        for model_type in ["ARIMA", "VARMA", "VARMA_known"]:
            if model_type == "ARIMA":
                predictions[model_type], fitted[model_type] = fit_arima(
                    train-min_demand, test-min_demand, p, q)
                predictions[model_type], fitted[model_type] = predictions[model_type] + \
                    min_demand, fitted[model_type]+min_demand

            elif model_type == "VARMA":
                predictions[model_type], fitted[model_type] = fit_varma(
                    train-min_demand, test-min_demand, p, q)
                predictions[model_type], fitted[model_type] = predictions[model_type] + \
                    min_demand, fitted[model_type]+min_demand
            elif model_type == "VARMA_known":
                if train_size is None:
                    predictions[model_type], fitted[model_type] = df_gen.iloc[split_size:,
                                                                          :], df_gen.iloc[:split_size, :]
                else:
                    predictions[model_type], fitted[model_type] = df_gen.iloc[train_size:train_size + test_size,
                                                                          :], df_gen.iloc[:train_size, :]

            else:
                raise ValueError("Unsupported model type")

            # Performance of model  on test data
            diff = test.values- predictions[model_type].values
            rmse = float(np.sqrt(np.mean(diff**2)))
            mape = float(np.mean(np.abs(diff) / np.maximum(np.abs(test.values), 1e-8)))*100
    
            # mape = mean_absolute_percentage_error(
            #     test.values, predictions[model_type].values, multioutput='raw_values')
            # rmse = root_mean_squared_error(
            #     test.values, predictions[model_type].values, multioutput='raw_values')
            forecast_performance[model_type][scenario]['mape'] = mape
            forecast_performance[model_type][scenario]['rmse'] = rmse

            # calculate sigma to input inventory optimizer                         
            residuals = train.values - fitted[model_type].values
            
            
            # Apply shrinkage estimation to correct bias
            lw = LedoitWolf()
            corrected_cov = lw.fit(residuals).covariance_
            
            # print("Bias-Corrected Covariance Matrix:\n", corrected_cov)
            # cov = pd.DataFrame(residuals).cov()
            sigma = np.diag(np.sqrt(corrected_cov))
            
            # if model_type == "VARMA_known":
            #     sigma = np.eye(k)*sigma_base

            # Inventory optimization
            optimizer = InventoryOptimizer(cost_params)
            policy[model_type][scenario] = optimizer.compute_policy(
                predictions[model_type].values, sigma)
            cost[model_type][scenario],h_cost[model_type][scenario],s_cost[model_type][scenario] = optimizer.evaluate_policy(
                test.values, policy[model_type][scenario])

        delta_c_gain[scenario] = calculate_improvement_percentage(
            cost["ARIMA"][scenario], cost["VARMA"][scenario])
        delta_c_est[scenario] = calculate_improvement_percentage(
            cost["VARMA"][scenario], cost["VARMA_known"][scenario])
    
    return delta_c_gain, cost, delta_c_est, forecast_performance,h_cost,s_cost
