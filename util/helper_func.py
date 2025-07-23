"""
Created on Thu Jan  9 11:04:31 2025

@author: Zhaleh
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from util.stat_tests import iqr
from collections import defaultdict
import numpy as np
import logging
import re
import os
from scipy.stats import iqr, ttest_rel, wilcoxon# Configure logging


def log_file():
    logging.basicConfig(
        filename="parallel_logs.log",  # Log file
        level=logging.INFO,
        format="%(asctime)s - %(process)d - %(message)s",
    )


def calculate_improvement_percentage(v1, v2):
    """ Performance Metric: Calculates the relative improvement percentage of two values"""
    
    return ((v1 - v2) / v1) * 100 if v1 else 0


def plot_cost_vs_variable(results_df, scenario, item, types):
    """Function to plot costs against a variable"""
    # Pivot the data for heatmap plotting
    heatmap_data = results_df.pivot_table(index='shortage_unit_cost',
                                          columns='holding_unit_cost',
                                          values='mean')
    # Pivot table with mean (std) annotation
    annot_data = results_df.pivot(index='shortage_unit_cost',
                                  columns='holding_unit_cost',
                                  values=['mean',
                                          'std']).apply(
        lambda row: row['mean'].astype(str) +
        ' (' + row['std'].round(1).astype(str) + ')', axis=1
    )

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=annot_data.values, fmt="",
                cmap='viridis', cbar=True, linewidth=.5)
    plt.title(f'{types}, {scenario}, {item} ')
    plt.suptitle('Mean (std) of Improvement%')
    plt.xlabel('Holding Unit Cost')
    plt.ylabel('Shortage Unit Cost')
    plt.tight_layout()
    plt.show()


def plot_varma_process(Y, title):
    """Plotting generated demand"""
    plt.figure(figsize=(12, 6))
    for i in range(Y.shape[1]):
        plt.plot(Y[:, i], label=f"Demand {i + 1}")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Demand Value")
    plt.legend()
    plt.grid()
    plt.show()


def summary_table(data, groupby_lst, summary_var):
    summary_tbl = pd.DataFrame()

    # Group by unique combination of groups
    summary_tbl = data.groupby(
        groupby_lst
    )[summary_var].agg(['mean', 'std', 'median', iqr]).reset_index()

    for col in ['mean', 'std', 'median', 'iqr']:
        summary_tbl[col] = summary_tbl[col].apply(lambda x: round(x, 2))

    return summary_tbl



def full_summary_table(df, groupby_cols):
    """
    Computes full summary: descriptive stats + paired tests + effect size.
    
    Returns a single DataFrame.
    """

    # Grouped stats for each variable
    def grouped_stats(var):
        return df.groupby(groupby_cols)[var].agg(['mean', 'std', 'median', iqr])

    cost_true_stats = grouped_stats('cost_varma_true').rename(columns=lambda x: f"true_{x}")
    cost_est_stats = grouped_stats('cost_varma_est').rename(columns=lambda x: f"est_{x}")
    diff_stats = grouped_stats('abs_percent_diff').rename(columns=lambda x: f"diff_{x}")

    # Merge all stats
    summary = cost_true_stats.join([cost_est_stats, diff_stats])

    # Add statistical test results
    results = []
    grouped = df.groupby(groupby_cols)
    for group_key, group in grouped:
        # Paired t-test
        try:
            t_pval = ttest_rel(group['cost_varma_est'], group['cost_varma_true']).pvalue
        except Exception:
            t_pval = None

        # Wilcoxon test
        try:
            w_pval = wilcoxon(group['cost_varma_est'], group['cost_varma_true']).pvalue
        except Exception:
            w_pval = None

        # Cohen’s d
        diff = group['cost_varma_est'] - group['cost_varma_true']
        pooled_std = diff.std()
        d = diff.mean() / pooled_std if pooled_std > 0 else 0

        results.append({
            **{col: val for col, val in zip(groupby_cols, group_key)},
            'pval_ttest': round(t_pval, 4) if t_pval is not None else None,
            'pval_wilcoxon': round(w_pval, 4) if w_pval is not None else None,
            "cohens_d": round(d, 3)
        })

    test_results_df = pd.DataFrame(results).set_index(groupby_cols)

    # Combine all
    final_summary = summary.join(test_results_df).reset_index()

    # Round numeric columns
    for col in final_summary.select_dtypes(include='number').columns:
        if col not in ['pval_ttest', 'pval_wilcoxon', 'cohens_d']:
            final_summary[col] = final_summary[col].round(2)

    return final_summary

def aggregate_autocorrelation_stats(all_results, max_lag=4):
    """
    Given a list of 'result' dictionaries, each containing
    'Autocorrelation Lag X' entries, this function sums them up 
    for each lag and returns the average and std (as NumPy arrays)
    across all rounds.
    """
    autocorr_data = defaultdict(list)

    # Collect autocorrelation data across all results
    for result in all_results:
        for lag in range(max_lag + 1):
            lag_key = f"Autocorrelation Lag {lag}"
            # Only add if it's not None
            if lag_key in result and result[lag_key] is not None:
                autocorr_data[lag_key].append(np.array(result[lag_key]))

    # Compute mean and std for each lag
    stats = {}
    for lag_key, values in autocorr_data.items():
        # Stack arrays along a new axis
        stacked = np.stack(values, axis=0)
        stats[lag_key] = {
            'mean': np.mean(stacked, axis=0),
            'std': np.std(stacked, axis=0)
        }

    return stats


def aggregate_by_scenario(all_results, max_lag=4):
    """
    1) Groups results by scenario.
    2) Computes mean/std of autocorrelation for each scenario.
    Returns a dict: { scenario_label: { 'Autocorrelation Lag 0': {...}, ... } }
    """
    grouped = group_results_by_scenario(all_results)
    scenario_stats = {}

    for scenario_label, scenario_result_list in grouped.items():
        scenario_stats[scenario_label] = aggregate_autocorrelation_stats(
            scenario_result_list, max_lag=max_lag
        )
    return scenario_stats


def group_results_by_scenario(all_results):
    """
    Group a list of result dictionaries by their Scenario key.
    Returns a dictionary mapping scenario_label -> list of result dicts.
    """
    scenario_data = defaultdict(list)
    for result in all_results:
        scenario_label = result["Scenario"]
        scenario_data[scenario_label].append(result)
    return scenario_data

def create_table_summary_delta_c(folder_path):
    
    
    # Initialize a list to store data for the summary
    summary_data = []
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # Extract model order (p, q) and num_products from the filename using regex
            match = re.match(r"sum_tbl_fit\(.*\)-Items=(\d+)-p=(\d+)-q=(\d+).csv", filename)
            if match:
                num_products = int(match.group(1))
                p = int(match.group(2))
                q = int(match.group(3))
    
                # Read the CSV file into a DataFrame
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
    
                # Filter rows where shortage_unit_cost = 1 and holding_unit_cost = 1
                filtered_df = df[(df["shortage_unit_cost"] == 1) & (df["holding_unit_cost"] == 1)]
    
                    
                # Ensure 'mean' and 'std' columns are numeric
                filtered_df["mean"] = pd.to_numeric(filtered_df["mean"], errors="coerce")
                filtered_df["std"] = pd.to_numeric(filtered_df["std"], errors="coerce")
    
                # Drop rows with NaN in 'mean' or 'std'
                filtered_df = filtered_df.dropna(subset=["mean", "std"])
                
                # Extract dependence types and append to summary_data
                for _, row in filtered_df.iterrows():
                    dependence_type = row["title"].split(",")[-1].strip()
                    summary_data.append({
                        "Items": num_products,
                        "p": p,
                        "q": q,
                        "Dependence": dependence_type,
                        "mean (std)": f"{row['mean']:.2f} ({row['std']:.2f})"
                    })


    
    # Convert the summary data into a DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Pivot the data to match the desired LaTeX table format
    pivot_table = summary_df.pivot(
        index=["Items","p", "q"],
        columns="Dependence",
        values= "mean (std)")
                                      

    
    # Save the summary to a CSV file
    output_file = "summary_table.csv"
    pivot_table.to_csv(output_file)
    
    print(f"Summary table saved to {output_file}")
