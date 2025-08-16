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
from scipy.stats import iqr, ttest_rel, wilcoxon,t


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
            # 'pval_ttest': round(t_pval, 4) if t_pval is not None else None,
            # 'pval_wilcoxon': round(w_pval, 4) if w_pval is not None else None,
            # "cohens_d": round(d, 3)
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

def _tcrit(n: int, alpha: float = 0.05) -> float:
    """Return t critical value for n-1 degrees of freedom and alpha significance level."""
    return t.ppf(1 - alpha / 2, df=n - 1) if n > 1 else float("nan")

def _mean_se_ci(x: pd.Series, alpha: float = 0.05) -> dict:
    """Return mean, SE, and t-based 95% CI for a numeric Series."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = int(x.shape[0])
    mean = float(x.mean()) if n else float("nan")
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    h = _tcrit(n, alpha) * se if n > 1 else 0.0
    return {"mean": mean, "se": se, "ci_low": mean - h, "ci_high": mean + h, "n": n}


def _agg_ci(g: pd.DataFrame, col: str, prefix: str = None) -> pd.Series:
    """Compute mean/SE/CI for a column in a group and return with prefixed keys."""
    stats = _mean_se_ci(g[col])
    pre = (prefix or col)
    return pd.Series({
        f"{pre}_mean": stats["mean"],
        f"{pre}_se": stats["se"],
        f"{pre}_ci_low": stats["ci_low"],
        f"{pre}_ci_high": stats["ci_high"],
        f"{pre}_n": stats["n"],
    })


from typing import Optional, Dict, Any


def epsilon_cap_from_A(A1: np.ndarray, safety: float = 0.5) -> Dict[str, Any]:
    """
    Safe epsilon for multiplicative, elementwise perturbations of A1:
        A1_hat = A1 ∘ (1 + ε * N),  N_ij ~ N(0,1)
    Uses the conservative bound: |ε| < m / (kappa(V) * ||A1||_F),
    where m = 1 - ρ(A1), and V are right eigenvectors of A1.

    Args:
        A1: (k x k) AR(1) coefficient matrix.
        safety: extra shrink factor (0 < safety ≤ 1) for reporting (default 0.5).

    Returns:
        dict with rho, margin m, ||A1||_F, κ(V), eps_cap, eps_safe.
    """
    A1 = np.asarray(A1, dtype=float)
    # Spectral radius & margin
    evals, V = np.linalg.eig(A1)
    rho = float(np.max(np.abs(evals)))
    m = max(1.0 - rho, 0.0)

    # Frobenius norm
    fro = float(np.linalg.norm(A1, ord="fro"))

    # Condition number of eigenvector matrix V (Bauer–Fike factor)
    try:
        kappa = float(np.linalg.cond(V))
        if not np.isfinite(kappa):
            kappa = np.inf
    except np.linalg.LinAlgError:
        kappa = np.inf  # treat defective case as extremely ill-conditioned

    # Bound and safe recommendation
    eps_cap = 0.0 if (fro == 0.0 or m <= 0.0 or not np.isfinite(kappa) or kappa == 0.0) else m / (kappa * fro)
    eps_safe = safety * eps_cap

    return {
        "rho_A1": rho,
        "margin_m": m,
        "fro_A1": fro,
        "kappa_V": kappa,
        "eps_cap": eps_cap,     # theoretical cap
        "eps_safe": eps_safe,   # safety-scaled (recommendation)
        "safety": safety,
    }

def invertibility_margin_B(B1: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    """
    For Θ(z) = I + B1 z (MA(1)), invertibility requires all roots |z|>1,
    which is equivalent to ρ(B1) < 1. This reports ρ(B1) and its margin.
    """
    if B1 is None:
        return None
    B1 = np.asarray(B1, dtype=float)
    rho = float(np.max(np.abs(np.linalg.eigvals(B1))))
    return {"rho_B1": rho, "invertibility_margin": 1.0 - rho}

# --- convenience wrapper for CONFIG using generator ---

def epsilon_cap_from_config(config: dict,
                            seed: int = 0,
                            title_contains: str = "High Dependence",
                            safety: float = 0.5) -> Dict[str, Any]:
    """
    Pulls A1 (and B1 if present) from varma_data_generator and computes ε caps.

    Expects generator to expose:
      - generate_scenarios() -> (data_fit, data_gen)
      - get_scenario_by_title(title) -> dict with "AR Coefficients", "MA Coefficients"

    Args:
        config: your BASIC_CONFIG for VARMA(1,1) with num_products=2, max_rho=0.8, etc.
        seed: RNG seed for reproducibility.
        title_contains: substring to select the scenario title.
        safety: extra shrink factor for the recommended epsilon.

    Returns:
        dict with A1/B1 info and epsilon bounds.
    """
    from data_prep.generate_varma_process import varma_data_generator

    gen = varma_data_generator(config=config, seed=seed)
    data_fit, _ = gen.generate_scenarios()

    # Pick a scenario title
    titles = list(data_fit.keys())
    title = next((t for t in titles if title_contains in t), (titles[0] if titles else None))
    if title is None:
        raise RuntimeError("No scenarios returned by generator.")

    row = gen.get_scenario_by_title(title)
    A_list = [np.array(A, dtype=float) for A in row.get("AR Coefficients", [])]
    B_list = [np.array(B, dtype=float) for B in row.get("MA Coefficients", [])]
    if not A_list:
        raise RuntimeError("Generator did not return AR Coefficients.")
    A1 = A_list[0]
    B1 = B_list[0] if B_list else None

    out = {"title": title, "A1": A1, "B1": B1}
    out.update(epsilon_cap_from_A(A1, safety=safety))
    out["ma_invertibility"] = invertibility_margin_B(B1)
    return out

