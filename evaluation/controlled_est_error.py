"""
VARMA cost–vs–error sensitivity (refined)

What this script does
---------------------
• Keeps a single VARMA system (default: 2 items, VARMA(4,1)).
• Varies uncertainty via small parameter perturbations (EPSILONS) while enforcing
  stability/invertibility/PSD.
• Logs per-run metrics so you can plot **cost change vs forecast/variance error**—
  the reviewer’s core request.
• Produces CSVs:
    - estimation_error_results.csv (all runs)
    - estimation_error_summary_by_epsilon.csv (means by epsilon & kind)
    - estimation_error_sensitivity_mu_bins.csv (cost vs RMSE bins)
    - estimation_error_sensitivity_var_bins.csv (cost vs variance-error bins)
• Optionally saves simple matplotlib figures for the paper.

Assumptions
-----------
Relies on your existing modules:
- util.stat_tests: check_stationarity_AR, check_invertibility_MA
- data_prep.generate_varma_process: varma_data_generator
- inventory_models.policy_optimization: InventoryOptimizer

Run example
-----------
if __name__ == "__main__":
    df, s_eps, s_mu, s_var = estimation_error_batch_run(
        n_runs=200,
        base_config=BASE_CONFIG,
        epsilons=EPSILONS,
        cost_params=COSTS,
        train_T=400,
        test_T=100,
        n_jobs=-1,
        out_dir="./outputs/estimation_error"
    )

"""

import os
import time
import math
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from util.stat_tests import check_stationarity_AR, check_invertibility_MA
from data_prep.generate_varma_process import varma_data_generator
from inventory_models.policy_optimization import InventoryOptimizer

# ------------------------- configuration -------------------------

BASE_CONFIG = {
    "time_steps": 500,
    "num_products": 2,
    "model_order": [2, 1],  # (p, q) — q=0 covers VAR
    "min_demand": 10,
    "noise_level": 1,
    "max_rho": 0.75,
    "alpha": 0.30,
}

COSTS = {
    "holding_cost": [1.0, 1.0],
    "shortage_cost": [5.0, 5.0],
}

# Levels of parameter perturbation (controls uncertainty magnitude)
EPSILONS = [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]

# ------------------------- utilities -------------------------

def nearest_psd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project symmetric A onto the PSD cone (eigenvalue clipping)."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T


def compute_mu_from_params(
    U: np.ndarray,
    A_list: List[np.ndarray],
    B_list: List[np.ndarray],
    intercept: Optional[np.ndarray] = None,
    base_config: dict= BASE_CONFIG,
) -> np.ndarray:
    """
    One-step-ahead conditional mean:
        μ_t = c + Σ_{k=1..p} A_k Y_{t-k} + Σ_{j=1..q} B_j U_{t-j}
    where c is a k-dim constant vector (if provided).
    Returns array shaped like Y.
    """
    cfg = dict(base_config)
    process = varma_data_generator(config=cfg, seed=None)
    mu= process.get_conditional_mean(A_list, B_list, U)
    return mu


def order_up_to_cost(
    demand: np.ndarray,
    mu: np.ndarray,
    sigma_vec: np.ndarray,
    cost_params: dict,
) -> Tuple[float, float, float]:
    """Compute base-stock policy from μ and σ and evaluate cost via InventoryOptimizer."""
    inv = InventoryOptimizer(cost_params)
    policy = inv.compute_policy(mu, sigma_vec)
    total, htot, stot = inv.evaluate_policy(demand, policy)
    return total, htot, stot


def mu_error_stats(mu_hat: np.ndarray, mu_true: np.ndarray) -> Tuple[float, float, float]:
    """Return RMSE, mean bias, and MAPE of the one-step conditional mean."""
    diff = mu_hat - mu_true
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))
    # Safe MAPE (μ_true protected by epsilon); your min_demand makes zeros unlikely
    mape = float(np.mean(np.abs(diff) / np.maximum(np.abs(mu_true), 1e-8)))
    return rmse, bias, mape


def var_err_diag(Sigma_hat: np.ndarray, Sigma_true: np.ndarray) -> float:
    """Relative error of noise variances (diagonal only)."""
    d_hat = np.clip(np.diag(Sigma_hat), 1e-12, None)
    d_true = np.clip(np.diag(Sigma_true), 1e-12, None)
    return float(np.mean(np.abs(d_hat - d_true) / d_true))

def error_stats(d: np.ndarray, d_hat: np.ndarray) -> Tuple[float, float, float, float]:
    """Return RMSE, mean bias, and MAPE of the error."""
    diff = d - d_hat
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))
    mape = float(np.mean(np.abs(diff) / np.maximum(np.abs(d_hat), 1e-8)))
    r2 = 1 - np.sum(diff**2) / np.sum((d - np.mean(d))**2)
    return rmse, bias, mape,r2

# ----------------- single-run with perturbation sweep -----------------

def estimation_error_single_run(
    run_id: int,
    base_config: dict,
    epsilons: List[float],
    cost_params: dict,
    ) -> pd.DataFrame:
    """
    One simulation + sweep over epsilons.
    Returns long-form DataFrame with cost deltas and μ/variance error diagnostics.
    """
    rng = np.random.default_rng(run_id)
    cfg = dict(base_config)
    k = cfg["num_products"]
    T= cfg["time_steps"]
    test_T = 100
    train_T = T - test_T    
    p, q = cfg["model_order"]
    burn = max(p, q)

    gen = varma_data_generator(cfg, seed=run_id)
    data_fit, data_gen = gen.generate_scenarios()
    
    # title = list(data_fit.keys())[0]
    title = f"Items={k}, p={p}, q={q}, High Dependence"

    Y = np.asarray(data_fit[title], dtype=float)            # realized demand
    mu_true_all = np.asarray(data_gen[title], dtype=float)  # true conditional means
    U = Y - mu_true_all                                     # innovations
       
    start_test = max(burn + 1, train_T)
    end_test = min(T, start_test + test_T)
    if end_test - start_test <= 0:
        raise ValueError(f"Empty test window: T={T}, start={start_test}, end={end_test}")

    demand_test = Y[start_test:end_test]
    mu_true_test = mu_true_all[start_test:end_test]

    # True parameters from generator
    row = gen.get_scenario_by_title(title)
    A_true = [np.array(A, dtype=float) for A in row["AR Coefficients"]]
    B_true = [np.array(B, dtype=float) for B in row["MA Coefficients"]]
    nu_true = getattr(gen, "nu_u", None)
    Sigma_true = getattr(gen, "sigma_u", None)
   
    if Sigma_true is None:
        if "Sigma_U" in row and row["Sigma_U"] is not None:
            Sigma_true = np.array(row["Sigma_U"], dtype=float)
        else:
            raise RuntimeError("Sigma_U not available from generator nor results.")
    Sigma_true = np.asarray(Sigma_true, dtype=float)
    sigma_true_vec = np.sqrt(np.clip(np.diag(Sigma_true), 1e-12, None))

    # Baseline (true) cost
    C_true, H_true, S_true = order_up_to_cost(demand_test, mu_true_test, sigma_true_vec, cost_params)

    records = []

    def make_A_hat(eps: float) -> List[np.ndarray]:
        if eps == 0.0:
            return [A.copy() for A in A_true]
        noise = [rng.normal(size=A.shape) for A in A_true]
        A_hat = [A_true[i] *(1+ eps * noise[i]) for i in range(len(A_true))]
        # if not check_stationarity_AR(A_hat):
            # Try the generator's helper to repair into admissible set
            # A_hat = gen.adjust_coefficients(A_hat, check_stationarity_AR)
        return A_hat

    def make_B_hat(eps: float) -> List[np.ndarray]:
        if len(B_true) == 0:
            return []
        if eps == 0.0:
            return [B.copy() for B in B_true]
        noise = [rng.normal(size=B.shape) for B in B_true]
        B_hat = [B_true[i] *(1+ eps * noise[i]) for i in range(len(B_true))]
        # if not check_invertibility_MA(B_hat):
            # B_hat = gen.adjust_coefficients(B_hat, check_invertibility_MA)
        return B_hat

    def make_Sigma_hat(eps: float) -> np.ndarray:
        if eps == 0.0:
            return Sigma_true.copy()
        noise = rng.normal(size=Sigma_true.shape)
        return Sigma_true *(1 + eps * noise)

    for eps in epsilons:
        # A-only
        A_hat = make_A_hat(eps)
        mu_A_all = compute_mu_from_params(U, A_hat, B_true, intercept=nu_true, base_config=base_config)
        mu_A = mu_A_all[start_test:end_test]
        C_A, H_A, S_A = order_up_to_cost(demand_test, mu_A, sigma_true_vec, cost_params)
        # rmse_A, bias_A, mape_A = mu_error_stats(mu_A, mu_true_test)
        rmse_A, bias_A, mape_A, r2_A = error_stats(demand_test, mu_A)
        
        # B-only
        if len(B_true) > 0:
            B_hat = make_B_hat(eps)
            mu_B_all = compute_mu_from_params(U, A_true, B_hat, intercept=nu_true, base_config=base_config)
            mu_B = mu_B_all[start_test:end_test]
            C_B, H_B, S_B = order_up_to_cost(demand_test, mu_B, sigma_true_vec, cost_params)
            # rmse_B, bias_B, mape_B = mu_error_stats(mu_B, mu_true_test)
            rmse_B, bias_B, mape_B,r2_B=error_stats(demand_test, mu_B)
        else:
            C_B = H_B = S_B = np.nan
            rmse_B = bias_B = mape_B = np.nan

        # Sigma-only (variance misspecification with true mean)
        Sigma_hat = make_Sigma_hat(eps)
        sigma_hat_vec = np.sqrt(np.clip(np.diag(Sigma_hat), 1e-12, None))
        C_S, H_S, S_S = order_up_to_cost(demand_test, mu_true_test, sigma_hat_vec, cost_params)
        # v_err = var_err_diag(Sigma_hat, Sigma_true)
        rmse_S, bias_S, mape_S,r2_S = error_stats(demand_test, mu_true_test)

        # Combined (mean + variance misspecification)
        A_hat2 = make_A_hat(eps)
        B_hat2 = make_B_hat(eps)
        mu_AB_all = compute_mu_from_params(U, A_hat2, B_hat2, intercept=nu_true, base_config=base_config)
        mu_AB = mu_AB_all[start_test:end_test]
        C_AB, H_AB, S_AB = order_up_to_cost(demand_test, mu_AB, sigma_hat_vec, cost_params)
        # rmse_AB, bias_AB, mape_AB = mu_error_stats(mu_AB, mu_true_test)
        rmse_AB, bias_AB, mape_AB,r2_AB = error_stats(demand_test, mu_AB)

        # Collect
        def rec(kind: str, C: float, H: float, S: float, rmse: float, bias: float, mape: float, r2: Optional[float] = None):
            return dict(
                run_id=run_id,
                epsilon=eps,
                kind=kind,
                C_true=C_true,
                C_est=C,
                DeltaC_pct=100.0 * (C - C_true) / max(C_true, 1e-12),  # % change vs baseline cost
                H_est=H,
                S_est=S,
                rmse_mu=rmse,
                bias_mu=bias,
                mape_mu=mape,
                r2 = r2
            )

        records.append(rec("A_only", C_A, H_A, S_A, rmse_A, bias_A, mape_A, r2_A))
        records.append(rec("Sigma_only", C_S, H_S, S_S, rmse_S, bias_S, mape_S,r2_S))
        records.append(rec("Combined", C_AB, H_AB, S_AB, rmse_AB, bias_AB, mape_AB, r2_AB))
        if not math.isnan(C_B):
            records.append(rec("B_only", C_B, H_B, S_B, rmse_B, bias_B, mape_B, r2_B))

    return pd.DataFrame.from_records(records)

# ----------------- plotting helpers (optional) -----------------

def _errbar(ax, x, y, yerr, label):
    ax.errorbar(x, y, yerr=yerr, fmt="-o", label=label, capsize=3)
    ax.set_ylabel("Relative cost difference(%)")
    ax.grid(True, alpha=0.3)


def plot_cost_vs_rmse(summary_mu_bins: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, grp in summary_mu_bins.groupby("kind"):
        _errbar(ax, grp["RMSE_mean"].values, grp["DeltaC_mean_pct"].values, grp["DeltaC_se"].values, label=k)
    ax.legend(title="Perturbation kind")
    ax.set_xlabel("RMSE of forecast")
    ax.set_title("Relative cost difference(%) vs rmse of forecast")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cost_vs_var(summary_var_bins: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    _errbar(ax, summary_var_bins["r2_mean"].values, summary_var_bins["DeltaC_mean_pct"].values, summary_var_bins["DeltaC_se"].values, label="Sigma_only")
    ax.legend()
    ax.set_xlabel("R² (variance error)")
    ax.set_title("Relative cost difference(%) vs variance error (diag Σ_U)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_cost_vs_eps(summary_by_eps: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for k, grp in summary_by_eps.groupby("kind"):
        _errbar(ax, grp["epsilon"].values*100, grp["DeltaC_mean_pct"].values, grp["DeltaC_se"].values, label=k)
    ax.legend(title="Perturbation kind")
    ax.set_xlabel("ε (perturbation level(%))")
    ax.set_title("Relative cost difference(%) vs epsilon")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
# ----------------- batch runner -----------------

def estimation_error_batch_run(
    n_runs: int,
    base_config: dict = BASE_CONFIG,
    epsilons: List[float]= EPSILONS,
    cost_params: dict= COSTS,
    n_jobs: int = -1,
    out_dir: str = "./outputs/estimation_error",
    make_plots: bool = True,
):
    start = time.time()
    os.makedirs(out_dir, exist_ok=True)

    runs = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(estimation_error_single_run)(r, base_config, epsilons, cost_params)
        for r in range(n_runs)
    )
    df = pd.concat(runs, ignore_index=True)

    # Save full panel
    path_full = os.path.join(out_dir, "estimation_error_results.csv")
    df.to_csv(path_full, index=False)

    # -------- summary 1: by epsilon & kind (for debugging / completeness) --------
    def _se(x: pd.Series) -> float:
        n = len(x)
        return float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    summary_by_eps = (
        df.groupby(["kind", "epsilon"], dropna=False)
          .agg(
              C_true_mean=("C_true", "mean"),
              C_est_mean=("C_est", "mean"),
              DeltaC_mean_pct=("DeltaC_pct", "mean"),
              DeltaC_se=("DeltaC_pct", _se),
                H_est_mean=("H_est", "mean"),
                S_est_mean=("S_est", "mean"),
              rmse_mu_mean=("rmse_mu", "mean"),
              mape_mu_mean=("mape_mu", "mean"),
              r2_mean=("r2", "mean"),
          )
          .reset_index()
    )

    summary_by_eps_path = os.path.join(out_dir, "estimation_error_summary_by_epsilon.csv")
    summary_by_eps.to_csv(summary_by_eps_path, index=False)

    # -------- summary 2: cost vs RMSE (binning over RMSE to get a smooth curve) --------
    df_mu = df[df["kind"].isin(["A_only", "B_only", "Sigma_only","Combined"]) & df["rmse_mu"].notna()].copy()
    if not df_mu.empty:
        # Use global quantile bins for interpretability; duplicates='drop' to be robust
        try:
            df_mu["rmse_bin"], bins = pd.qcut(df_mu["rmse_mu"], q=6, retbins=True, duplicates="drop")
        except ValueError:
            # Fallback: simple fixed bins
            bins = np.array([0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 1.00])
            df_mu["rmse_bin"] = pd.cut(df_mu["rmse_mu"], bins=bins, include_lowest=True)

        summary_mu_bins = (
            df_mu.groupby(["kind", "rmse_bin"], observed=True)
                 .agg(
                     DeltaC_mean_pct=("DeltaC_pct", "mean"),
                     DeltaC_se=("DeltaC_pct", _se),
                     RMSE_mean=("rmse_mu", "mean"),
                 )
                 .reset_index()
                 .sort_values(["kind", "RMSE_mean"])
        )
    else:
        summary_mu_bins = pd.DataFrame(columns=["kind", "rmse_bin", "DeltaC_mean_pct", "DeltaC_se", "RMSE_mean"]) 

    path_mu_bins = os.path.join(out_dir, "estimation_error_sensitivity_mu_bins.csv")
    summary_mu_bins.to_csv(path_mu_bins, index=False)

    # -------- summary 3: cost vs variance error (Sigma_only) --------
    df_var = df[df["kind"] == "Sigma_only"].copy()
    if not df_var.empty:
        try:
            df_var["var_bin"], vbins = pd.qcut(df_var["r2"], q=6, retbins=True, duplicates="drop")
        except ValueError:
            vbins = np.array([0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.00])
            df_var["var_bin"] = pd.cut(df_var["r2"], bins=vbins, include_lowest=True)

        summary_var_bins = (
            df_var.groupby(["var_bin"], observed=True)
                  .agg(
                      DeltaC_mean_pct=("DeltaC_pct", "mean"),
                      DeltaC_se=("DeltaC_pct", _se),
                      r2_mean=("r2", "mean"),
                  )
                  .reset_index()
                  .sort_values(["r2_mean"]) 
        )
    else:
        summary_var_bins = pd.DataFrame(columns=["var_bin", "DeltaC_mean_pct", "DeltaC_se", "r2_mean"]) 

    path_var_bins = os.path.join(out_dir, "estimation_error_sensitivity_var_bins.csv")
    summary_var_bins.to_csv(path_var_bins, index=False)

    # -------- optional: figures --------
    if make_plots:
        if not summary_mu_bins.empty:
            plot_cost_vs_rmse(summary_mu_bins, os.path.join(out_dir, "fig_cost_vs_rmse.png"))
        if not summary_var_bins.empty:
            plot_cost_vs_var(summary_var_bins, os.path.join(out_dir, "fig_cost_vs_var.png"))
        if not summary_by_eps.empty:
            plot_cost_vs_eps(summary_by_eps, os.path.join(out_dir, "fig_cost_vs_eps.png"))
        
    elapsed = time.time() - start
    print(f"Saved: {path_full}")
    print(f"Saved: {summary_by_eps_path}")
    print(f"Saved: {path_mu_bins}")
    print(f"Saved: {path_var_bins}")
    if make_plots:
        if not summary_mu_bins.empty:
            print(f"Saved: {os.path.join(out_dir, 'fig_cost_vs_rmse.png')}")
        if not summary_var_bins.empty:
            print(f"Saved: {os.path.join(out_dir, 'fig_cost_vs_var.png')}")
    print(f"Elapsed: {elapsed:.1f}s")

    return df, summary_by_eps, summary_mu_bins, summary_var_bins


if __name__ == "__main__":
    # Example run (adjust n_runs to taste)
    df, s_eps, s_mu, s_var = estimation_error_batch_run(
        n_runs=50,              # start small; raise to 200 for paper
        base_config=BASE_CONFIG,
        epsilons=EPSILONS,
        cost_params=COSTS,
        n_jobs=-1,
        out_dir="./outputs/estimation_error",
        make_plots=True,
    )
