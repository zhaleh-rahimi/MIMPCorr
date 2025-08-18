import os
import time
import math
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from util.helper_func import _agg_ci,epsilon_cap_from_config
from data_prep.generate_varma_process import varma_data_generator
from inventory_models.policy_optimization import InventoryOptimizer

# ------------------------- configuration -------------------------

BASE_CONFIG = {
    "time_steps": 500,
    "num_products": 2,
    "model_order": [1, 1],  
    "min_demand": 10,
    "noise_level": 1,
    "max_rho": 0.80,
    "alpha": 0.10,
}

COSTS = {
    "holding_cost": [10.0 for _ in range(BASE_CONFIG["num_products"])],
    "shortage_cost": [50.0 for _ in range(BASE_CONFIG["num_products"])],
}
# Levels of parameter perturbation (controls uncertainty magnitude)
info = epsilon_cap_from_config(BASE_CONFIG, seed=0, title_contains="High Dependence", safety=0.5)
eps_rng = min(info["eps_safe"], 0.1)  # Cap for perturbation levels
EPSILONS =np.linspace(-eps_rng, eps_rng, 6).tolist()
# EPSILONS = [-0.05,-0.03, -0.01, 0.0, 0.01, 0.03, 0.05]  # Perturbation levels

# ------------------------- utilities -------------------------

def compute_mu_from_params(
    U: np.ndarray,
    A_list: List[np.ndarray],
    B_list: List[np.ndarray],
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
    mu = process.get_conditional_mean(A_list, B_list, U, title = "mu")
    return mu


def order_up_to_cost(
    demand: np.ndarray,
    mu: np.ndarray,
    sigma_vec: np.ndarray,
    cost_params: dict,
) -> Tuple[float, float, float]:
    """Compute policy from μ and σ and evaluate cost via InventoryOptimizer."""
    inv = InventoryOptimizer(cost_params)
    policy = inv.compute_policy(mu, sigma_vec)
    total, htot, stot = inv.evaluate_policy(demand, policy)
    return total, htot, stot


def error_stats(d: np.ndarray, d_hat: np.ndarray) -> Tuple[float, float, float, float]:
    """Return RMSE, mean bias, and MAPE of the error."""
    diff = d - d_hat
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))
    mape = float(np.mean(np.abs(diff) / np.maximum(np.abs(d), 1e-8)))*100
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
    Returns long-form DataFrame with cost deltas and error diagnostics.
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
    
    title = f"Items={k}, p={p}, q={q}, High Dependence"

    Y = np.asarray(data_fit[title], dtype=float)            # realized demand
    mu_true_all = np.asarray(data_gen[title], dtype=float)  # true conditional means
    U = Y - mu_true_all                                     # innovations
       
    start_test = max(burn + 1, train_T)
    end_test = min(T, start_test + test_T)
    if end_test - start_test <= 0:
        raise ValueError(f"Empty test window: T={T}, start={start_test}, end={end_test}")

    # Split data into training and test sets
    demand_test = Y[start_test:end_test]
    mu_true_test = mu_true_all[start_test:end_test]

    # True parameters from generator
    row = gen.get_scenario_by_title(title)
    A_true = [np.array(A, dtype=float) for A in row["AR Coefficients"]]
    B_true = [np.array(B, dtype=float) for B in row["MA Coefficients"]]
    Sigma_true = getattr(gen, "sigma_u", None)   
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
        return A_hat

    def make_B_hat(eps: float) -> List[np.ndarray]:
        if len(B_true) == 0:
            return []
        if eps == 0.0:
            return [B.copy() for B in B_true]
        noise = [rng.normal(size=B.shape) for B in B_true]
        B_hat = [B_true[i] *(1+ eps * noise[i]) for i in range(len(B_true))]
        return B_hat

    def make_Sigma_hat(eps: float) -> np.ndarray:
        if eps == 0.0:
            return Sigma_true.copy()
        noise = rng.normal(size=Sigma_true.shape)
        return Sigma_true *(1 + eps * noise)

    for eps in epsilons:
        # A-only
        A_hat = make_A_hat(eps)
        mu_A_all = compute_mu_from_params(U, A_hat, B_true, base_config=base_config)
        if mu_A_all is None:
            print(f"Failed to compute mu_A_all for eps={eps}")
            continue
        mu_A = mu_A_all[start_test:end_test]
        C_A, H_A, S_A = order_up_to_cost(demand_test, mu_A, sigma_true_vec, cost_params)
        rmse_A, bias_A, mape_A, r2_A = error_stats(demand_test, mu_A)
        
        # B-only
        if len(B_true) > 0:
            B_hat = make_B_hat(eps)
            mu_B_all = compute_mu_from_params(U, A_true, B_hat, base_config=base_config)
            if mu_B_all is None:
                print(f"Failed to compute mu_B_all for eps={eps}")
                continue
            mu_B = mu_B_all[start_test:end_test]
            C_B, H_B, S_B = order_up_to_cost(demand_test, mu_B, sigma_true_vec, cost_params)
            rmse_B, bias_B, mape_B,r2_B=error_stats(demand_test, mu_B)
        else:
            C_B = H_B = S_B = np.nan
            rmse_B = bias_B = mape_B = np.nan

        # Sigma-only (variance misspecification with true mean)
        Sigma_hat = make_Sigma_hat(eps)
        sigma_hat_vec = np.sqrt(np.clip(np.diag(Sigma_hat), 1e-12, None))
        C_S, H_S, S_S = order_up_to_cost(demand_test, mu_true_test, sigma_hat_vec, cost_params)
        rmse_S, bias_S, mape_S,r2_S = error_stats(demand_test, mu_true_test)

        # Combined (mean + variance misspecification)
        A_hat2 = make_A_hat(eps)
        B_hat2 = make_B_hat(eps)
        mu_AB_all = compute_mu_from_params(U, A_hat2, B_hat2, base_config=base_config)
        if mu_AB_all is None:
            print(f"Failed to compute mu_AB_all for eps={eps}")
            continue
        mu_AB = mu_AB_all[start_test:end_test]
        C_AB, H_AB, S_AB = order_up_to_cost(demand_test, mu_AB, sigma_hat_vec, cost_params)
        rmse_AB, bias_AB, mape_AB,r2_AB = error_stats(demand_test, mu_AB)

        # Collect
        def rec(kind: str, C: float, H: float, S: float, rmse: float, bias: float, mape: float, r2: Optional[float] = None):
            return dict(
                run_id=run_id,
                epsilon=eps,
                kind=kind,
                C_true=C_true,
                C_est=C,
                DeltaC_pct= 100.0 * (C - C_true) / max(C_true, 1e-12),  # % change vs baseline cost
                H_true=H_true,
                H_est=H,
                S_true=S_true,
                S_est=S,
                rmse_mu=rmse,
                bias_mu=bias,
                mape_mu=mape,
                r2 = r2
            )
        records.append(rec("True", C_true, H_true, S_true, 0, 0, 0, 1))
        records.append(rec("A_only", C_A, H_A, S_A, rmse_A, bias_A, mape_A, r2_A))
        records.append(rec("Sigma_only", C_S, H_S, S_S, rmse_S, bias_S, mape_S,r2_S))
        records.append(rec("Combined", C_AB, H_AB, S_AB, rmse_AB, bias_AB, mape_AB, r2_AB))
        if not math.isnan(C_B):
            records.append(rec("B_only", C_B, H_B, S_B, rmse_B, bias_B, mape_B, r2_B))

    return pd.DataFrame.from_records(records)


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

    # -------- summary 1: by epsilon & kind with 95% CIs --------
    df_eps = df[df["kind"].isin(["A_only", "B_only", "Sigma_only", "Combined"]) & df["epsilon"].notna()].copy()
    grp = df_eps.groupby(["kind", "epsilon"], dropna=False)

    rows = []
    for keys, g in grp:
        rec = {
            "kind": keys[0],
            "epsilon": keys[1],
        }
        # Core metrics with CIs
        rec.update(_agg_ci(g, "C_true", "C_true"))
        rec.update(_agg_ci(g, "C_est", "C_est"))
        rec.update(_agg_ci(g, "DeltaC_pct", "DeltaC_pct"))
        rec.update(_agg_ci(g, "rmse_mu", "RMSE"))
        rec.update(_agg_ci(g, "mape_mu", "MAPE"))
        # Optional extras:
        
        rec.update(_agg_ci(g, "H_est", "H_est"))
        rec.update(_agg_ci(g, "H_true", "H_true"))
        rec.update(_agg_ci(g, "S_est", "S_est"))
        rec.update(_agg_ci(g, "S_true", "S_true"))
        rec.update(_agg_ci(g, "r2", "R2"))
        rows.append(rec)

    summary_by_eps = pd.DataFrame(rows).sort_values(["kind", "epsilon"]).reset_index(drop=True)

    summary_by_eps_path = os.path.join(out_dir, "estimation_error_summary_by_epsilon_CI.csv")
    summary_by_eps.to_csv(summary_by_eps_path, index=False)


    # -------- summary 2: cost vs MAPE (with 95% CI) --------
    df_mu = df[df["kind"].isin(["A_only", "B_only", "Sigma_only", "Combined"]) & df["mape_mu"].notna()].copy()
    if not df_mu.empty:
        try:
            df_mu["mape_bin"], bins = pd.qcut(df_mu["mape_mu"], q=6, retbins=True, duplicates="drop")
        except ValueError:
            bins = np.array([0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 1.00])
            df_mu["mape_bin"] = pd.cut(df_mu["mape_mu"], bins=bins, include_lowest=True)

        # Aggregate using CI helpers
        rows_mu = []
        for (k, b), g in df_mu.groupby(["kind", "mape_bin"], observed=True):
            rec = {"kind": k, "mape_bin": b}
            # Outcome on y-axis
            rec.update(_agg_ci(g, "DeltaC_pct", "DeltaC_pct"))
            # X plotting coordinate (bin center via mean)
            rec.update(_agg_ci(g, "mape_mu", "MAPE"))
            rows_mu.append(rec)

        summary_mu_bins = (pd.DataFrame(rows_mu)
                               .sort_values(["kind", "MAPE_mean"])
                               .reset_index(drop=True))
    else:
        summary_mu_bins = pd.DataFrame(columns=[
            "kind", "mape_bin",
            "DeltaC_pct_mean", "DeltaC_pct_se", "DeltaC_pct_ci_low", "DeltaC_pct_ci_high",
            "MAPE_mean", "MAPE_se", "MAPE_ci_low", "MAPE_ci_high"
        ])

    path_mu_bins = os.path.join(out_dir, "estimation_error_sensitivity_mu_bins.csv")
    summary_mu_bins.to_csv(path_mu_bins, index=False)
    # -------- summary 3: MAPE vs epsilon (with 95% CI) --------
    df_eps_mu = df[df["kind"].isin(["A_only", "B_only", "Sigma_only", "Combined"]) & df["epsilon"].notna() & df["mape_mu"].notna()].copy()
    if not df_eps_mu.empty:
        try:
            df_mu["mape_bin"], bins = pd.qcut(df_mu["mape_mu"], q=6, retbins=True, duplicates="drop")
        except ValueError:
            bins = np.array([0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 1.00])
            df_mu["mape_bin"] = pd.cut(df_mu["mape_mu"], bins=bins, include_lowest=True)

        rows_eps_mu = []
        for (k, eps), g in df_eps_mu.groupby(["kind", "epsilon"], observed=True):
            rec = {"kind": k, "epsilon": eps}
            # Outcome on y-axis
            rec.update(_agg_ci(g, "mape_mu", "MAPE"))
            # X plotting coordinate (epsilon)
            rec.update(_agg_ci(g, "epsilon", "epsilon"))
            rows_eps_mu.append(rec)

        summary_mape_by_eps = (pd.DataFrame(rows_eps_mu)
                               .sort_values(["kind", "epsilon"])
                               .reset_index(drop=True))
    else:
        summary_mape_by_eps = pd.DataFrame(columns=[
            "kind", "epsilon",
            "MAPE_mean", "MAPE_se", "MAPE_ci_low", "MAPE_ci_high"
        ])  
    
    path_eps_mu = os.path.join(out_dir, "estimation_error_sensitivity_mape_by_epsilon.csv")
    summary_mape_by_eps.to_csv(path_eps_mu, index=False)
    # -------- figures --------
    if make_plots:
        if not summary_mu_bins.empty:
            plot_cost_vs_mape(summary_mu_bins, os.path.join(out_dir, "fig_cost_vs_mape.png"))
        if not summary_by_eps.empty:
            plot_cost_vs_eps(summary_by_eps, os.path.join(out_dir, "fig_cost_vs_eps.png"))
        if not summary_mape_by_eps.empty:
            plot_mape_vs_eps(summary_mape_by_eps, os.path.join(out_dir, "fig_mape_vs_eps.png"))
    elapsed = time.time() - start
    print(f"Saved: {path_full}")
    print(f"Saved: {summary_by_eps_path}")
    print(f"Saved: {path_mu_bins}")
    
    if make_plots:
        if not summary_mu_bins.empty:
            print(f"Saved: {os.path.join(out_dir, 'fig_cost_vs_mape.png')}")
        print(f"Elapsed: {elapsed:.1f}s")

    return df, summary_by_eps, summary_mu_bins

# ----------------- plotting helpers -----------------

def _errbar(ax, x, y, yerr, label):
    ax.errorbar(x, y, yerr=yerr, fmt="-o", label=label, capsize=3)
    ax.set_ylabel("Relative cost difference (%)")
    ax.grid(True, alpha=0.3)

def plot_cost_vs_mape(summary_mu_bins: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, grp in summary_mu_bins.groupby("kind"):
        _errbar(
            ax,
            grp["MAPE_mean"].values,
            grp["DeltaC_pct_mean"].values,
            (grp["DeltaC_pct_mean"].values - grp["DeltaC_pct_ci_low"].values,  # asymmetric OK
             grp["DeltaC_pct_ci_high"].values - grp["DeltaC_pct_mean"].values),
            label=k
        )
    ax.legend(title="Perturbation kind")
    ax.set_xlabel("MAPE of forecast")
    ax.set_title("Relative cost difference (%) vs MAPE of forecast (95% CI)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_cost_vs_eps(summary_by_eps: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for k, grp in summary_by_eps.groupby("kind"):
        _errbar(
            ax,
            grp["epsilon"].values * 100,
            grp["DeltaC_pct_mean"].values,
            (grp["DeltaC_pct_mean"].values - grp["DeltaC_pct_ci_low"].values,
             grp["DeltaC_pct_ci_high"].values - grp["DeltaC_pct_mean"].values),
            label=k
        )
    ax.legend(title="Perturbation kind")
    ax.set_xlabel("ε (perturbation level, %)")
    ax.set_title("Relative cost difference (%) vs ε (95% CI)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mape_vs_eps(summary_mape_by_eps: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for k, grp in summary_mape_by_eps.groupby("kind"):
        _errbar(
            ax,
            grp["epsilon"].values * 100,
            grp["MAPE_mean"].values,
            (grp["MAPE_mean"].values - grp["MAPE_ci_low"].values,
             grp["MAPE_ci_high"].values - grp["MAPE_mean"].values),
            label=k
        )
    ax.legend(title="Perturbation kind")
    ax.set_xlabel("ε (perturbation level, %)")
    ax.set_ylabel("Forecast MAPE (%)")
    ax.set_title("Forecast MAPE (%) vs ε (95% CI)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
