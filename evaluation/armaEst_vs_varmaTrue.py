# delta_c_est_ci_report.py
# Compact pipeline: run sims, compute 95% CI summaries, save CSVs & plots.

import os, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError

from data_prep.generate_varma_process import varma_data_generator
from evaluation.evaluate_varma_order_policy import evaluate_varma_order_policy
from util.helper_func import _mean_se_ci as ci_stats

# ---------- helpers ----------

def summarize_with_ci(df: pd.DataFrame, keys=None) -> pd.DataFrame:
    """CI summary over whole df if keys is None/empty; else group by keys."""
    if not keys:
        groups = [((), df)]
        key_cols = []
    else:
        key_cols = [k for k in keys if k in df.columns]
        groups = list(df.groupby(key_cols, dropna=False)) if key_cols else [((), df)]

    rows = []
    for k, g in groups:
        rec = {}
        if key_cols:
            if isinstance(k, tuple): rec.update(dict(zip(key_cols, k)))
            else: rec[key_cols[0]] = k
        for col, pref in [
            ("cost_varma_true", "Ctrue"),
            ("cost_arma_est",  "Cest"),
            ("diff_pct_signed", "DiffPct"),
            ("diff_pct_abs",    "AbsDiff"),
            ("mape_est",        "mape_est"),
            ("mape_true",       "mape_true"),
            ("diff_pct_signed_shortage", "DiffPctShortage"),
            ("diff_pct_signed_holding", "DiffPctHolding")
        ]:
            s = ci_stats(g[col])
            rec.update({f"{pref}_{m}": s[m] for m in ("n","mean","interval","se","ci_low","ci_high")})
        rows.append(rec)

    out = pd.DataFrame(rows)
    return out.sort_values(key_cols).reset_index(drop=True) if key_cols else out


def slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))

def savefig(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig); print(f"Saved: {path}")

# ---------- core single-run ----------

def armaEst_varmaTrue_single_run(run_id: int, config: dict):
    """
    Compare multi-period costs (ARMA est vs VARMA known) for cost pairs where H == S.
    Returns rows: [run_id, title, H, S, C_est, C_true, signed%, abs%].
    """
    out = []
    try:
        gen = varma_data_generator(config=config, seed=run_id)
        data_fit, data_gen = gen.generate_scenarios()
        model_order = config["model_order"]
        min_y = config["min_demand"]
        k = config["num_products"]
        cost_params = {
            "holding_cost": [[10]*k,[10]*k,[10]*k],
            "shortage_cost": [[2]*k,[10]*k,[50]*k],
        }

        for title, df_title in data_fit.items():
            if "High Dependence" not in title: 
                continue
            df = {title: df_title}

            for idx, h in enumerate(cost_params["holding_cost"]):
                s = cost_params["shortage_cost"][idx]
                # if h != s: 
                #     continue

                costs = {k: v[idx] for k, v in cost_params.items() if len(v) > idx}
                try:
                    _, cost, _, forecast_performance,h_cost,s_cost = evaluate_varma_order_policy(df, costs, model_order, data_gen, min_y)
                    C_est = float(cost["ARIMA"][title]); C_true = float(cost["VARMA_known"][title])
                    C_s_est = float(s_cost["ARIMA"][title])
                    C_h_est = float(h_cost["ARIMA"][title])
                    C_s_true = float(s_cost["VARMA_known"][title])
                    C_h_true = float(h_cost["VARMA_known"][title])
                    mape_est = float(forecast_performance["ARIMA"][title]["mape"])
                    mape_true = float(forecast_performance["VARMA_known"][title]["mape"])
                    denom = max(C_true, 1e-12)
                    denom_s = max(C_s_true, 1e-12)
                    denom_h = max(C_h_true, 1e-12)
                    d_signed = 100.0 * (C_est - C_true) / denom
                    d_signed_s = 100.0 * (C_s_est - C_s_true) / denom_s
                    d_signed_h = 100.0 * (C_h_est - C_h_true) / denom_h
                    out.append([run_id, title, h[0], s[0], C_est, C_true, d_signed, abs(d_signed),d_signed_s, d_signed_h,mape_est, mape_true])
                except LinAlgError:
                    # skip this (run, cost) if policy evaluation failed
                    continue
    except LinAlgError:
        # skip this run if data generation failed
        pass
    return out

# ---------- runner (file & directory) ----------

def run_config_file(func, file_path: str, n_run: int):
    with open(file_path, "r") as f: config = json.load(f)
    os.makedirs("outputs/csv", exist_ok=True); os.makedirs("outputs/figures", exist_ok=True)
    tag = f"{func.__name__}(x{n_run}runs)-Items={config['num_products']}-p={config['model_order'][0]}-q={config['model_order'][1]}"
    t0 = time.time()

    rows = Parallel(n_jobs=-1)(delayed(func)(r, config) for r in range(n_run))
    rows = [r for sub in rows for r in sub]
    if not rows: 
        print("No valid results."); return

    cols = ["run_id","title","holding_unit_cost","shortage_unit_cost",
            "cost_arma_est","cost_varma_true","diff_pct_signed","diff_pct_abs","diff_pct_signed_shortage","diff_pct_signed_holding","mape_est","mape_true"]
    df = pd.DataFrame(rows, columns=cols) 
    # ---- CI summaries ----
    overall = summarize_with_ci(df)  # whole df
    by_title_cost = summarize_with_ci(df, ["title","holding_unit_cost","shortage_unit_cost"])
    
    # Quick headline
    o = overall.iloc[0]
    print(
        f"\nAcross {int(o['Ctrue_n'])} sims: "
        f"mean signed Δ% = {o['DiffPct_mean']:.2f}% [{o['DiffPct_ci_low']:.2f}%, {o['DiffPct_ci_high']:.2f}%], "
        f"mean |Δ%| = {o['AbsDiff_mean']:.2f}% [{o['AbsDiff_ci_low']:.2f}%, {o['AbsDiff_ci_high']:.2f}%]."
    )

    # ---- CSVs ----
    by_tc_csv = f"outputs/csv/{tag}-by_title_cost_CI.csv"; 
    by_title_cost.to_csv(by_tc_csv, index=False)
    
    # ---- plots ----
    figs_tag = f"outputs/figures/{tag}"

    # 1) Estimated vs True scatter + 45°
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(df["cost_varma_true"], df["cost_arma_est"], alpha=0.65)
    m = max(df["cost_varma_true"].max(), df["cost_arma_est"].max()); m = 1.05*m if np.isfinite(m) else 1.0
    ax.plot([0,m],[0,m],"--"); ax.set_xlim(0,m); ax.set_ylim(0,m)
    ax.set_xlabel("True cost (VARMA_known)"); ax.set_ylabel("Estimated cost (ARMA)")
    ax.set_title("Estimated vs True cost"); ax.grid(alpha=0.3)
    savefig(fig, f"{figs_tag}-scatter_true_vs_est.png")


    print(f"Elapsed: {time.time()-t0:.2f}s")

def run_config_directory(func, dir_path: str, n_run: int):
    for fn in os.listdir(dir_path):
        if fn.lower().endswith(".json"):
            run_config_file(func, os.path.join(dir_path, fn), n_run)

