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
from util.helper_func import full_summary_table
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
            ("cost_varma_est",  "Cest"),
            ("diff_pct_signed", "DiffPct"),
            ("diff_pct_abs",    "AbsDiff"),
            ("ratio_est_over_true", "Ratio"),
        ]:
            s = ci_stats(g[col])
            rec.update({f"{pref}_{m}": s[m] for m in ("n","mean","se","ci_low","ci_high")})
        rows.append(rec)

    out = pd.DataFrame(rows)
    return out.sort_values(key_cols).reset_index(drop=True) if key_cols else out

def add_relative_cols(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-12
    denom = np.maximum(df["cost_varma_true"].astype(float), eps)
    df = df.copy()
    df["diff_abs"] = df["cost_varma_est"] - df["cost_varma_true"]
    df["diff_pct_signed"] = 100.0 * df["diff_abs"] / denom
    df["diff_pct_abs"] = df["diff_pct_signed"].abs()
    df["ratio_est_over_true"] = df["cost_varma_est"] / denom
    return df

def slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))

def savefig(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig); print(f"Saved: {path}")

# ---------- core single-run ----------

def delta_c_est_single_run(run_id: int, config: dict):
    """
    Compare multi-period costs (VARMA est vs VARMA known) for cost pairs where H == S.
    Returns rows: [run_id, title, H, S, C_est, C_true, signed%, abs%].
    """
    out = []
    try:
        gen = varma_data_generator(config=config, seed=run_id)
        data_fit, data_gen = gen.generate_scenarios()
        model_order = config["model_order"]
        min_y = config["min_demand"]
        cost_params = config["cost_params"]

        for title, df_title in data_fit.items():
            if "High Dependence" not in title: 
                continue
            df = {title: df_title}

            for idx, h in enumerate(cost_params["holding_cost"]):
                s = cost_params["shortage_cost"][idx]
                if 5*h[0] != s[0]: 
                    continue

                costs = {k: v[idx] for k, v in cost_params.items() if len(v) > idx}
                try:
                    _, cost, _, *_ = evaluate_varma_order_policy(df, costs, model_order, data_gen, min_y)
                    C_est = float(cost["VARMA"][title]); C_true = float(cost["VARMA_known"][title])
                    denom = max(C_true, 1e-12)
                    d_signed = 100.0 * (C_est - C_true) / denom
                    out.append([run_id, title, h[0], s[0], C_est, C_true, d_signed, abs(d_signed)])
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
            "cost_varma_est","cost_varma_true","diff_pct_signed","diff_pct_abs"]
    df = add_relative_cols(pd.DataFrame(rows, columns=cols))

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
    # panel_csv = f"outputs/csv/{tag}-panel.csv"; df.to_csv(panel_csv, index=False)
    by_tc_csv = f"outputs/csv/{tag}-by_title_cost_CI.csv"; by_title_cost.to_csv(by_tc_csv, index=False)
    
    legacy_csv = f"outputs/csv/{func.__name__}(x{n_run}runs)-Items={config['num_products']}-p={config['model_order'][0]}-q={config['model_order'][1]}.csv"
    full_summary_table(
        df.rename(columns={"diff_pct_abs":"abs_percent_diff"}),
        ["title","holding_unit_cost","shortage_unit_cost"]
    ).to_csv(legacy_csv, index=False)

    # ---- plots ----
    figs_tag = f"outputs/figures/{tag}"

    # 1) Estimated vs True scatter + 45°
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(df["cost_varma_true"], df["cost_varma_est"], alpha=0.65)
    m = max(df["cost_varma_true"].max(), df["cost_varma_est"].max()); m = 1.05*m if np.isfinite(m) else 1.0
    ax.plot([0,m],[0,m],"--"); ax.set_xlim(0,m); ax.set_ylim(0,m)
    ax.set_xlabel("True cost (VARMA_known)"); ax.set_ylabel("Estimated cost (VARMA)")
    ax.set_title("Estimated vs True cost"); ax.grid(alpha=0.3)
    savefig(fig, f"{figs_tag}-scatter_true_vs_est.png")

    # print("\nSaved CSVs:\n ", panel_csv, "\n ", by_tc_csv, "\n ", by_t_csv, "\n ", by_c_csv, "\n ", legacy_csv)
    print(f"Elapsed: {time.time()-t0:.2f}s")

def run_config_directory(func, dir_path: str, n_run: int):
    for fn in os.listdir(dir_path):
        if fn.lower().endswith(".json"):
            run_config_file(func, os.path.join(dir_path, fn), n_run)

