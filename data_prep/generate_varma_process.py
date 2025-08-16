"""
Created on Thu Jan  9 11:04:31 2025
@author: Zhaleh
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from util.stat_tests import (
    check_stationarity_AR,
    check_invertibility_MA,
    perform_adfuller_test,
)


class varma_data_generator:
    

    def __init__(self, config, seed=None):
        self.scenario_results = pd.DataFrame()  # store coefficients and autocorrelation matrices

        # Config with safe defaults
        self.steps = config.get("time_steps", 500)
        self.k = config.get("num_products", 2)
        self.sigma_base = config.get("noise_level", 1.0)
        model_order = config.get("model_order", (1, 0))
        self.p = int(model_order[0])
        self.q = int(model_order[1])
        self.min_y = config.get("min_demand", 10)
        self.max_rho = config.get("max_rho", 0.8)
        self.alpha = config.get("alpha", 0.3)

        # Noise covariance
        self.sigma_u = np.eye(self.k) * self.sigma_base
        self.base_coeff = self.sigma_base / max(self.min_y, 1e-12)

        # RNG seed: set global RNG once if provided; store for later
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

    
    def generate_scenarios(self):
        """Wrapper to Generate Scenarios.
        Returns (results_dict, predictions_dict)
        """
        results = {}
        results_pred = {}
        k, p, q = self.k, self.p, self.q

        # Base noise shared across scenarios to make them comparable (like your v2)
        U_base = self.generate_uncorrelated_noise(np.zeros(k), self.seed)

        # High Dependence Scenario
        cov_high = self.generate_high_dependence_cov(self.max_rho, self.alpha)
        phi_matrices_high = self.yule_walker_multivariate(cov_high)
        theta_matrices = self.generate_theta_matrices(k, q)
        title_high = f"Items={k}, p={p}, q={q}, High Dependence"

        Y_high, pred_high = self.generate_varma_adjusted(
            phi_matrices_high, theta_matrices, U_base, title_high
        )
        results[title_high] = Y_high
        results_pred[title_high] = pred_high

        # Medium/Low
        reduction_factors = {"Medium Dependence": 0.5, "Low Dependence": 0.05}
        for strength_name, reduction_factor in reduction_factors.items():
            cov_matrices = [self.reduce_dependence(cov, reduction_factor) for cov in cov_high]
            phi_matrices = self.yule_walker_multivariate(cov_matrices)
            title = f"Items={k}, p={p}, q={q}, {strength_name}"
            Y, pred = self.generate_varma_adjusted(phi_matrices, theta_matrices, U_base, title)
            results[title] = Y
            results_pred[title] = pred

        return results, results_pred

    # ---------- Model building blocks ----------
    def generate_high_dependence_cov(self, max_rho, alpha, reverse=False):
        """Construct a high-dependence covariance structure across lags (p)."""
        phi1 = (
            np.eye(self.k) * 0.1 * self.sigma_base
            + np.ones((self.k, self.k)) * max_rho * self.sigma_base
            - np.eye(self.k) * max_rho * self.sigma_base
        )
        idx_iter = reversed(range(self.p)) if reverse else range(self.p)
        matrix = [phi1 * (max_rho * (alpha ** i)) for i in idx_iter]
        return np.array(matrix)

    def reduce_dependence(self, base_matrix, factor):
        reduced_matrix = base_matrix * factor
        np.fill_diagonal(reduced_matrix, np.diag(base_matrix))  # keep diagonal strength
        return reduced_matrix

    def generate_varma_adjusted(self, ar_matrices, ma_matrices, U, title):
        """Generate a VARMA process with coefficient checks and ADF-based stationarity enforcement."""
        # Adjust coefficients (scale until conditions hold)
        ar_matrices = self.adjust_coefficients(ar_matrices, check_stationarity_AR)
        ma_matrices = (
            self.adjust_coefficients(ma_matrices, check_invertibility_MA) if self.q > 0 else []
        )

        # Generate
        Y, pred = self.generate_varma(ar_matrices, ma_matrices, U)

        # Check invertibility for VMA
        is_invertible = check_invertibility_MA(ma_matrices) if self.q > 0 else True

        # Enforce stationarity: regenerate with fresh noise each attempt 
        Y, pred, ar_matrices, is_stationary = self.enforce_stationarity_adf(
            Y, pred, ar_matrices, ma_matrices, U, title
        )

        # ADF test reporting
        perform_adfuller_test(Y, title)

        # Autocorrelation matrices
        autocor_matrices = self.compute_autocorrelation_matrix(Y, max_lag=len(ar_matrices))

        # Enforce positivity
        Y, pred = self.enforce_positivity(Y, pred)

        # Persist
        self.save_scenario_results(
            title, ar_matrices, ma_matrices, is_stationary, is_invertible, autocor_matrices
        )

        return Y, pred

    def generate_varma(self, ar_matrices, ma_matrices, U):
        """Generate VARMA process (output: Y, pred = conditional mean)."""
        k, p, q, steps = self.k, self.p, self.q, self.steps
        Y = np.zeros((steps, k))
        pred = np.zeros((steps, k))

        Y[: (max(p, q))] = U[: (max(p, q))]
        for t in range(max(p, q), steps):
            ar_part = sum(ar_matrices[i] @ Y[t - i - 1] for i in range(p)) if p > 0 else 0
            ma_part = sum(ma_matrices[j] @ U[t - j - 1] for j in range(q)) if q > 0 else 0
            pred[t] = ar_part + ma_part
            Y[t] = pred[t] + U[t]
        return Y, pred

    def get_conditional_mean(self, ar_matrices, ma_matrices,U, title):
        """Generate a VARMA process with coefficient checks and ADF-based stationarity enforcement."""
        # Generate
        Y, pred = self.generate_varma(ar_matrices, ma_matrices, U)

        # Check invertibility for VMA
        is_invertible = check_invertibility_MA(ma_matrices) if self.q > 0 else True
        is_stationary = check_stationarity_AR(ar_matrices)
        if is_stationary and is_invertible:
            # Enforce positivity
            _, mu = self.enforce_positivity(Y, pred)
            return mu 
        
        return None

    def var_to_varma(self, var_coeffs):
        """Compute a finite VMA(q) representation from VAR(p) coefficients up to order q."""
        p, q = self.p, self.q
        k = var_coeffs[0].shape[0]
        identity_matrix = np.eye(k)

        psi_coeffs = [identity_matrix]
        for i in range(1, q + 1):
            psi_i = np.zeros((k, k))
            for j in range(1, min(i, p) + 1):
                psi_i += var_coeffs[j - 1] @ psi_coeffs[i - j]
            psi_coeffs.append(psi_i)
        vma_coeffs = psi_coeffs[1:]
        return var_coeffs, vma_coeffs

    def generate_theta_matrices(self, k, q):
        """Generate invertible MA coefficient matrices with one-time seeding."""
        ma_matrices = []
        # Seed ONCE (if provided) to make the set of thetas deterministic
        if self.seed is not None:
            np.random.seed(self.seed)
        for _ in range(q):
            theta = np.random.uniform(0.1, 0.4, (k, k))
            theta = self.adjust_coefficients([theta], check_invertibility_MA)
            ma_matrices.append(theta[0])
        return ma_matrices

    def yule_walker_multivariate(self, cov_matrices):
        """Compute AR coefficients using multivariate Yule–Walker equations."""
        n, p = self.k, self.p
        cov_matrices = np.append(self.sigma_u, cov_matrices).reshape((p + 1, n, n))
        R_blocks = [[cov_matrices[abs(i - j)] for j in range(p)] for i in range(p)]
        R = np.block(R_blocks)
        r = cov_matrices[1:].reshape((p * n, n))
        ar_flat = np.linalg.solve(R, r)
        return ar_flat.reshape((p, n, n))

    def compute_autocorrelation_matrix(self, Y, max_lag):
        """Autocorrelation matrices for lags 0..max_lag."""
        n_samples, n_variables = Y.shape
        autocov_matrices = []
        for lag in range(max_lag + 1):
            Gamma_k = np.zeros((n_variables, n_variables))
            for t in range(lag, n_samples):
                Gamma_k += np.outer(Y[t], Y[t - lag])
            Gamma_k /= max(n_samples - lag, 1)
            autocov_matrices.append(Gamma_k)
        Gamma_0 = autocov_matrices[0]
        autocorr_matrices = []
        D = np.diag(np.diag(Gamma_0))
        # numerical safety: avoid sqrt of zeros on diagonal
        D_safe = np.where(D > 0, D, 1e-12)
        D_inv_sqrt = np.linalg.inv(np.sqrt(D_safe))
        for Gamma_k in autocov_matrices:
            rho_k = D_inv_sqrt @ Gamma_k @ D_inv_sqrt
            autocorr_matrices.append(rho_k)
        return autocorr_matrices

    def adjust_coefficients(self, coefficients, check_func, scaling_factor=0.97, max_attempts=10**3):
        """Scale down coefficients until constraint passes or attempts run out."""
        attempts = 0
        adjusted = [c.copy() for c in coefficients] if len(coefficients) else []
        while len(adjusted) and (not check_func(adjusted)) and attempts < max_attempts:
            adjusted = [coef * scaling_factor for coef in adjusted]
            attempts += 1
        if attempts == max_attempts and len(adjusted) and (not check_func(adjusted)):
            raise ValueError("Unable to adjust coefficients to meet the condition.")
        return adjusted

    def generate_uncorrelated_noise(self, mean, seed):
        """Draw multivariate Gaussian noise and Ljung–Box filter for serial correlation."""
        cov, n_samples, n_dims = self.sigma_u, self.steps, self.k
        if seed is not None:
            np.random.seed(seed)
        noise = np.random.multivariate_normal(mean, cov, size=n_samples)
        for dim in range(n_dims):
            p_value = acorr_ljungbox(noise[:, dim], lags=[10], return_df=True)["lb_pvalue"].iloc[0]
            if p_value < 0.05:
                # v1-like: bump the seed to change the RNG stream deterministically
                next_seed = (seed + 1) if (seed is not None) else None
                return self.generate_uncorrelated_noise(mean, next_seed)
        return noise

    def enforce_stationarity_adf(
        self,
        Y,
        pred,
        ar_matrices,
        ma_matrics,
        U,
        title,
        scaling_factor=0.97,
        max_attempts=10**3,
    ):
        """Reduce AR magnitude until all series pass the ADF test. Regenerates noise each attempt."""
        attempts = 0
        while attempts < max_attempts:
            stationary = True
            for i in range(Y.shape[1]):
                result = adfuller(Y[:, i])
                if result[1] >= 0.05:  # Non-stationary
                    stationary = False
                    break
            if stationary:
                return Y, pred, ar_matrices, True

            # Scale down AR coefficients
            ar_matrices = [phi * scaling_factor for phi in ar_matrices]

            # Regenerate with fresh noise each attempt 
            U = self.generate_uncorrelated_noise(np.zeros(self.k), self.seed)
            Y, pred = self.generate_varma(ar_matrices, ma_matrics, U)
            attempts += 1

        print(f"Failed to enforce stationarity after {max_attempts} attempts for {title}.")
        return Y, pred, ar_matrices, False

    def enforce_positivity(self, Y, pred):
        """Pure upward shift by min_y"""
        shift = self.min_y
        return Y + shift, pred + shift

    def save_scenario_results(
        self, scenario, ar_matrices, ma_matrices, stationary, invertible, autocor_matrices
    ):
        result = {
            "Scenario": scenario,
            "AR Coefficients": ar_matrices,
            "MA Coefficients": ma_matrices,
            "Is Stationary": stationary,
            "Is Invertible": invertible,
            "Autocorrelation Lag 0": autocor_matrices[0],
            "Autocorrelation Lag 1": autocor_matrices[1] if len(autocor_matrices) > 1 else None,
            "Autocorrelation Lag 2": autocor_matrices[2] if len(autocor_matrices) > 2 else None,
            "Autocorrelation Lag 3": autocor_matrices[3] if len(autocor_matrices) > 3 else None,
            "Autocorrelation Lag 4": autocor_matrices[4] if len(autocor_matrices) > 4 else None,
        }
        self.scenario_results = pd.concat(
            [self.scenario_results, pd.DataFrame([result])], ignore_index=True
        )

    def get_scenario_results(self):
        return self.scenario_results

    def get_scenario_by_title(self, title, return_all=False):
        """Return row(s) where Scenario == title.
        If return_all=False, returns latest match as a dict (or None if not found).
        If return_all=True, returns list of dicts for all matches.
        """
        df = self.scenario_results
        matches = df.loc[df["Scenario"].astype(str) == str(title)]
        if matches.empty:
            return None
        if return_all:
            return matches.to_dict(orient="records")
        else:
            return matches.iloc[-1].to_dict()
