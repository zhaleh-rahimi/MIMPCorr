"""
Created on Thu Jan  9 11:04:31 2025
@author: Zhaleh
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from util.stat_tests import check_stationarity_AR, check_invertibility_MA, perform_adfuller_test


class varma_data_generator:

    def __init__(self, config, seed = None):
        self.scenario_results = pd.DataFrame()  # DataFrame to store coefficients and autocorrelationmatrices
        
        self.steps = config["time_steps"] if "time_steps" in config else 500
        self.k = config["num_products"] if "num_products" in config else 2
        self.sigma_base = config["noise_level"]
        self.p = config["model_order"][0] if "model_order" in config else 1
        self.q = config["model_order"][1] if "model_order" in config else 0
        self.min_y = config["min_demand"] if "min_demand" in config else 10
 
        self.max_rho = config['max_rho'] if 'max_rho' in config else 0.8
        self.alpha = config['alpha'] if 'alpha' in config else 0.3
        
        self.sigma_u = np.eye(self.k) * self.sigma_base
        self.base_coeff = self.sigma_base / self.min_y
        
        self.seed = seed

    def generate_scenarios(self):
        """ Wrapper to Generate Scenarios
            output: dictionary of VARMA process, VARMA process without noise with scenario title , 
            strong, medium and low dependence
        """

        results = {}
        results_pred = {}
        k, p, q = self.k, self.p, self.q

        # High Dependence Scenario
        cov_high = self.generate_high_dependence_cov(self.max_rho, self.alpha)

        phi_matrices_high = self.yule_walker_multivariate(cov_high)
        title_high = f"Items={k}, p={p}, q={q}, High Dependence"

        theta_matrices = self.generate_theta_matrices(k, q)

        Y_high, pred_high = self.generate_varma_adjusted(
            phi_matrices_high, theta_matrices, title_high)

        results[title_high] = Y_high
        results_pred[title_high] = pred_high

        # Define reduction factors for Medium and Low dependence
        reduction_factors = {"Medium Dependence": 0.5, "Low Dependence": 0.05}

        # Generate Medium and Low Dependence based on High Dependence
        for strength_name, reduction_factor in reduction_factors.items():

            cov_matrices = [self.reduce_dependence(cov, reduction_factor)
                            for cov in cov_high]

            phi_matrices = self.yule_walker_multivariate(cov_matrices)

            title = f"Items={k}, p={p}, q={q}, {strength_name}"
            Y, pred = self.generate_varma_adjusted(
                phi_matrices, theta_matrices, title)
            results[title] = Y
            results_pred[title] = pred

        return results, results_pred

    def generate_high_dependence_cov(self, max_rho, alpha):
        """
        Generate a high-dependence covariance structure.
        """
        phi1 = np.eye(self.k) * 0.1 * self.sigma_base + \
            np.ones((self.k, self.k)) * max_rho * self.sigma_base - \
            np.eye(self.k) * max_rho*self.sigma_base

        matrix = [phi1 * (max_rho * alpha**(i)) for i in range(self.p)]
        return np.array(matrix)

    def reduce_dependence(self, base_matrix, factor):
        reduced_matrix = base_matrix * factor  # Handles element-wise scaling
        np.fill_diagonal(reduced_matrix, np.diag(base_matrix))  # Restore diagonals

        return reduced_matrix

    def generate_varma_adjusted(self, ar_matrices, ma_matrices, title):
        """ Wrapper of a VARMA process generation
        Considering the adjusted coeeficients, and enforced stationarity to process
        output: Adjusted VARMA process , and VARMA process without noise"""

        # Adjust coefficients
        ar_matrices = self.adjust_coefficients(ar_matrices, check_stationarity_AR)
        ma_matrices = self.adjust_coefficients(
            ma_matrices, check_invertibility_MA) if self.q > 0 else []

        # Generate VARMA process
        Y, pred = self.generate_varma(ar_matrices, ma_matrices)

        # Check invertibility for VMA
        is_invertible = check_invertibility_MA(ma_matrices) if self.q > 0 else True

        # Enforce stationarity if needed
        Y, pred, ar_matrices, is_stationary = self.enforce_stationarity_adf(
            Y, pred, ar_matrices, ma_matrices, title)
        if not is_stationary:
            print(f"Warning: Unable to enforce stationarity for {title}.")

        # Perform ADF test
        perform_adfuller_test(Y, title)

        # Compute autocovariance matrices
        autocor_matrices = self.compute_autocorrelation_matrix(Y, max_lag=len(ar_matrices))

        # Enforce positivity on generated demand data if needed
        Y, pred = self.enforce_positivity(Y, pred)

        # Save results to DataFrame
        self.save_scenario_results(title, ar_matrices, ma_matrices,
                                   is_stationary, is_invertible, autocor_matrices)

        return Y, pred

    def generate_varma(self, ar_matrices, ma_matrices, seed=None):
        """Generate VARMA Process (output: process, and process without noise)"""
        k, p, q, steps = self.k, self.p, self.q, self.steps

        Y = np.zeros((steps, k))
        pred = np.zeros((steps, k))

        if seed is not None:
            np.random.seed(seed)

        # Generate multivariate normal noise with mu=0 and Sigma=I
        mean = np.zeros(k)
        U = self.generate_uncorrelated_noise(mean, self.seed)

        Y[:(max(p, q))] = U[:(max(p, q))]
        for t in range(max(p, q), steps):
            ar_part = sum(ar_matrices[i] @ Y[t - i - 1] for i in range(p))
            ma_part = sum(ma_matrices[j] @ U[t - j - 1] for j in range(q))
            pred[t] = ar_part + ma_part
            Y[t] = pred[t] + U[t]
        return Y, pred

    def var_to_varma(self, var_coeffs):
        """
        Convert a VAR(p) process to an equivalent VARMA(p, q) process.

        Parameters:
        - var_coeffs: List of VAR coefficient matrices (list of np.array of shape (k, k)).
        - p: Order of the VAR process (integer).
        - q: Desired order of the VMA process (integer).

        Returns:
        - varma_coeffs: Tuple containing:
            - List of VAR coefficient matrices (p terms, as input).
            - List of VMA coefficient matrices (q terms).
        """
        p, q = self.p, self.q
        k = var_coeffs[0].shape[0]  # Dimension of the VAR process
        identity_matrix = np.eye(k)

        # Initialize Psi coefficients for the VMA representation
        psi_coeffs = [identity_matrix]  # Psi_0 is always the identity matrix

        # Compute Psi coefficients iteratively up to order q
        for i in range(1, q + 1):
            psi_i = np.zeros((k, k))
            for j in range(1, min(i, p) + 1):
                psi_i += var_coeffs[j - 1] @ psi_coeffs[i - j]
            psi_coeffs.append(psi_i)

        # Truncate Psi coefficients to q terms (Psi_1 to Psi_q)
        vma_coeffs = psi_coeffs[1:]

        # Return the VARMA process parameters
        return var_coeffs, vma_coeffs

    def generate_theta_matrices(self, k, q):
        """
        Generate invertible MA coefficient matrices.
        """
        ma_matrices = []
        for _ in range(q):
            if self.seed is not None:
                # Set the seed
                np.random.seed(self.seed)
            theta = np.random.uniform(0.1, 0.4, (k, k))
            theta = self.adjust_coefficients([theta], check_invertibility_MA)
            ma_matrices.append(theta[0])
        return ma_matrices

    def yule_walker_multivariate(self, cov_matrices):
        """
        Compute AR coefficients using Yule-Walker equations.
        """
        n, p = self.k, self.p
        cov_matrices = np.append(self.sigma_u, cov_matrices).reshape((p+1, n, n))
        R_blocks = [[cov_matrices[abs(i - j)] for j in range(p)] for i in range(p)]
        R = np.block(R_blocks)
        r = cov_matrices[1:].reshape((p*n, n))
        ar_flat = np.linalg.solve(R, r)
        return ar_flat.reshape((p, n, n))

    def compute_autocorrelation_matrix(self, Y, max_lag):
        """
        Compute the autocorrelation matrices for a multivariate time series.

        Args:
            Y (numpy.ndarray): Multivariate time series of shape (n_samples, n_variables).
            max_lag (int): Maximum lag for which to compute autocorrelation matrices.

        Returns:
            list: List of autocorrelation matrices, one for each lag (0 to max_lag).
        """
        n_samples, n_variables = Y.shape

        # Compute covariance matrices for all lags
        autocov_matrices = []
        for lag in range(max_lag + 1):
            Gamma_k = np.zeros((n_variables, n_variables))
            for t in range(lag, n_samples):
                Gamma_k += np.outer(Y[t], Y[t - lag])
            Gamma_k /= (n_samples - lag)
            autocov_matrices.append(Gamma_k)

        # Compute Gamma_0 (variance at lag 0)
        Gamma_0 = autocov_matrices[0]

        # Convert covariance matrices to correlation matrices
        autocorr_matrices = []
        D = np.diag(np.diag(Gamma_0))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        for Gamma_k in autocov_matrices:
            rho_k = D_inv_sqrt @ Gamma_k @ D_inv_sqrt
            autocorr_matrices.append(rho_k)

        return autocorr_matrices

    def adjust_coefficients(self, coefficients, check_func, scaling_factor=0.97, max_attempts=1e3):
        """Adjust Coefficients if not stationary or invertible (by scaling down)
        """
        attempts = 0
        adjusted_coefficients = coefficients.copy()
        while not check_func(adjusted_coefficients) and attempts < max_attempts:
            adjusted_coefficients = [coef * scaling_factor for coef in adjusted_coefficients]
            attempts += 1
        if attempts == max_attempts:
            raise ValueError("Unable to adjust coefficients to meet the condition.")
        return adjusted_coefficients

    def generate_uncorrelated_noise(self, mean, seed):
        """
        Generate serially uncorrelated multivariate normal noise.

        Args:
            n_samples (int): Number of samples to generate.
            n_dims (int): Number of dimensions.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            np.ndarray: Array of shape (n_samples, n_dims) with generated noise.
        """
        # variables
        cov, n_samples, n_dims = self.sigma_u, self.steps, self.k

        if seed is not None:
            np.random.seed(seed)

        # Generate multivariate normal noise with mu=0 and Sigma=I

        noise = np.random.multivariate_normal(mean, cov, size=n_samples)

        # Check for serial correlation and resample if needed
        for dim in range(n_dims):
            p_value = acorr_ljungbox(noise[:, dim], lags=[10], return_df=True)['lb_pvalue'].iloc[0]
            if p_value < 0.05:  # Significant autocorrelation found
                # print(f"Resampling dimension {dim} to remove serial correlation...")
                return self.generate_uncorrelated_noise(mean, self.seed + 1 if self.seed else None)

        return noise

    def enforce_stationarity_adf(self, Y, pred,  ar_matrices, ma_matrics, title, scaling_factor=0.97, max_attempts=1e3):
        """
        Modify AR coefficients to enforce stationarity based on the ADF test results.

        Parameters:
        - Y (ndarray): Generated VARMA process. 
        - pred (ndarray): generated VARMA without noise term
        - ar_matrices (list of ndarray): List of AR coefficient matrices [Phi_1, Phi_2, ...].
        - title (str): Scenario title for printing.
        - scaling_factor (float): Scaling factor to reduce AR coefficients.
        - max_attempts (int): Maximum number of scaling attempts.

        Returns:
        - adjusted_ar_matrices (list of ndarray): Adjusted AR coefficient matrices.
        - stationary (bool): Whether the process is stationary after adjustments.
        """
        attempts = 0
        while attempts < max_attempts:
            stationary = True
            for i in range(Y.shape[1]):
                result = adfuller(Y[:, i])
                if result[1] >= 0.05:  # Non-stationary
                    stationary = False
                    break

            if stationary:
                # print(f"Process became stationary after {attempts} adjustments for {title}.")
                return Y, pred, ar_matrices, True

            # Scale down AR coefficients to reduce dependency
            ar_matrices = [phi * scaling_factor for phi in ar_matrices]

            # Regenerate the process with adjusted coefficients
            Y, pred = self.generate_varma(ar_matrices, ma_matrics)
            attempts += 1

        print(f"Failed to enforce stationarity after {max_attempts} attempts for {title}.")
        return Y, pred, ar_matrices, False

    def enforce_positivity(self, Y, pred):
        """ Demands should always be non-negative, this function enforces non-negativity by 
        shifting the generated demand towards positive values"""
        # Compute column-wise minimums for Y and pred
        min_Y = np.min(Y, axis=0)
        min_pred = np.min(pred, axis=0)

        # Determine shifts for columns where positivity needs to be enforced
        shifts = np.maximum(0, -(np.maximum(min_Y, min_pred)) + self.min_y)

        # Apply the shifts to all rows for each column
        Y_new = Y + shifts
        pred_new = pred + shifts

        return Y_new, pred_new

    def save_scenario_results(self, scenario, ar_matrices, ma_matrices, stationary, invertible, autocor_matrices):
        """# Function to save results to the DataFrame"""
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
            "Autocorrelation Lag 4": autocor_matrices[4] if len(autocor_matrices) > 4 else None
        }

        self.scenario_results = pd.concat(
            [self.scenario_results, pd.DataFrame([result])], ignore_index=True)

    def get_scenario_results(self):
        return self.scenario_results
