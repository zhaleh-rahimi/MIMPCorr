"""
Created on Thu Jan  9 11:04:31 2025

@author: Zhaleh
"""
from statsmodels.tsa.stattools import adfuller
import numpy as np
from scipy.linalg import eigvals
from scipy.stats import shapiro


def check_stationarity(series, col_name, prnt=False):
    """ TS test of stationarity using ADFULLER for a single vector """
    result = adfuller(series)
    if prnt:
        print(f'ADF Statistic for {col_name}: {result[0]}, p-value: {result[1]}')
    return result[1] <= 0.05  # Stationary if p-value <= 0.05


def perform_adfuller_test(Y, title):
    """Perform ADF Test for stationarity of multiple vectors in a numpy array
    input: ndarray and title for the test
    output: print the stationary for each column in ndarray
    """
    # print(f"\nADF Test Results for Scenario: {title}")
    for i in range(Y.shape[1]):
        result = adfuller(Y[:, i])
        # print(
            # f"Process {i + 1}: p-value = {result[1]:.4f} (Stationary? {'Yes' if result[1] < 0.05 else 'No'})")
    return all(result[1] < 0.05 for result in [adfuller(Y[:, i]) for i in range(Y.shape[1])])

def check_stationarity_AR(phi_matrices):
    """Stationarity Check for AR part"""
    k = phi_matrices[0].shape[0]
    p = len(phi_matrices)
    top = np.hstack(phi_matrices)
    bottom = np.eye(k * (p - 1), k * p)
    companion_matrix = np.vstack([top, bottom])
    eigenvalues = eigvals(companion_matrix)
    return np.all(np.abs(eigenvalues) < 1)


def check_invertibility_MA(theta_matrices):
    """Invertibility Check for MA part"""
    k = theta_matrices[0].shape[0]
    q = len(theta_matrices)
    top = -np.hstack(theta_matrices)
    bottom = np.eye(k * (q - 1), k * q)
    companion_matrix = np.vstack([top, bottom])
    eigenvalues = eigvals(companion_matrix)
    return np.all(np.abs(eigenvalues) < 1)


def iqr(x):
    # Custom aggregation function for IQR
    return np.percentile(x, 75) - np.percentile(x, 25)


def test_normality(group, column='percentage_improvement'):
    """Function for normality testing"""
    stat, p_value = shapiro(group[column])
    return {'normal' if p_value > 0.05 else 'not normal'}


def compute_coefficient_of_variation(data):
    mean_vector = np.mean(data, axis=0)

    covariance_matrix = np.cov(data, rowvar=False)
    total_variance = np.diag(covariance_matrix)

    cv = np.sqrt(total_variance) / mean_vector
    return [mean_vector, total_variance, cv]


def relative_noise_dispersion(mu_Y, Sigma_epsilon):
    """
    Compute the relative noise dispersion for a VARMA process.

    Parameters:
    - mu_Y (numpy array): Mean vector of the VARMA process (steady-state mean).
    - Sigma_epsilon (numpy array): Covariance matrix of the noise terms (innovations).

    Returns:
    - dispersion_vector (numpy array): Relative noise dispersion for each component.
    - overall_dispersion (float): Overall noise dispersion measure.
    """

    # Compute standard deviation of noise (sqrt of diagonal elements of Sigma_epsilon)
    sigma_epsilon = np.sqrt(np.diag(Sigma_epsilon))

    # Compute the relative noise dispersion for each component
    dispersion_vector = np.abs(sigma_epsilon / mu_Y)  # Element-wise division

    # Compute the overall dispersion
    trace_sigma = np.sqrt(np.trace(Sigma_epsilon))  # Sum of variances (approx total noise level)
    norm_mu_Y = np.linalg.norm(mu_Y)  # Euclidean norm of the mean vector
    overall_dispersion = trace_sigma / norm_mu_Y if norm_mu_Y != 0 else np.nan  # Avoid division by zero

    return overall_dispersion, dispersion_vector


