"""
Support Measure Machines - Core Implementation
Based on: "Learning from Distributions via Support Measure Machines"
Muandet, Fukumizu, Dinuzzo, Schölkopf (2012)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# MEAN EMBEDDING KERNEL (Expected Kernel)
# Equation (4) from the paper - empirical estimate
# K_emp(P_hat_n, Q_hat_m) = (1/nm) * sum_i sum_j k(x_i, z_j)
# ============================================================

def rbf_kernel_matrix(X, Z, gamma=1.0):
    """Compute RBF kernel matrix between rows of X and Z."""
    # ||x - z||^2 = ||x||^2 + ||z||^2 - 2*x^T*z
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    Z_sq = np.sum(Z**2, axis=1, keepdims=True)
    dist_sq = X_sq + Z_sq.T - 2 * X @ Z.T
    return np.exp(-gamma / 2.0 * dist_sq)


def linear_kernel_matrix(X, Z):
    """Linear kernel matrix."""
    return X @ Z.T


def empirical_expected_kernel(samples_i, samples_j, gamma=1.0, kernel='rbf'):
    """
    Compute empirical expected kernel K_emp(P_i, P_j)
    Equation (4): (1/nm) * sum_i sum_j k(x_i, z_j)
    
    samples_i: array of shape (n_i, d) — samples from distribution P_i
    samples_j: array of shape (n_j, d) — samples from distribution P_j
    """
    if kernel == 'rbf':
        K_mat = rbf_kernel_matrix(samples_i, samples_j, gamma=gamma)
    else:
        K_mat = linear_kernel_matrix(samples_i, samples_j)
    return np.mean(K_mat)


def build_smm_kernel_matrix(distributions, gamma=1.0, kernel='rbf'):
    """
    Build the full m x m SMM kernel matrix.
    Each entry K[i,j] = K_emp(P_i, P_j)
    
    distributions: list of arrays, each of shape (n_samples, d)
    """
    m = len(distributions)
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            k_val = empirical_expected_kernel(
                distributions[i], distributions[j], gamma=gamma, kernel=kernel)
            K[i, j] = k_val
            K[j, i] = k_val
    return K


def build_smm_kernel_matrix_train_test(test_distributions, train_distributions,
                                        gamma=1.0, kernel='rbf'):
    """Build (n_test, n_train) kernel matrix for prediction with sklearn precomputed."""
    m_test = len(test_distributions)
    m_train = len(train_distributions)
    K = np.zeros((m_test, m_train))
    for i in range(m_test):
        for j in range(m_train):
            K[i, j] = empirical_expected_kernel(
                test_distributions[i], train_distributions[j],
                gamma=gamma, kernel=kernel)
    return K


# ============================================================
# GAUSSIAN ANALYTICAL KERNEL (closed form from Table 1)
# For Gaussian N(m; Sigma) with Gaussian RBF embedding kernel:
# K(P_i, P_j) = exp(-1/2 * (m_i - m_j)^T (Sigma_i+Sigma_j+gamma^{-1}I)^{-1} (m_i-m_j))
#               / |gamma*Sigma_i + gamma*Sigma_j + I|^{1/2}
# ============================================================

def gaussian_expected_rbf_kernel(m_i, Sigma_i, m_j, Sigma_j, gamma=1.0):
    """
    Closed-form expected RBF kernel between two Gaussians.
    From Table 1 of the paper.
    """
    d = len(m_i)
    diff = m_i - m_j
    M = Sigma_i + Sigma_j + (1.0 / gamma) * np.eye(d)
    M_inv = np.linalg.inv(M)
    det_M_scaled = np.linalg.det(gamma * Sigma_i + gamma * Sigma_j + np.eye(d))
    
    exponent = -0.5 * diff @ M_inv @ diff
    normalizer = det_M_scaled ** 0.5
    return np.exp(exponent) / normalizer


def build_gaussian_smm_kernel(means, covariances, gamma=1.0):
    """Build SMM kernel matrix using closed-form Gaussian expected RBF kernel."""
    m = len(means)
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            k_val = gaussian_expected_rbf_kernel(
                means[i], covariances[i], means[j], covariances[j], gamma=gamma)
            K[i, j] = k_val
            K[j, i] = k_val
    return K


# ============================================================
# DATA GENERATION
# ============================================================

def generate_gaussian_distributions(n_distributions_per_class=50, 
                                     n_samples_per_dist=30, 
                                     d=2, seed=42):
    """
    Generate two classes of Gaussian distributions.
    Class +1: means drawn from N(m+, Sigma_class), covariances from Wishart
    Class -1: means drawn from N(m-, Sigma_class), covariances from Wishart
    
    Follows the experimental setup of Section 6.1 in the paper.
    """
    rng = np.random.RandomState(seed)
    
    m_pos = np.ones(d) * 1.0
    m_neg = np.ones(d) * 2.0
    Sigma_class = 0.5 * np.eye(d)
    
    distributions = []
    labels = []
    means_list = []
    covs_list = []
    
    for cls, m_cls in [(+1, m_pos), (-1, m_neg)]:
        for _ in range(n_distributions_per_class):
            # Sample mean for this distribution
            mean_i = rng.multivariate_normal(m_cls, Sigma_class)
            
            # Sample covariance from Wishart
            df = d + 2
            scale = 0.3 * np.eye(d) if cls == +1 else 0.6 * np.eye(d)
            A = rng.randn(df, d)
            cov_i = (A.T @ A) * scale[0, 0] / df
            cov_i += 0.1 * np.eye(d)  # ensure positive definite
            
            # Sample data points from this distribution
            samples = rng.multivariate_normal(mean_i, cov_i, size=n_samples_per_dist)
            
            distributions.append(samples)
            means_list.append(mean_i)
            covs_list.append(cov_i)
            labels.append(cls)
    
    return distributions, np.array(labels), means_list, covs_list


def generate_toy_dataset_smm(n_per_class=60, n_samples=20, d=2, seed=42):
    """
    Generate toy dataset for SMM reproduction (Task 2).
    Two classes of Gaussian blobs with different spreads.
    Uses the Breast Cancer Wisconsin features as inspiration but is synthetic.
    """
    rng = np.random.RandomState(seed)
    
    # Class 1: tight clusters (benign-like)
    m1 = np.array([2.0, 2.0])
    # Class 2: spread clusters (malignant-like) 
    m2 = np.array([5.0, 5.0])
    
    distributions = []
    labels = []
    
    for _ in range(n_per_class):
        mean = m1 + rng.randn(d) * 0.5
        cov = np.diag(rng.uniform(0.1, 0.4, d))
        samples = rng.multivariate_normal(mean, cov, size=n_samples)
        distributions.append(samples)
        labels.append(+1)
    
    for _ in range(n_per_class):
        mean = m2 + rng.randn(d) * 0.8
        cov = np.diag(rng.uniform(0.3, 1.0, d))
        samples = rng.multivariate_normal(mean, cov, size=n_samples)
        distributions.append(samples)
        labels.append(-1)
    
    return distributions, np.array(labels)


if __name__ == '__main__':
    print("SMM Core Implementation loaded successfully.")
    
    # Quick sanity check
    dist1 = np.random.randn(20, 2)
    dist2 = np.random.randn(20, 2) + 3
    k_val = empirical_expected_kernel(dist1, dist2, gamma=1.0)
    print(f"Empirical K between two separated Gaussians: {k_val:.4f}")
    
    k_self = empirical_expected_kernel(dist1, dist1, gamma=1.0)
    print(f"Self-kernel: {k_self:.4f}")
    print("Sanity check passed.")
