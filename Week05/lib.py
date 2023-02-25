

import numpy as np
from numpy import linalg as la
import pandas as pd
import scipy
import copy

# Covariance estimation techniques
def generate_ew_correlation_matrix_variance_vector(data, lam):
    # Calculate the weighted covariance matrix
    weight = lam ** np.arange(data.shape[0])
    cov_matrix = np.cov(data, rowvar=False, aweights=weight)
    return cov_matrix



#Non PSD fixes for correlation matrices
def near_psd(a, epsilon=0.0):
    n = a.shape[0]

    invSD = None
    out = a.copy()

    if np.count_nonzero(np.isclose(np.diag(out), 1.0)) != n:
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = np.dot(np.dot(invSD, out), invSD)

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / np.sum(vecs * vecs * vals, axis=1)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = np.dot(np.dot(T, vecs), l)
    out = np.dot(B, B.T)

    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = np.dot(np.dot(invSD, out), invSD)

    return out


def isPD(B):
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
    
    
def Higham(A):
    B = (A + A.T) / 2
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

    
    
#simulation methods

def chol_pd(a):
    n = a.shape[0]
    root = np.zeros((n, n))
    for j in range(n):
        s = np.sum(root[j, :j] ** 2)
        root[j, j] = np.sqrt(a[j, j] - s)
        ir = 1.0 / root[j, j]
        for i in range(j + 1, n):
            s = np.dot(root[i, :j], root[j, :j])
            root[i, j] = (a[i, j] - s) * ir
    return root



def direct_simulation(cov, n_samples=25000):
    B = chol_psd(cov)
    r = scipy.random.randn(len(B[0]), n_samples)
    return B @ r



def pca_simulation(cov, pct_explained, n_samples=25000):
    eigen_values, eigen_vectors = np.linalg.eigh(cov)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalues = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    evr = sorted_eigenvalues / sorted_eigenvalues.sum()
    cumulative_evr = evr.cumsum()
    cumulative_evr[-1] = 1
    idx = bisect_left(cumulative_evr, pct_explained)
    explained_vals = np.clip(sorted_eigenvalues[:idx + 1], 0, np.inf)
    explained_vecs = sorted_eigenvectors[:, :idx + 1]
    B = explained_vecs @ np.diag(np.sqrt(explained_vals))
    r = scipy.random.randn(B.shape[1], n_samples)
    return B @ r





# VaR calculation methods (all discussed)
def calculate_var(data, mean=0, alpha=0.05):
    return mean - np.quantile(data, alpha)


def history(data,S,sorted_returns, alpha = 0.05):
    n = len(sorted_returns)
    cutoff_index = int(np.floor(n * alpha))
    VaR = - sorted_returns[cutoff_index]
    return VaR

def t_var(data, mean=0, alpha=0.05, nsamples=10000):
    params = scipy.stats.t.fit(data, method="MLE")
    df, loc, scale = params
    simulation_t = scipy.stats.t(df, loc, scale).rvs(nsamples)
    var_t = calculate_var(simulation_t, mean, alpha)
    return var_t



#ES calculation
def calculate_es(data, mean=0, alpha=0.05):
    var = calculate_var(data, mean, alpha)
    return -np.mean(data[data <= -var])






#return calculate 
def pd_return_calculate(prices, method="arithmetic"):
    price_change_percent = (prices / prices.shift(1))[1:]
    if method == "arithmetic":
        return price_change_percent - 1
    elif method == "log":
        return np.log(price_change_percent)
    
