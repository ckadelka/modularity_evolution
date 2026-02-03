# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 16:00:00 2026

@author: faith
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fr Feb 14 23:23:15 2025

@author: ckadelka
"""

import scipy.integrate as integrate
from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from globals import *
from pathlib import Path
import pandas as pd



try:
    import src.utils as utils
except ModuleNotFoundError:
    import utils


#difference equation
def rho_beverton_holt(N,K):
    return K/(K+N)

def rho_exponential(N,K):
    return np.exp(-K*N)

rho = rho_beverton_holt

def TYC(x,sx,sy,sz,a,c,K,mu):
    X,Y,Z = x
    dummy = rho(c*X,K)
    Xnew = sx * X + a * c * dummy * Y/(Y+Z) * X
    Ynew = sy * Y + (1-a) * c * dummy * Y/(Y+Z) * X + c * dummy * Z/(Y+Z) * X
    Znew = sz * Z + mu
    return (Xnew,Ynew,Znew)

def TYC_delta(x,t,sx,sy,sz,a,c,K,mu):
    X,Y,Z = x
    dummy = rho(c*X,K)
    Xnew = sx * X + a * c * dummy * Y/(Y+Z) * X - X
    Ynew = sy * Y + (1-a) * c * dummy * Y/(Y+Z) * X + c * dummy * Z/(Y+Z) * X - Y
    Znew = sz * Z + mu - Z
    return (Xnew,Ynew,Znew)

def get_solution(model,x0,T,args=()):
    sol = np.zeros((T+1,len(x0)))
    sol[0,:] = x0
    for i in range(T):
        sol[i+1,:] = model(sol[i,:],*args)
    return sol

def get_solution_start_from_equilibrium(model,x0,T,T_to_equilibrium=20,args=()):
    sol = np.zeros((T+1,len(x0)))
    sol[0,:] = x0
    for i in range(T):
        sol[i+1,:] = model(sol[i,:],*args if i>=T_to_equilibrium else tuple(list(args[:-1])+[0]))
    return sol
    
# ============================================================
# Sensitivity analysis module for the TYC discrete-time model
# ============================================================

# ------------------------------------------------------------
# Helper: run model and extract outcomes
# ------------------------------------------------------------
def run_model_and_classify(
    params,
    T=500,
    eps=1e-6,
    x0=10.0,
    y0=10.0,
    z0=0.0
):
    """
    Runs the discrete-time TYC model and classifies outcome.

    Returns
    -------
    extinct : bool
        True if female population goes extinct
    t_extinct : int or None
        Time to extinction (None if no extinction)
    x_final : float
        Final female abundance
    """

    # Unpack parameters
    mx, my, mz, a, c, K, mu = params

    x, y, z = x0, y0, z0

    for t in range(T):
        # --- model update (copied from your main loop logic) ---
        rho = max(0.0, 1.0 - (c * x) / K)

        x_new = (1 - mx) * x + a * c * rho * (y / (y + z + 1e-12)) * x
        y_new = (1 - my) * y + c * rho * (
            (1 - a) * (y / (y + z + 1e-12)) + (z / (y + z + 1e-12))
        ) * x
        z_new = (1 - mz) * z + mu

        x, y, z = max(x_new, 0.0), max(y_new, 0.0), max(z_new, 0.0)

        if x < eps:
            return True, t, x

    return False, None, x

# ------------------------------------------------------------
# Global sensitivity sampler
# ------------------------------------------------------------
def global_sensitivity_analysis_v1(
    param_ranges,
    N=2000,
    T=500,
    seed=0
):
    """
    Performs global sensitivity analysis via random sampling.

    param_ranges: dict with keys
        ['mx','my','mz','a','c','K','mu']
        and (min,max) tuples
    """

    rng = np.random.default_rng(seed)

    results = []

    for _ in range(N):
        params = np.array([
            rng.uniform(*param_ranges['mx']),
            rng.uniform(*param_ranges['my']),
            rng.uniform(*param_ranges['mz']),
            rng.uniform(*param_ranges['a']),
            rng.uniform(*param_ranges['c']),
            rng.uniform(*param_ranges['K']),
            rng.uniform(*param_ranges['mu']),
        ])

        extinct, t_ext, x_final = run_model_and_classify(params, T=T)

        results.append({
            'params': params,
            'extinct': extinct,
            't_extinct': t_ext if extinct else T,
            'x_final': x_final
        })

    return results

# ============================================================
# PRCC utilities 
# ============================================================

# -----------------------------
# Rank transform (with tie handling)
# -----------------------------
def rankdata_average_ties(x):
    """
    Returns ranks in [1..n] with average ranks for ties (like scipy.stats.rankdata(method='average')).
    """
    x = np.asarray(x)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    # Average ties
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (i + 1 + j)  # average of rank positions (1-indexed)
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _residualize(y, X):
    """
    Residualize y against columns of X via OLS with intercept.
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    # Add intercept
    Xd = np.column_stack([np.ones(n), X])

    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    y_hat = Xd @ beta
    resid = (y - y_hat).ravel()
    return resid


def _corr(a, b):
    """
    Pearson correlation of 1D arrays.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0:
        return np.nan
    return float(np.sum(a * b) / denom)


def prcc(X, y, param_names=None, rank=True):
    """
    Partial Rank Correlation Coefficients (PRCC).

    Parameters
    ----------
    X : array, shape (n_samples, n_params)
        Parameters.
    y : array, shape (n_samples,)
        Scalar outcome (continuous recommended).
        For binary extinction, prefer using time-to-extinction or x_final.
    param_names : list[str] or None
    rank : bool
        If True, rank-transform X and y before computing partial correlations.

    Returns
    -------
    out : dict with keys:
        'names', 'prcc'
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape

    if param_names is None:
        param_names = [f"p{i}" for i in range(p)]

    if rank:
        Xr = np.column_stack([rankdata_average_ties(X[:, j]) for j in range(p)])
        yr = rankdata_average_ties(y)
    else:
        Xr = X.copy()
        yr = y.copy()

    prcc_vals = np.zeros(p, dtype=float)

    for j in range(p):
        others = [k for k in range(p) if k != j]
        X_others = Xr[:, others]

        rx = _residualize(Xr[:, j], X_others)
        ry = _residualize(yr, X_others)

        prcc_vals[j] = _corr(rx, ry)

    return {"names": param_names, "prcc": prcc_vals}


def plot_prcc(prcc_out, ylabel="Partial Rank Correlation Coefficient (PRCC)", savepath=None,title=''):
    """
    Simple bar plot of PRCC values.
    """
    names = prcc_out["names"]
    vals = prcc_out["prcc"]

    order = np.argsort(np.abs(vals))[::-1]
    names_sorted = [names[i] for i in order]
    vals_sorted = vals[order]

    plt.figure()
    plt.bar(range(len(vals_sorted)), vals_sorted)
    plt.axhline(0, linewidth=1)
    plt.xticks(range(len(vals_sorted)), names_sorted)
    plt.ylabel(ylabel)
    if title!='':
        plt.title(title)
    plt.grid(axis='y')
    plt.gca().set_axisbelow(True) 
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    return order


def my_global_sensitivity_analysis(param_range, NUM_SAMPLES=10, M=100, N=10, q=.2, g=20, n_reps=1, GSA_OUTPUT_PATH=None, seed=42):  
    
    data_path = os.path.join(GLOBAL_MASTER_DIR, "data", "gsa")
    print(f"Creating data path at {data_path}")
    os.makedirs(os.path.abspath(data_path), exist_ok=True)
    
    param_names = list(param_range.keys())
    
    # 2. Generate LHS Samples (Efficient Stratified Sampling)
    print(f"Generating {NUM_SAMPLES} LHS samples...")
    sampler = qmc.LatinHypercube(d=len(param_names), seed=seed)
    sample_matrix = sampler.random(n=NUM_SAMPLES)
    
    # Scale samples to actual ranges
    params_df = pd.DataFrame(sample_matrix, columns=param_names)
    for col in param_names:
        lower, upper = param_range[col]
        params_df[col] = lower + params_df[col] * (upper - lower)
    
    # Integers must be rounded
    #params_df['N'] = params_df['N'].round().astype(int)
    #params_df['n'] = params_df['n'].round().astype(int)
    params_df['k'] = np.floor(params_df['k']).astype(int)

    
    # 3. Setup Constants
    selection_method = 'roulette_wheel_with_replacement'
    DEBUG = True
    STRONGLY_CONNECTED = False
    NO_SELF_REGULATION = True
    MUTATE_ONLY_CHILDREN = True
    indegree_distribution = 'poisson'
    
    results = []
    
    print("...starting simulation...")
    
    for i, row in params_df.iterrows():
        # --- THE FIX: Calculate Alphas from the Ratio ---
        a_attr = row['alpha_attr']
        ratio  = row['ratio_rob']
        
        remainder = 1.0 - a_attr
        a_rob   = remainder * ratio
        a_match = remainder * (1.0 - ratio)
        
        a_attr  = round(a_attr, 4)
        a_rob   = round(a_rob, 4)
        a_match = round(a_match, 4)
        
        # Ensure they sum to exactly 1
        diff = 1.0 - (a_attr + a_rob + a_match)
        a_match += diff 
        
        try:
            p_val = row['p']
            k_val = int(row['k'])
            lam_val = (row['lam'])
            
            
            # --- RUN SIMULATION ---
            data = utils.run_evolutionary_study_with_replicates_with_ph_match(
                M=M, 
                q=q, 
                g=g, 
                N=N, 
                n=n, 
                k=k_val,
                alpha_ph_rob=a_rob, 
                alpha_n_attractors=a_attr,
                alpha_ph_match=a_match, 
                alpha_fragility=0, 
                alpha_fhd=0,
                mutation_probability=p_val, 
                STRONGLY_CONNECTED=STRONGLY_CONNECTED,
                NO_SELF_REGULATION=NO_SELF_REGULATION, 
                MUTATE_ONLY_CHILDREN=MUTATE_ONLY_CHILDREN,
                indegree_distribution=indegree_distribution, 
                n_reps=n_reps, 
                selection_method=selection_method, 
                selection_strength=lam_val, 
                DEBUG=DEBUG, 
                data_path=data_path
            )
            
            _, attr, pheno, sccs, _, _, _, ph_match, _, _ = data
            res_scc = np.mean(sccs) 
            res_rob = np.mean(pheno)
            res_match = np.mean(ph_match)

            results.append({
                'N': N, 'p': p_val, 'n': n, 'k': k_val, 'lam': lam_val, 'ratio_rob': ratio, 
                'alpha_attr': a_attr, 'alpha_rob': a_rob, 'alpha_match': a_match,
                'scc_outcome': res_scc,
                'robustness_outcome': res_rob,
                'match_outcome': res_match
            })
            
            if i % 50 == 0:
                print(f"Sample {i}/{NUM_SAMPLES} complete.")

        except Exception as e:
            print(f"Error on sample {i}: {e}")

    results_df = pd.DataFrame(results)
    save_path = os.path.join(os.getcwd(), "gsa_results.csv") if GSA_OUTPUT_PATH is None else Path(GSA_OUTPUT_PATH)
    results_df.to_csv(save_path, index=False)
    print(f"Done. Saved to {save_path}")
    return results_df
    
    


parser = argparse.ArgumentParser(description="bn_optimizer: A computational study on evolutionary multi-objective optimization of Boolean GRNs")
parser.add_argument("-s", "--SLURM_ID", dest="SLURM_ID", required=False, type=int)
parser.add_argument("-M", "--population-size", dest="M", required=True, type=int)
parser.add_argument("-q", "--top-percent-parents", dest="q", required=False, type=float)
parser.add_argument("-g", "--generations", dest="g", required=True, type=int)
parser.add_argument("-N", "--network-size", dest="N", required=True, type=int)
parser.add_argument("-n", "--degree", dest="n", required=True, type=int)
# parser.add_argument("-k", "--canalizing-depth", dest="k", required=False, default=0, type=int)
# parser.add_argument("-d", "--indegree-distribution", dest="d", required=False, default="poisson",
#                     choices=["poisson", "constant"])
parser.add_argument("-p", "--mutation-probability", dest="p", required=False, default=.01, type=float)
parser.add_argument("-r", "--number-of-replications", dest="r", required=True, type=int)




if __name__=="__main__":
    #implementing GSA using PRCC (Partial Rank Correlation Co-efficient)
    N=10
    n=3
    
    param_range = {
        # 'N': [10, 12],              # Network Size
        'p': [0., 0.05],          # Mutation Prob
        # 'n': [1, 5],                # In-degree
        'k': [0, n],                # Canalizing depth
        'lam': [0, 5],             # Selection Strength
        'alpha_attr': [0.0, 0.005],   # Weight for Attractors
        'ratio_rob': [0.0, 1.0],    # Split ratio: 0=All Match, 1=All Robustness
    }

    NUM_SAMPLES = 300
    M = 2         
    q = 0.1        
    g = 20          
    n_reps = 1
    GSA_OUTPUT_PATH = "gsa_results2.csv"
    
    results = my_global_sensitivity_analysis(param_range, NUM_SAMPLES=NUM_SAMPLES,M=M, N=N, q=q, g=g, n_reps=n_reps, GSA_OUTPUT_PATH=GSA_OUTPUT_PATH, seed=0)


    