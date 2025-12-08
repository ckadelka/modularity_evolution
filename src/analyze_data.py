import sys
import numpy as np
from collections import Counter

import utils
import generate_data


try:
    filename = sys.argv[0]
    SLURM_ID = int(sys.argv[1])
except:
    filename = ''
    SLURM_ID = 0
if len(sys.argv)>2:
    M = int(sys.argv[2])
else:
    M = 100

if len(sys.argv)>3:
    q = float(sys.argv[3])
else:
    q = .1

if len(sys.argv)>4:
    g = int(sys.argv[4])
else:
    g = 20
    
if len(sys.argv)>5:
    N = int(sys.argv[5])
else:
    N = 10

if len(sys.argv)>6:
    n = int(sys.argv[6])
else:
    n = 3
    
if len(sys.argv)>7:
    k = int(sys.argv[7])
else:
    k = 0    
    
if len(sys.argv)>8:
    mutation_probability = float(sys.argv[8])
else:
    mutation_probability = 0.01
    
    
if __name__ == "__main__":
    print(f'Running {filename} with SLURM_ID={SLURM_ID}, M={M}, q={q}, g={g}, N={N}, n={n}, k={k}, mutation_probability={mutation_probability}')
    DEBUG_MODE = False
    STRONGLY_CONNECTED = False
    NO_SELF_REGULATION = True
    MUTATE_ONLY_CHILDREN = True
    indegree_distribution = 'poisson' #poisson #constant
        
    n_alphas = 5  
    #another experiment
    population_size = 10
    n_reps = 2
    alpha_dyn = [0, 0.005]
    convex_2d = utils.generate_convex_combinations(1/n_alphas,dimension=2)
    all_alphas = utils.generate_alpha_combinations(alpha_dyn)
    mutation_probabilities = [0.05]


    sccs = []
    sccs2 = []
    ph_match2 = []
    final_degrees=[]
    pheno = []
    attr = []
    ph_match = []
    fit2 = []
    ranks = []
    weights = []
    n = 3
    keys = [f'({round(alpha_ph_rob, 4)}, {round(alpha_n_attractors, 4)}, {round(alpha_ph_match, 4)})' for i, (alpha_ph_rob, alpha_n_attractors, alpha_ph_match) in sorted(enumerate(all_alphas), key=lambda x: x[1][0])]
    selection_method = 'roulette_wheel_with_replacement'
    selection_strengths = [0, 1, 3, 10]
    mutation_probability = 0.05
    g =  100
    
    # f"{M}{q}{g}{N}{n}{k}{mutation_probability}{selection_method}{selection_strengths}{indegree_distribution}{n_alphas}{population_size}{n_sim}"
    signature = utils.hash_params(M=M, q=q, g=g, N=N, n=n, k=k, mutation_probability=mutation_probability, selection_method=selection_method, selection_strengths=selection_strengths, indegree_distribution=indegree_distribution, n_alphas=n_alphas, population_size=population_size, n_reps=n_reps)
    print(f'Run Signature: {signature}')
    sccs, final_degrees, pheno, attr, ph_match, ranks, weights = generate_data.load_generated_data(signature)
    print(np.mean(final_degrees[:,:,:,-1,:],(2,3)))
    
    
    
    
    
    