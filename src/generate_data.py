import os
import sys
import csv
import numpy as np
import pandas as pd

try:
    import src.utils as utils
except ModuleNotFoundError:
    import utils
import json



try:
    filename = sys.argv[0]
    SLURM_ID = int(sys.argv[1])
except:
    filename = ''
    SLURM_ID = 0
if len(sys.argv)>2:
    M = int(sys.argv[2])
else:
    M = 100 #population size

if len(sys.argv)>3:
    q = float(sys.argv[3])
else:
    q = .1 #

if len(sys.argv)>4:
    g = int(sys.argv[4])
else:
    g = 20 #number generations
    
if len(sys.argv)>5:
    N = int(sys.argv[5])
else:
    N = 10 #network size

if len(sys.argv)>6:
    n = int(sys.argv[6])
else:
    n = 3 #degree
    
if len(sys.argv)>7:
    k = int(sys.argv[7])
else:
    k = 0 #minimal canalizing depth
    
if len(sys.argv)>8:
    mutation_probability = float(sys.argv[8])
else:
    mutation_probability = 0.01   
    
if __name__ == "__main__":
    print(f'Running {filename} with SLURM_ID={SLURM_ID}, M={M}, q={q}, g={g}, N={N}, n={n}, k={k}, mutation_probability={mutation_probability}')
    DEBUG_MODE = True
    STRONGLY_CONNECTED = False
    NO_SELF_REGULATION = True
    MUTATE_ONLY_CHILDREN = True
    indegree_distribution = 'poisson' #poisson #constant
        
    n_alphas = 5  
    population_size = 8
    n_reps = 2
    alpha_dyn = [0, 0.005]
    convex_2d = utils.generate_convex_combinations(1/n_alphas,dimension=2)
    all_alphas = utils.generate_alpha_combinations(alpha_dyn)
    n = 3
    keys = [f'({round(alpha_ph_rob, 4)}, {round(alpha_n_attractors, 4)}, {round(alpha_ph_match, 4)})' for i, (alpha_ph_rob, alpha_n_attractors, alpha_ph_match) in sorted(enumerate(all_alphas), key=lambda x: x[1][0])]
    selection_method = 'roulette_wheel_with_replacement' #others: sexual, 
    selection_strengths = [0, 1, 3, 10]
    mutation_probability = 0.05
    g =  2
    M = 2
    signature = utils.hash_params(10, M=M, q=q, g=g, N=N, n=n, k=k, mutation_probability=mutation_probability, selection_method=selection_method, selection_strengths=str(selection_strengths), indegree_distribution=indegree_distribution, n_alphas=n_alphas, n_reps=n_reps)
    print(f'Run Signature: {signature}')
    print("Logging run parameters to master_data_file.csv")
    row_items = [M, q, g, N, n, k, mutation_probability, selection_method, json.dumps(selection_strengths), indegree_distribution, n_alphas, json.dumps(alpha_dyn), STRONGLY_CONNECTED, MUTATE_ONLY_CHILDREN, NO_SELF_REGULATION, n_reps, signature, str(pd.Timestamp.now())]
    print(f"row items >>> {row_items}")
    data_run_exists = pd.read_csv('master_data_file.csv')
    if not signature in data_run_exists['signature'].values:
        print("data signature has not been logged yet, logging now.nnoiiiiiiiiiiiiiiiii")   
        with open('master_data_file.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_items)

    data_path = os.path.abspath(os.path.join('data/', f'{signature}'))
    if os.path.exists(data_path) and any(os.scandir(data_path)):
        print(f'Warning: Data path {data_path} already exists and data has been generated and saved. You may load the the data from this path.')
        exit(0)
        
    print(f"Creating data path at {data_path}")
    os.makedirs(os.path.abspath(data_path), exist_ok=True)
    print(f'Saving data to {data_path}')
    for selection_strength in selection_strengths:
        for i, (alpha_ph_rob, alpha_ph_match, alpha_n_attractors) in sorted(enumerate(all_alphas), key=lambda x: x[1][0]):
            data = utils.run_evolutionary_study_with_replicates_with_ph_match(population_size, q, g, N, n=n, k=k, 
                                          alpha_ph_rob=alpha_ph_rob, alpha_n_attractors=alpha_n_attractors, alpha_ph_match=alpha_ph_match, alpha_fragility=0, alpha_fhd=0, 
                                          mutation_probability=mutation_probability, STRONGLY_CONNECTED=STRONGLY_CONNECTED, 
                                          NO_SELF_REGULATION=NO_SELF_REGULATION, MUTATE_ONLY_CHILDREN=MUTATE_ONLY_CHILDREN,
                                          indegree_distribution=indegree_distribution, n_reps=n_reps, selection_method=selection_method, selection_strength=selection_strength, DEBUG=DEBUG_MODE, data_path=data_path)


