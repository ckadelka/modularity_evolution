import os
import sys
import numpy as np
import pandas as pd

import utils



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
    
    

def load_generated_data(signature):
    data_path = os.path.join('../data/', f'{signature}')
    print(f'Loading data from {data_path}')
    df = pd.read_csv(os.path.join("../", 'master_data_file.csv'))
    if df[df['signature'] == signature].empty():
        print(f'Error: Data path {data_path} does not exist. Kindly check the signature or verify that you have run the data generation step.')
        return None
    
    #my dilemma is to find a way to load the data for this signature folder
    #into my NdArray variables sccs, final_degrees, pheno, attr, ph_match, ranks, weights
    
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
    for selection_strength in selection_strengths:
        sccs.append([])
        final_degrees.append([])
        pheno.append([])
        attr.append([])
        sccs2.append([])
        ph_match2.append([])
        ph_match.append([])
        fit2.append([])
        ranks.append([])
        weights.append([])
        for i, (alpha_ph_rob, alpha_ph_match, alpha_n_attractors) in sorted(enumerate(all_alphas), key=lambda x: x[1][0]):
            data = utils.run_evolutionary_study_with_replicates_with_ph_match(population_size, q, g, N, n=n, k=k, 
                                          alpha_ph_rob=alpha_ph_rob, 
                                          alpha_n_attractors=alpha_n_attractors, alpha_ph_match=alpha_ph_match, 
                                          alpha_fragility=0, alpha_fhd=0, 
                                          mutation_probability=mutation_probability, STRONGLY_CONNECTED=STRONGLY_CONNECTED, 
                                          NO_SELF_REGULATION=NO_SELF_REGULATION, MUTATE_ONLY_CHILDREN=MUTATE_ONLY_CHILDREN,
                                          indegree_distribution=indegree_distribution, 
                                          n_reps=n_reps, 
                                          selection_method=selection_method, 
                                          selection_strength=selection_strength,
                                          Debug=DEBUG_MODE, data_path=data_path)
            fitness_mat, num_attractors_mat, phenotypical_robustness_mat, strongly_connected_component_mat, fragility_mat, final_hamming_distance_mat, final_degree_mat, ph_match_fitness_mat,ranks_mat, weights_mat = data       
            sccs[-1].append(strongly_connected_component_mat)
            key = f'({round(alpha_ph_rob, 4)}, {round(alpha_n_attractors, 4)}, {round(alpha_ph_match, 4)})'
            sccs2[-1].append({key: strongly_connected_component_mat}) 
            final_degrees[-1].append(final_degree_mat)
            pheno[-1].append(phenotypical_robustness_mat)
            attr[-1].append(num_attractors_mat)
            ph_match[-1].append(ph_match_fitness_mat)
            ph_match2[-1].append({key: ph_match_fitness_mat})
            fit2[-1].append({key: fitness_mat})
            ranks[-1].append(ranks_mat)
            weights[-1].append(weights_mat)
            
    sccs = np.array(sccs)
    final_degrees = np.array(final_degrees)
    pheno = np.array(pheno)
    attr = np.array(attr)
    ph_match = np.array(ph_match)
    ranks = np.array(ranks)
    weights = np.array(weights)
    
    print(np.mean(final_degrees[:,:,:,-1,:],(2,3))) 
    
    return sccs, final_degrees, pheno, attr, ph_match, ranks, weights
    
    
if __name__ == "__main__":
    print(f'Running {filename} with SLURM_ID={SLURM_ID}, M={M}, q={q}, g={g}, N={N}, n={n}, k={k}, mutation_probability={mutation_probability}')
    DEBUG_MODE = False
    STRONGLY_CONNECTED = False
    NO_SELF_REGULATION = True
    MUTATE_ONLY_CHILDREN = True
    indegree_distribution = 'poisson' #poisson #constant
        
    n_alphas = 5  
    population_size = 100
    n_reps = 2
    alpha_dyn = [0, 0.005]
    convex_2d = utils.generate_convex_combinations(1/n_alphas,dimension=2)
    all_alphas = utils.generate_alpha_combinations(alpha_dyn)
    n = 3
    keys = [f'({round(alpha_ph_rob, 4)}, {round(alpha_n_attractors, 4)}, {round(alpha_ph_match, 4)})' for i, (alpha_ph_rob, alpha_n_attractors, alpha_ph_match) in sorted(enumerate(all_alphas), key=lambda x: x[1][0])]
    selection_method = 'roulette_wheel_with_replacement' #others: sexual, 
    selection_strengths = [0, 1, 3, 10]
    mutation_probability = 0.05
    g =  20
    # sign_str = f"{M}{q}{g}{N}{n}{k}{mutation_probability}{selection_method}{selection_strengths}{indegree_distribution}{n_alphas}{population_size}{n_sim}"
    # signature = hashlib.sha256(sign_str.encode()).hexdigest()[:10]
    signature = utils.hash_params(M=M, q=q, g=g, N=N, n=n, k=k, mutation_probability=mutation_probability, selection_method=selection_method, selection_strengths=selection_strengths, indegree_distribution=indegree_distribution, n_alphas=n_alphas, population_size=population_size, n_reps=n_reps)
    print(f'Run Signature: {signature}')
    # data = pd.DataFrame({
    #     'M': M,
    #     'q': q,
    #     'g': g,
    #     'N': N,
    #     'n': n,
    #     'k': k,
    #     'mutation_probability': mutation_probability,
    #     'selection_method': selection_method,
    #     'selection_strengths': str(selection_strengths),
    #     'indegree_distribution': indegree_distribution,
    #     'n_alphas': n_alphas,
    #     'population_size': population_size,
    #     'n_reps': n_reps,
    #     'signature': signature,
    #     'date' : pd.Timestamp.now()
    # }, index=[0])
    f = open('../master_data_file.csv', mode='a')
    f.write('\t'.join([#TODO]))
    f.close()
    data_path = os.path.join('../data/', f'{signature}')
    if os.path.exists(data_path):
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
                                          indegree_distribution=indegree_distribution, n_reps=n_reps, selection_method=selection_method, selection_strength=selection_strength, Debug=DEBUG_MODE, data_path=data_path)


