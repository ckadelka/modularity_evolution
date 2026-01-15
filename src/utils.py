import numpy as np
import pandas as pd
import hashlib
import random
import os
import pickle
import itertools
import boolforge
import json
import copy
from globals import *
from numba import njit, prange


def hash_params(count, **kwargs):
    """Generate a unique hash for a given set of python function parameters. It also sorts the params so that order passed doesn't matter.
    pass -1 in count param to get the full hash.
    count is the length of the hash you want to retrieve. .e.g passing 10 retireve first 10 characters of the hash
    """
    sig_str = ''.join(f'{value}' for _ , value in sorted(kwargs.items()))
    keys = ', '.join(f'{key}={value}' for key, value in sorted(kwargs.items()))
    return hashlib.sha256(sig_str.encode()).hexdigest()[:] if count == -1 else hashlib.sha256(sig_str.encode()).hexdigest()[:count]


def generate_convex_combinations(step_size, dimension=3):
    # Generate the possible values based on the step size
    values = [round(i * step_size, 4) for i in range(int(1 / step_size) + 1)]
    
    # Generate all combinations of three entries
    combinations = list(itertools.product(values, repeat=dimension))
    
    valid_vectors = []
    for comb in combinations:
        rounded_comb = tuple(round(x, 4) for x in comb)
        if abs(sum(rounded_comb) - 1) < 1e-9:
            valid_vectors.append(rounded_comb)

    # Remove duplicates while preserving order
    seen = set()
    unique_vectors = []
    for vec in valid_vectors:
        if vec not in seen:
            seen.add(vec)
            unique_vectors.append(vec)

    return unique_vectors


def generate_alpha_combinations(alpha_dyn, step_size=0.2):
    combos = []
    for a3 in alpha_dyn:
        remaining = round(1 - a3, 3)
        steps = [round(i * step_size * remaining, 3) for i in range(int(1 / step_size) + 1)]
        for a1 in steps:
            a2 = round(remaining - a1, 3)
            combos.append((a1, a2, a3))
    return combos


@njit(cache=True)
def int_to_binvec_numba(x, N):
    """
    Fast bitwise conversion of integer to binary array.
    Replaces the slow string formatting method.
    """
    out = np.zeros(N, dtype=np.int8)
    for i in range(N):
        # Extract the i-th bit using bitwise shift and AND
        out[N - 1 - i] = (x >> i) & 1
    return out

@njit(cache=True)
def _fast_attractor_hamming(attractor_states, target_ph, N):
    """
    Numba-accelerated kernel to calculate average Hamming distance 
    for one specific attractor cycle.
    """
    total_dist = 0.0
    steps = len(attractor_states)
    
    for i in range(steps):
        state_int = attractor_states[i]
        # Use the fast bitwise converter we wrote above
        state_bin = int_to_binvec_numba(state_int, N)
        
        # Manual Hamming distance calculation is faster in Numba than np.mean(a!=b)
        mismatch_count = 0
        for b in range(N):
            if state_bin[b] != target_ph[b]:
                mismatch_count += 1
        
        norm_dist = mismatch_count / N
        total_dist += norm_dist

    return total_dist / steps

def compute_ph_match_score_numbarized(target_ph, attractors, basin_sizes, N):
    """
    Optimized wrapper. It keeps the outer structure compatible with your objects,
    but offloads the heavy math to Numba.
    """
    # Ensure basin_sizes is a numpy array for vector math
    basin_sizes = np.array(basin_sizes, dtype=np.float64)
    basin_sizes /= basin_sizes.sum()

    match_scores = np.zeros(len(attractors))
    
    # Ensure target is the correct type for Numba
    target_ph_arr = np.array(target_ph, dtype=np.int8)

    for i, attractor_item in enumerate(attractors):
        # boolforge likely returns a list of ints; convert to array for Numba
        attractor_arr = np.array(attractor_item, dtype=np.int64)      
        # Call the fast Numba function
        match_scores[i] = _fast_attractor_hamming(attractor_arr, target_ph_arr, N)

    return np.dot(match_scores, basin_sizes)

# numba- Helper function for roulette-wheel selection weight calculation
@njit(cache=True)
def _calc_rank_weights(M, lam):
    weights = np.zeros(M)
    for rank_position in range(M):
        # Rank based score
        score = ((M - rank_position) / M) ** lam
        weights[rank_position] = score
    return weights

def roulette_wheel_selection_numbarized(M, current_fitness_values, lam=3):
    sorted_indices = np.argsort(current_fitness_values)[::-1]
    
    # Use the fast helper to generate raw weights
    raw_weights = _calc_rank_weights(M, lam)
    
    # Sync weights to original indices
    weights = np.zeros(M)
    for rank_pos, original_idx in enumerate(sorted_indices):
        weights[original_idx] = raw_weights[rank_pos]

    # Normalize
    weights_sum = np.sum(weights)
    if weights_sum > 0:
        weights = weights / weights_sum
    else:
        weights = np.ones(M) / M
        
    children = np.random.choice(M, M, replace=True, p=weights)
    return children, sorted_indices, weights

def int_to_binvec(x, N):
    return np.array([int(b) for b in format(x, f'0{N}b')])


def save_data(data, filename, folder='data'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.abspath(os.path.join(folder, filename))
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    # print(f"Data successfully saved to {file_path}")
    
    
def load_data(filename, folder='data'):
    file_path = os.path.abspath(os.path.join(folder, filename))
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")
    with open(file_path, 'rb') as file:
        # print(f"Data file FOUND it's actually LOADING DATA..................")
        data = pickle.load(file)
    return data


def compute_ph_match_score(target_ph, attractors, basin_sizes, N):
    basin_sizes = np.array(basin_sizes, dtype=float)  
    basin_sizes /= basin_sizes.sum()

    match_scores = np.zeros(len(attractors))

    for i, attractor_item in enumerate(attractors):
        for state_in_cycle_int in attractor_item:
            attractor_state_binvec = int_to_binvec(state_in_cycle_int, N)
            norm_hamming_distance = np.mean(attractor_state_binvec != target_ph)
            match_scores[i] += norm_hamming_distance / len(attractor_item)

    return np.dot(match_scores, basin_sizes)
# def roulette_wheel_selection(M, current_fitness_values, lam=1):
#     """
#     updated roulette wheel selection by fitness rank instead of fitness value...
#     Also include a new simulation-wide hyperParameter selection strength that enables selection pressue

    
#     """
#     print(f'Current fitness values... {current_fitness_values}')

#     # Step 1: Rank individuals (1 = best, M = worst)
#     ranks = np.argsort(-np.array(current_fitness_values)) + 1

#     # Step 2: Compute selection probabilities based on lambda
#     if lam == 0:
#         # No selection strength → uniform selection
#         weights = np.ones(M) / M
#     else:
#         # Rank-based selection strength: exponential bias
#         # weights = np.exp(lam * (ranks / M))
#         weights = [((M-r)/M)**lam for r in  ranks]
#         weights = weights / np.sum(weights)

#     print(f'Lambda: {lam}, Weights sum: {np.sum(weights)}')

#     # Step 3: Perform roulette wheel selection
#     children = np.random.choice(M, M, replace=True, p=weights)
#     return children, ranks, weights 

def roulette_wheel_selection(M, current_fitness_values, lam=3):
    """
    Corrected rank-based roulette wheel selection.
    """
    sorted_indices = np.argsort(current_fitness_values)[::-1]
    
    # Step 2: Initialize an empty weights array aligned with the original population
    weights = np.zeros(M)
    
    # Step 3: Assign weights based on Rank Position
    for rank_position, original_index in enumerate(sorted_indices):
        # rank_position 0 is the Best. rank_position M-1 is the Worst.
        # We use rank_position to determine the score, but store it at original_index
        
        # Formula: Higher rank_position (worse) -> Lower score
        # (M - 0) / M = 1.0 (Best)
        # (M - (M-1)) / M = Tiny (Worst)
        score = ((M - rank_position) / M) ** lam
        
        weights[original_index] = score

    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)

    # Step 4: Perform selection
    # Now weights[0] actually corresponds to Index 0
    children = np.random.choice(M, M, replace=True, p=weights)
    
    return children, sorted_indices, weights

def roulette_wheel_selection_v2(M, current_fitness_values, lam=1):
    # We don't add +1 here, we just want the sorted indices
    sorted_indices = np.argsort(-np.array(current_fitness_values))
    
    # Step 2: Assign weights based on POSITION in the sorted list
    weights = []
    
    # We iterate through the sorted list. 'i' is the rank (0=best, M-1=worst)
    for i in range(M):
        weight = ((M - i) / M) ** lam
        weights.append(weight)
        
    weights = np.array(weights)
    weights = weights / np.sum(weights) # Normalize to sum to 1

    # We select POSITIONS first (0 to M-1), then map them back to original indices
    selected_positions = np.random.choice(M, M, replace=True, p=weights)
    
    # Map the selected positions back to the actual network indices
    children_indices = sorted_indices[selected_positions]
    
    return children_indices, sorted_indices, weights



def load_generated_data(signature):
    data_path = os.path.join(GLOBAL_MASTER_DIR, 'data', f'{signature}')
    print(f'Loading data from {data_path}')
    if not os.path.exists(data_path):
        print(f"Error: Data path '{data_path}' does not exist. Kindly check the signature or verify that you have run the data generation step.")
        return None
    
    data_file = os.path.abspath(os.path.join(GLOBAL_MASTER_DIR, 'master_data_file.csv'))
    if not os.path.exists(data_file):
        print(f'Error: master_data_file.csv does not exist. Kindly check the signature or verify that you have run the data generation step.')
        return None
    df = pd.read_csv(data_file)
    row_df = df[df['signature'] == signature]
    if len(row_df) == 0:
        print(f'Error: Data path with this signature does not exist. Kindly check the signature or verify that you have run the data generation step.')
        return None
    print(f'Found data for signature {signature} in master_data_file.csv')
    print(df)

    row_data = row_df.iloc[0].to_dict()
    print(f'Row data: {row_data}')
    
    #my dilemma is to find a way to load the data for this signature folder
    #into my NdArray variables sccs, final_degrees, pheno, attr, ph_match, ranks, weights
    #now deserialize things from the row record of this data signature in dataframe 
    print(f"Loading data for signature {signature}, from {data_path}")
    print(f"columns are {df.columns}")
    selection_strengths = json.loads(row_data['selection_strengths'])
    print(f'Selection strengths: {selection_strengths}')
    print(f'type of selection_strengths: {type(selection_strengths)}')
    print(f"n_alphas: {row_data['n_alphas']}")
    n_alphas = int(row_data['n_alphas'])
    population_size = int(row_data['population_size'])
    n_reps = int(row_data['n_reps'])
    mutation_probability = float(row_data['mutation_probability'])
    M = int(row_data['population_size'])
    print(f"retrieved M is {M}")
    q = float(row_data['percent_selected'])
    g = int(row_data['generations'])
    N = int(row_data['network_size'])
    n = int(row_data['degree'])    
    k = int(row_data['canalizing_depth'])
    alpha_dyn = json.loads(row_data['alpha_dyn'])
    all_alphas = generate_alpha_combinations(alpha_dyn)
    indegree_distribution = row_data['indegree_distribution']
    selection_method = row_data['selection_method']
    STRONGLY_CONNECTED = bool(row_data['STRONGLY_CONNECTED'])
    MUTATE_ONLY_CHILDREN = bool(row_data['MUTATE_ONLY_CHILDREN'])
    NO_SELF_REGULATION = bool(row_data['NO_SELF_REGULATION'])
    DEBUG_MODE = False
    
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
    print(f"selection strengths before we start iterating data... >>> {selection_strengths}")
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
            data = run_evolutionary_study_with_replicates_with_ph_match(population_size, q, g, N, n=n, k=k, 
                                          alpha_ph_rob=alpha_ph_rob, 
                                          alpha_n_attractors=alpha_n_attractors, alpha_ph_match=alpha_ph_match, 
                                          alpha_fragility=0, alpha_fhd=0, 
                                          mutation_probability=mutation_probability, STRONGLY_CONNECTED=STRONGLY_CONNECTED, 
                                          NO_SELF_REGULATION=NO_SELF_REGULATION, MUTATE_ONLY_CHILDREN=MUTATE_ONLY_CHILDREN,
                                          indegree_distribution=indegree_distribution, 
                                          n_reps=n_reps, 
                                          selection_method=selection_method, 
                                          selection_strength=selection_strength,
                                          DEBUG=DEBUG_MODE, data_path=data_path)
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
    
    return sccs, sccs2, final_degrees, pheno, attr, ph_match, ranks, weights

def run_evolutionary_study_with_replicates_with_ph_match(
    M, q, g, N, n, k=0, 
    alpha_ph_rob=1, alpha_n_attractors=0, alpha_ph_match=0, alpha_fragility=0, alpha_fhd=0,
    mutation_probability=0, 
    target_ph=None,
    STRONGLY_CONNECTED=False, 
    NO_SELF_REGULATION=True, 
    MUTATE_ONLY_CHILDREN=False, 
    indegree_distribution='constant', 
    n_reps=500,
    selection_method='sexual', #others are tournament, roulette_wheel_with_replacement
    selection_strength=1,
    DEBUG=False,
    data_path='data'
):
    """
    Modified version of our initial study with replicates with adds another fitness component PHENOTYPICAL_MATCH on specific phenotypic profile or attractor profile
    
    in simple terms, adding target attractor selection ACCURACY as a secondary criterion.
    """
    hash_stamp = hash_params(-1, M=M, q=q, g=g, N=N, n=n, k=k, alpha_ph_rob=alpha_ph_rob, alpha_n_attractors=alpha_n_attractors, alpha_ph_match=alpha_ph_match, alpha_fragility=alpha_fragility, alpha_fhd=alpha_fhd, mutation_probability=mutation_probability, target_ph=target_ph, STRONGLY_CONNECTED=STRONGLY_CONNECTED, NO_SELF_REGULATION=NO_SELF_REGULATION, MUTATE_ONLY_CHILDREN=MUTATE_ONLY_CHILDREN, indegree_distribution=indegree_distribution, n_reps=n_reps, selection_method=selection_method, selection_strength=selection_strength)
    # Generate target phenotype if not provided
    if target_ph is None:
        target_ph = np.random.randint(0, 2, N)
    else:
        if isinstance(target_ph[0], int) and max(target_ph) > 1:
            target_ph = int_to_binvec(target_ph[0], N)
        target_ph = np.array(target_ph)
    
    hashed_str = hashlib.sha256(hash_stamp.encode()).hexdigest()

    try:
       data = load_data('results_evolutionary_study_%s.pkl' % (hashed_str), folder=data_path)
       fitness_mat,num_attractors_mat,phenotypical_robustness_mat,strongly_connected_component_mat,fragility_mat,final_hamming_distance_mat, final_degree_mat, ph_match_fitness_mat, ranks_mat, weights_mat = data
    except FileNotFoundError:
        fitness_mat = np.zeros((M, g, n_reps))
        ph_match_fitness_mat = np.zeros((M, g, n_reps))
        num_attractors_mat = np.nan * np.zeros((M, g, n_reps))
        phenotypical_robustness_mat = np.nan * np.zeros((M, g, n_reps))
        strongly_connected_component_mat = np.nan * np.zeros((M, g, n_reps))
        fragility_mat = np.nan * np.zeros((M, g, n_reps))
        final_hamming_distance_mat = np.nan * np.zeros((M, g, n_reps))
        final_degree_mat = np.nan * np.zeros((M, g, n_reps))
        ranks_mat = np.nan * np.zeros((g, n_reps, M))
        weights_mat = np.nan * np.zeros((g, n_reps, M))

        # if DEBUG:
        #     print(f"M={M} && (a1, a2, a3): ({alpha_ph_rob}, {alpha_n_attractors} {alpha_ph_match})")
        #     print(f"Length of fitness mat >>> {len(fitness_mat)}")

        for nth_sim in range(n_reps):
            bns = []
            
            for index in range(M):
                
                bn = boolforge.random_network(N,
                                              n,
                                              depth=k,
                                              STRONGLY_CONNECTED=STRONGLY_CONNECTED,
                                              NO_SELF_REGULATION=NO_SELF_REGULATION,
                                              indegree_distribution=indegree_distribution)
                
                bns.append(bn)
            
            for gen in range(g):
                # if DEBUG:
                #     print('alphas %s gth %i' % (', '.join(list(map(str,[alpha_ph_rob,alpha_n_attractors,alpha_fragility,alpha_fhd]))), gen))
                for index, bn in enumerate(bns):
                    attractor_info = bn.get_attractors_and_robustness_measures_synchronous_exact()
                    attractors = attractor_info['Attractors']
                    basin_sizes = attractor_info['BasinSizes']
                    number_of_attractors = attractor_info['ExactNumberOfAttractors']
                    fragility = attractor_info['Fragility']
                    coherence = attractor_info['Coherence']                    
                    ph_match = compute_ph_match_score(target_ph, attractors, basin_sizes, N)
                                        
                    sccs = bn.get_strongly_connected_components()
                    strongly_connected_component_mat[index,gen,nth_sim] = len(sccs)
                    
                    num_attractors_mat[index,gen,nth_sim] = number_of_attractors
                    phenotypical_robustness_mat[index,gen,nth_sim] = coherence
                    fragility_mat[index,gen,nth_sim] = fragility
                    final_hamming_distance_approximation = 0 #not currently implemented
                    final_hamming_distance_mat[index,gen,nth_sim] = final_hamming_distance_approximation
                    final_degree_mat[index,gen,nth_sim] = np.mean(bn.indegrees)
        
                    fit = (alpha_ph_rob * coherence + 
                           alpha_n_attractors * number_of_attractors + 
                           alpha_fragility * (1 - fragility) + 
                           alpha_fhd * final_hamming_distance_approximation + 
                           alpha_ph_match * ph_match
                           )
                    fitness_mat[index,gen,nth_sim] = fit
                    ph_match_fitness_mat[index,gen,nth_sim] = ph_match
                    
                if selection_method == 'sexual':
                    #Step B: select parents for next generation
                    dummy = np.argsort(fitness_mat[:,gen,nth_sim])
                    indices_parents = dummy[-int(q*M):]
                    indices_children = dummy[:int(q*M)]
                    
            
                    #Step C: create next generation by copying the best performing networks (parents) and creating children through reproduction
                    for index in indices_children:
                        #select parents
                        
                        parents = indices_parents[np.random.choice(int(q*M),2,replace=False)]
                        for i in range(N):
                            r = int(np.round(random.random()>0.5))
                            bns[index].F[i] = bns[parents[r]].F[i].copy()
                            bns[index].I[i] = bns[parents[r]].I[i].copy()
                    
                    #Step D: mutate each network
                    for index in (indices_children if MUTATE_ONLY_CHILDREN else range(M)): #with some probability the regulators of a gene change randomly
                        for i in range(N):
                            if random.random()<mutation_probability:
                                n_regulators = bns[index].indegrees[i]
                                if NO_SELF_REGULATION == False:
                                    bns[index].I[i] = np.random.choice(N,n_regulators,replace=False)
                                else:
                                    dummy = np.random.choice(N-1,n_regulators,replace=False)
                                    dummy[dummy>=i] += 1
                                    bns[index].I[i] = dummy
                elif selection_method == 'roulette_wheel_with_replacement':    
                    # Fitness based...roulette wheel selection reproduction....
                    # Step B: Roulette Wheel Selection (Fitness-Proportional, With Replacement)
                    fitness_scores = fitness_mat[:, gen, nth_sim]  # Fitness of all networks in current gen

                    #roulette-wheel selection...
                    # indices_selected, ranks, weights = roulette_wheel_selection(M, fitness_scores, lam=selection_strength)
                    #UPDATE on Jan 1, 2026
                    # --- Step A: Elite preservation --- 
                    # elite_index = np.argmax(fitness_scores) 
                    # elite_bn = copy.deepcopy(bns[elite_index]) 
                    # --- Step B: Roulette wheel selection (rank-based) --- 
                    indices_selected, ranks, weights = roulette_wheel_selection(M, fitness_scores, lam=selection_strength) 

                    ranks_mat[gen, nth_sim, :] = ranks 
                    weights_mat[gen, nth_sim, :] = weights 
                    # --- Step C: Create next generation by copying --- 
                    new_bns = [] 
                    for idx in indices_selected: 
                        new_bns.append(copy.deepcopy(bns[idx])) 
                    # Insert elite (replace worst child) 
                    # new_bns[-1] = elite_bn 
                    bns = new_bns
                    
                    
                    # Step D: Mutate the selected networks (now applies to ALL M networks)
                    for index in range(M):  # Mutate all selected networks (no parent/child distinction)
                        for i in range(N):   # For each Node(gene) in the network
                            if random.random() < mutation_probability:
                                n_regulators = bns[index].indegrees[i]                    
                                # Randomly reassign regulators
                                if NO_SELF_REGULATION == False:
                                    bns[index].I[i] = np.random.choice(N, n_regulators, replace=False)
                                else:
                                    # Avoid self-regulation: exclude current index 'i' from choices
                                    dummy = np.random.choice(N - 1, n_regulators, replace=False)
                                    dummy[dummy >= i] += 1  # Shift indices >= i to skip self
                                    bns[index].I[i] = dummy
                                    
                else:
                    raise ValueError(f"Unknown selection method: {selection_method}")
                    
                # if DEBUG:
                #     print('Generation %i completed' % gen)
            
        data = (fitness_mat,num_attractors_mat,phenotypical_robustness_mat,strongly_connected_component_mat,fragility_mat,final_hamming_distance_mat, final_degree_mat, ph_match_fitness_mat, ranks_mat, weights_mat)
        save_data(data,'results_evolutionary_study_%s.pkl' % (hashed_str), folder=data_path)
        
    return fitness_mat,num_attractors_mat,phenotypical_robustness_mat,strongly_connected_component_mat,fragility_mat,final_hamming_distance_mat, final_degree_mat, ph_match_fitness_mat, ranks_mat, weights_mat

