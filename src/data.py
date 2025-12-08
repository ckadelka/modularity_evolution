import numpy as np
import hashlib
import random
import os
import pickle
import itertools
import boolforge





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

def int_to_binvec(x, N):
    return np.array([int(b) for b in format(x, f'0{N}b')])
def save_data(data, filename, folder='data'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, filename)
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data successfully saved to {file_path}")
    
    
def load_data(filename, folder='data'):
    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data
def compute_ph_match_score(target_ph, attractors, basin_sizes):
    basin_sizes = np.array(basin_sizes, dtype=float)  
    basin_sizes /= basin_sizes.sum()

    match_scores = np.zeros(len(attractors))

    for i, attractor_item in enumerate(attractors):
        for state_in_cycle_int in attractor_item:
            attractor_state_binvec = int_to_binvec(state_in_cycle_int, N)
            norm_hamming_distance = np.mean(attractor_state_binvec != target_ph)
            match_scores[i] += norm_hamming_distance / len(attractor_item)

    return np.dot(match_scores, basin_sizes)
def roulette_wheel_selection(M, current_fitness_values, lam=1):
    """
    updated roulette wheel selection by fitness rank instead of fitness value...
    Also include a new simulation-wide hyperParameter selection strength that enables selection pressue

    
    """
    print(f'Current fitness values... {current_fitness_values}')

    # Step 1: Rank individuals (1 = best, M = worst)
    ranks = np.argsort(-np.array(current_fitness_values)) + 1

    # Step 2: Compute selection probabilities based on lambda
    if lam == 0:
        # No selection strength → uniform selection
        weights = np.ones(M) / M
    else:
        # Rank-based selection strength: exponential bias
        # weights = np.exp(lam * (ranks / M))
        weights = [((M-r)/M)**lam for r in  ranks]
        weights = weights / np.sum(weights)

    print(f'Lambda: {lam}, Weights sum: {np.sum(weights)}')

    # Step 3: Perform roulette wheel selection
    children = np.random.choice(M, M, replace=True, p=weights)
    return children, ranks, weights 
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
    Debug=False,
    data_path='data'
):
    """
    Modified version of our initial study with replicates with adds another fitness component PHENOTYPICAL_MATCH on specific phenotypic profile or attractor profile
    
    in simple terms, adding target attractor selection ACCURACY as a secondary criterion.
    """
    # hash_stamp = f"{M}{q}{g}{N}{n}{k}{alpha_ph_rob}{alpha_n_attractors}{alpha_fragility}{alpha_fhd}{mutation_probability}{STRONGLY_CONNECTED}{NO_SELF_REGULATION}{MUTATE_ONLY_CHILDREN}{indegree_distribution}{n_reps}"    
    hash_stamp = f"{M}{q}{g}{N}{n}{k}{alpha_ph_rob}{alpha_n_attractors}{alpha_fragility}{alpha_fhd}{mutation_probability}{target_ph}{STRONGLY_CONNECTED}{NO_SELF_REGULATION}{MUTATE_ONLY_CHILDREN}{indegree_distribution}{n_reps}{selection_method}{selection_strength}{DEBUG_MODE}"
   
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

        print(f"M={M} && (a1, a2, a3): ({alpha_ph_rob}, {alpha_n_attractors} {alpha_ph_match})")
        print(f"Length of fitness mat >>> {len(fitness_mat)}")

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
                if Debug:
                    print('alphas %s gth %i' % (', '.join(list(map(str,[alpha_ph_rob,alpha_n_attractors,alpha_fragility,alpha_fhd]))), gen))
                for index, bn in enumerate(bns):
                    attractor_info = bn.get_attractors_and_robustness_measures_synchronous_exact()
                    attractors = attractor_info['Attractors']
                    basin_sizes = attractor_info['BasinSizes']
                    number_of_attractors = attractor_info['ExactNumberOfAttractors']
                    fragility = attractor_info['Fragility']
                    coherence = attractor_info['Coherence']                    
                    ph_match = compute_ph_match_score(target_ph, attractors, basin_sizes)
                                        
                    sccs = bn.get_strongly_connected_components()
                    strongly_connected_component_mat[index,gen,nth_sim] = len(sccs)
                    
                    num_attractors_mat[index,gen,nth_sim] = number_of_attractors
                    phenotypical_robustness_mat[index,gen,nth_sim] = coherence
                    fragility_mat[index,gen,nth_sim] = fragility
                    final_hamming_distance_approximation = 0#not currently implemented
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
                    indices_selected, ranks, weights = roulette_wheel_selection(M, fitness_scores, lam=selection_strength)
                    print(f'{len(indices_selected)} children selected from the roulette wheel selection')
                    ranks_mat[gen,nth_sim, :] = ranks
                    weights_mat[gen,nth_sim, :] = weights
                    # Step C: Create next generation by DIRECT COPYING...
                    new_bns = []
                    for index in indices_selected:
                        new_bns.append(bns[index])
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
                    
                if Debug:
                    print('Generation %i completed' % gen)
            
        data = (fitness_mat,num_attractors_mat,phenotypical_robustness_mat,strongly_connected_component_mat,fragility_mat,final_hamming_distance_mat, final_degree_mat, ph_match_fitness_mat, ranks_mat, weights_mat)
        save_data(data,'results_evolutionary_study_%s.pkl' % (hashed_str), folder=data_path)
        
    return fitness_mat,num_attractors_mat,phenotypical_robustness_mat,strongly_connected_component_mat,fragility_mat,final_hamming_distance_mat, final_degree_mat, ph_match_fitness_mat, ranks_mat, weights_mat

