import sys
import numpy as np
from collections import Counter

import utils
import generate_data
import utils_plotting as plot


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
    population_size = 8
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
    g =  2
    M = 2
    
    signature = utils.hash_params(10, M=M, q=q, g=g, N=N, n=n, k=k, mutation_probability=mutation_probability, selection_method=selection_method, selection_strengths=str(selection_strengths), indegree_distribution=indegree_distribution, n_alphas=n_alphas, n_reps=n_reps)
    print(f'Run Signature: {signature}')
    print("loading data...")
    sccs, final_degrees, pheno, attr, ph_match, ranks, weights = utils.load_generated_data(signature)
    print("data loaded")
    print(np.mean(final_degrees[:,:,:,-1,:],(2,3)))

    for i, s in enumerate(sccs):
        plot.plot_grouped_bars(s, keys, 0.05, selection_strengths[i])
    
    plot.plot_heatmap(sccs, keys, selection_strengths=selection_strengths, tag=f'Number of SCCs at µ={mutation_probability}')
    plot.plot_heatmap(final_degrees, keys, selection_strengths=selection_strengths, tag=f'Average Final Degrees at µ={mutation_probability}')
    plot.plot_heatmap(pheno, keys, selection_strengths=selection_strengths, tag=f'Avg. Phenotypical Robustness at µ={mutation_probability}')
    plot.plot_heatmap(attr, keys, selection_strengths=selection_strengths, tag=f'Avg. Attrs at µ={mutation_probability}')
    plot.plot_heatmap(ph_match, keys, selection_strengths=selection_strengths, tag=f'Avg. Phenotypical Accuracy at µ={mutation_probability}')


    plot.plot_evolution_performance(sccs, keys, lambdas=selection_strengths, tag=f'Number of SCCs at µ={mutation_probability}')
    plot.plot_evolution_performance(final_degrees, keys, lambdas=selection_strengths, tag=f'Average Final Degrees at µ={mutation_probability}')
    plot.plot_evolution_performance(pheno, keys, lambdas=selection_strengths, tag=f'Avg. Phenotypical Robustness at µ={mutation_probability}')
    plot.plot_evolution_performance(attr, keys, lambdas=selection_strengths, tag=f'Avg. Attrs at µ={mutation_probability}')
    plot.plot_evolution_performance(ph_match, keys, lambdas=selection_strengths, tag=f'Avg. Phenotypical Accuracy at µ={mutation_probability}')
    
        
            
    
    for i, mp in enumerate(selection_strengths):
        plot.plot_emergency_scc_convergence(sccs2, i)
        
    for i, mp in enumerate(selection_strengths):
        plot.plot_scc_scatter(sccs2, i)
        
    plot.plot_selection_strength2(ranks, weights, selection_strengths)

    
    
    
    
    
    