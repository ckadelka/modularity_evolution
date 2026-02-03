import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from globals import *


try:
    import src.utils as utils
except ModuleNotFoundError:
    import utils
import json


parser = argparse.ArgumentParser(description="bn_optimizer: A computational study on evolutionary multi-objective optimization of Boolean GRNs")
parser.add_argument("-s", "--SLURM_ID", dest="SLURM_ID", required=False, type=int)
parser.add_argument("-M", "--population-size", dest="M", required=True, type=int)
parser.add_argument("-q", "--top-percent-parents", dest="q", required=False, type=float)
parser.add_argument("-g", "--generations", dest="g", required=True, type=int)
parser.add_argument("-N", "--network-size", dest="N", required=True, type=int)
parser.add_argument("-n", "--degree", dest="n", required=True, type=int)
parser.add_argument("-k", "--canalizing-depth", dest="k", required=False, default=0, type=int)
parser.add_argument("-d", "--indegree-distribution", dest="d", required=False, default="poisson",
                    choices=["poisson", "constant"])
parser.add_argument("-p", "--mutation-probability", dest="p", required=False, default=.01, type=float)
parser.add_argument("-r", "--number-of-replications", dest="r", required=True, type=int)

    
def main(args):
    DEBUG_MODE = False
    STRONGLY_CONNECTED = False
    NO_SELF_REGULATION = True
    MUTATE_ONLY_CHILDREN = True
    indegree_distribution = "poisson" #poisson #constant
        
    n_alphas = 5  
    # population_size = 100
    # n_reps = r * SLURM_ID
    alpha_dyn = [0, 0.005]
    all_alphas = utils.generate_alpha_combinations(alpha_dyn)
    # keys = [f'({round(alpha_ph_rob, 4)}, {round(alpha_n_attractors, 4)}, {round(alpha_ph_match, 4)})' for i, (alpha_ph_rob, alpha_n_attractors, alpha_ph_match) in sorted(enumerate(all_alphas), key=lambda x: x[1][0])]
    selection_method = "roulette_wheel_with_replacement" #others: sexual, 
    selection_strengths = [0, 1, 3, 10]
    signature = utils.hash_params(10, M=args.M, q=args.q, g=args.g, N=args.N, n=args.n, k=args.k, mutation_probability=args.p, selection_method=selection_method, selection_strengths=str(selection_strengths), indegree_distribution=indegree_distribution, n_alphas=n_alphas, n_reps=args.r)
    print(f"Run Signature: {signature}")
    print("Logging run parameters to master_data_file.csv")
    # #find it in the two potential paths
    master_data_file = os.path.join(GLOBAL_MASTER_DIR, "master_data_file.csv")
    print("Logging run parameters to master_data_file.csv")
    row_items = [args.M, args.q, args.g, args.N, args.n, args.k, args.p, selection_method, json.dumps(selection_strengths), indegree_distribution, n_alphas, json.dumps(alpha_dyn), STRONGLY_CONNECTED, MUTATE_ONLY_CHILDREN, NO_SELF_REGULATION, args.r, signature, str(pd.Timestamp.now())]
    
    # if not os.path.exists(master_data_file):
    #     pass 
    
    data_run_exists = pd.read_csv(master_data_file)
    if not signature in data_run_exists["signature"].values:
        with open(master_data_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row_items)


    base_data_path = os.path.abspath(os.path.join(os.path.dirname(master_data_file), "data", f"{signature}"))
    data_path = os.path.join(base_data_path, f"job_{args.SLURM_ID}")
    # if os.path.exists(data_path) and any(os.scandir(data_path)):
    #     print(f'Warning: Data path {data_path} already exists and data has been generated and saved. You may load the the data at this location using the analyze_data.py module.')
    #     sys.exit(0)
        
    print(f"Creating data path at {data_path}")
    os.makedirs(os.path.abspath(data_path), exist_ok=True)
    print(f"Saving data to {data_path}")
    for selection_strength in selection_strengths:
        for i, (alpha_ph_rob, alpha_ph_match, alpha_n_attractors) in sorted(enumerate(all_alphas), key=lambda x: x[1][0]):
            data = utils.run_evolutionary_study_with_replicates_with_ph_match(args.M, args.q, args.g, args.N, n=args.n, k=args.k, 
                                          alpha_ph_rob=alpha_ph_rob, alpha_n_attractors=alpha_n_attractors, alpha_ph_match=alpha_ph_match, alpha_fragility=0, alpha_fhd=0, 
                                          mutation_probability=args.p, STRONGLY_CONNECTED=STRONGLY_CONNECTED, 
                                          NO_SELF_REGULATION=NO_SELF_REGULATION, MUTATE_ONLY_CHILDREN=MUTATE_ONLY_CHILDREN,
                                          indegree_distribution=indegree_distribution, n_reps=args.r, selection_method=selection_method, selection_strength=selection_strength, DEBUG=DEBUG_MODE, data_path=data_path)

    

if __name__=="__main__":
    args = parser.parse_args()
    main(args)

