import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def plot_scc_scatter(sccs, all_alphas=None, selection_strengths=None):
    for index, _ in enumerate(selection_strengths):
        measure_name = 'scc'
        f, ax = plt.subplots()
        sns.set_theme(style="whitegrid")
        plt.style.use('seaborn-v0_8-whitegrid')
        
        #create a colormap with distinct colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_alphas)))
        # Select data for mutation probability 0.03 (index from input)
        scc_p = sccs[index]  
        
        for i, scc_data in enumerate(scc_p):
            if not scc_data:
                continue
            
            key, data = next(iter(scc_data.items()))
            alpha1, alpha2, alpha3 = map(float, key.strip('()').split(','))
            
            if data.size == 0:
                continue
            
            #mean across population and replications...
            avg_convergence = data.mean(axis=(0, 2))  
            
            #plot scatter points for this alpha combination
            print(f'length of avg convergence >>> {len(avg_convergence)}')
            ax.scatter(
                np.arange(len(avg_convergence)),
                avg_convergence,
                color=colors[i],
                s=20,
                alpha=0.8,
                label=fr"$\alpha_1={round(alpha1,4)}$, $\alpha_2={round(alpha2,4)}$, $\alpha_3={round(alpha3,4)}$"
            )
        ax.text(0.02, 0.98, f'μ = 0.05, λ = {selection_strengths[index]}',
            transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel(measure_name.upper(), fontsize=12)
        ax.set_xticks(np.arange(0, 20, 2))  
        ax.tick_params(axis='x', labelsize=10)  
        ax.tick_params(axis='y', labelsize=10)  
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            fontsize=9,
            frameon=True,
            title='Alpha Combinations',
            title_fontsize='10'
        )
        
        plt.tight_layout()
        plt.show()

    



def plot_evolution_performance(measure, keys=None, lambdas=None, tag='Number of SCCs'):
    """
    Plot evolution performance for measures like SCCs, Attractor Length, etc.
    
    Parameters
    ----------
    measure : np.ndarray
        5D array of shape (num_lambdas, num_alphas, num_sims, num_generations, num_pop)
    keys : list or None
        List of α labels, e.g., ['α1', 'α2', ..., 'α12'].
        If None, it auto-generates them as bold α₁ … αₙ.
    lambdas : list
        List of λ values corresponding to each selection strength.
    tag : str
        Label for y-axis and plot title (e.g. 'Number of SCCs').
    """
    import numpy as np
    import matplotlib.pyplot as plt

    num_lambdas = measure.shape[0]
    num_alphas = measure.shape[1]

    if keys is None:
        keys = [fr'$\mathbf{{\alpha_{{{i+1}}}}}$' for i in range(num_alphas)]
    else:
        keys = [fr'$\mathbf{{\alpha_{{{i+1}}}}}$' for i in range(len(keys))]

    mean_values = np.mean(measure, axis=(2, 3, 4)) 

    fig, ax = plt.subplots()

    colors = ['blue', 'red', 'orange', 'green']
    markers = ['o', 's', 'D', '^']

    for i in range(num_lambdas):
        ax.plot(
            range(num_alphas),
            mean_values[i],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=fr'$\lambda={lambdas[i]}$' if lambdas is not None else fr'$\lambda_{i+1}$',
            linewidth=2,
            markersize=6
        )

    ax.set_xticks(range(num_alphas))
    ax.set_xticklabels(keys, rotation=45)
    ax.set_xlabel(r'$\mathbf{\alpha}$ (network configuration index)', fontsize=13)
    ax.set_ylabel(tag, fontsize=13)
    title = fr"Evolution of {tag} across $\mathbf{{\alpha}}$ for different $\lambda$"
    print(f"Title is >>> {title}")
    ax.legend(title=r'selection strength: $\lambda$')
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def plot_evolution_performance_all_v2(measures, keys=None, lambdas=None, measures_names=None):
        """
        Plot evolution performance for measures like SCCs, Attractor Length, etc.
        
        Parameters
        ----------
        measures : np.ndarray
            list of 5D arrays of shape (num_lambdas, num_alphas, num_sims, num_generations, num_pop)
        keys : list or None
            List of α labels, e.g., ['α1', 'α2', ..., 'α12'].
            If None, it auto-generates them as bold α₁ … αₙ.
        lambdas : list
            List of lambda values corresponding to each selection strength.
        tag : str
            Label for y-axis and plot title (e.g. 'Number of SCCs').
        """
        import numpy as np
        import matplotlib.pyplot as plt

        num_lambdas = measures[0].shape[0]
        num_alphas = measures[0].shape[1]

        keys_mod = np.array(list(map(lambda x: x.replace(')','').replace('(','').split(', '),keys)),dtype=float)
        
        #replications, generations, and population axes
        mean_values = []
        for ii in range(len(measures)):
            mean_values.append(np.mean(measures[ii], axis=(2, 3, 4))) 

        nrows = 4
        ncols = 1
        fig, ax = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,figsize=(4,10))

        colors = ['blue', 'red', 'orange', 'green']
        markers = ['o', 'x']
        lss = ['-','--']
        for ii in range(4):
            for j,dyn_compl in enumerate([0,0.005]):
                which = np.where(keys_mod[:,2]==dyn_compl)[0]
                for i in range(num_lambdas):
                    ax[ii].plot(
                        keys_mod[which,0],
                        mean_values[ii][i][which],
                        marker=markers[j],
                        color=colors[i],
                        ls=lss[j],
                        label=(fr'$\lambda={lambdas[i]}$' if lambdas is not None else fr'$\lambda_{i+1}$') if ii==0 and j==0 else '__nolabel__',
                        linewidth=2,
                        markersize=6
                    )
            if ii==3:
                ax[ii].set_xlabel('weight phenotypical robustness')
                xticks = [0,0.2,0.4,0.6,0.8,1]
                xticklabels = [str(round(el*100)) + '%' for el in xticks]
                ax[ii].set_xticks(xticks)
                ax[ii].set_xticklabels(xticklabels)
                y1,y2 = ax[ii].get_ylim()
                y1 = y1 - 0.25*(y2-y1)
                for iii,x in enumerate(xticks):
                    ax[ii].text(x,y1+0.05*(y2-y1),str(xticklabels[-1-iii]),va='center',ha='center')
                ax[ii].text(0.5,y1+0.15*(y2-y1),'weight phenotypical match',va='center',ha='center')
                ax[ii].set_ylim([y1,y2])
                x1,x2 = ax[ii].get_xlim()
                ax[ii].set_xlim([x1-0.05*(x2-x1),x1+1.05*(x2-x1)])
            if measures_names is not None:
                ax[ii].set_ylabel(measures_names[ii], fontsize=13)
            ax[ii].grid(False)
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(title=r'selection strength:',
                              ncol=1,
                              frameon=False,
                              loc='center',
                              bbox_to_anchor=[1.1,0.75])
        plt.tight_layout()
        plt.show()
    

def plot_heatmap(sccs, keys,  mutation_probabilities=None, k_canalizing=None, selection_strengths=None, tag='Number of SCCs'):
    import matplotlib.colors as mcolors
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    heatmap_data = []
    if mutation_probabilities is not None:
        for i, mp in enumerate(mutation_probabilities):
            row = []
            for scc_mat in sccs[i]:
                gg = np.mean(scc_mat, axis=(0, 2))
                scc_gen0 = gg[0]
                scc_genX = gg[-1]
                lc = np.log2(scc_genX / scc_gen0)
                row.append(lc)
            heatmap_data.append(row)
            
    if k_canalizing is not None:
        for i, mp in enumerate(k_canalizing):
            row = []
            for scc_mat in sccs[i]:
                gg = np.mean(scc_mat, axis=(0, 2))
                scc_gen0 = gg[0]
                scc_genX = gg[-1]
                lc = np.log2(scc_genX / scc_gen0)
                row.append(lc)
            heatmap_data.append(row)
            
    if selection_strengths is not None:
        for i, mp in enumerate(selection_strengths):
            row = []
            for scc_mat in sccs[i]:
                gg = np.mean(scc_mat, axis=(0, 2))
                scc_gen0 = gg[0]
                scc_genX = gg[-1]
                lc = np.log2(scc_genX / scc_gen0)
                row.append(lc)
            heatmap_data.append(row)
        
    
    
    heatmap_data = np.array(heatmap_data)
    
    colors = ["red", "white", "blue"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_red_white_blue", colors)
    
    max_abs_value = np.max(np.abs(heatmap_data))
    
    xticklabels = keys
    yticklabels = []
    ylabel = ''
    if mutation_probabilities is not None:
        ylabel = "Mutation Probability (µ)"
        yticklabels = mutation_probabilities
    elif k_canalizing is not None:
        ylabel = "k (canalization)"
        yticklabels = k_canalizing
    else:
        ylabel = "selection strength (λ)"
        yticklabels = selection_strengths
        
        
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        heatmap_data, 
        cmap=custom_cmap, 
        center=0, 
        annot=True,
        fmt=".2f",  
        xticklabels=xticklabels, 
        yticklabels=yticklabels,
        vmin=-max_abs_value, 
        vmax=max_abs_value
    )
    
    cbar = ax.collections[0].colorbar
    cbar.set_label(f'Log2 Change in {tag}')
    
    plt.xlabel("α (Phenotypical Robustness, Number of Attractors, PhMatch)")
    plt.ylabel(ylabel)
    plt.title(f"Heatmap of Log2 Change in {tag}")
    plt.tight_layout()
    plt.show()


def plot_emergency_scc_convergence(sccs, all_alphas=None, selection_strengths=None, g=20):
    for index, _ in enumerate(selection_strengths):
        measure_name = 'scc'
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.set_theme(style="whitegrid")
        plt.style.use('seaborn-v0_8-whitegrid')

        colors = plt.cm.tab20(np.linspace(0, 1, len(all_alphas)))

        print(f"current index is >>> {index}")
        scc_p = sccs[index]
        for i, scc_data in enumerate(scc_p):
            if not scc_data:
                continue

            key, data = next(iter(scc_data.items()))
            alpha1, alpha2, alpha3 = map(float, key.strip('()').split(','))
            print(f"Label {i}: alpha1={alpha1}, alpha2={alpha2}, alpha3={alpha3}")

            if data.size == 0:
                continue

            avg_convergence = data.mean(axis=(0, 2))

            ax.plot(
                avg_convergence,
                color=colors[i],
                linewidth=2.5,
                marker='o',
                markersize=5,
                label=fr"$\alpha_1={round(alpha1,4)}$, $\alpha_2={round(alpha2,4)}$, $\alpha_3={round(alpha3,4)}$"
            )

        ax.text(0.02, 0.98, f'μ = 0.05, λ = {selection_strengths[index]}',
                transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))

        ax.set_title(f'Convergence of {measure_name.upper()}s Across All Alpha Combinations\nPopulation Size: 100 | Generations: {g} | Iterations: 100',
                    fontsize=14, pad=g)
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel(measure_name.upper(), fontsize=12)
        ax.set_xticks(np.arange(0, g, 2))
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            fontsize=9,
            frameon=True,
            title='Alpha Combinations',
            title_fontsize='10'
        )

        fig.tight_layout()
        plt.show()
    
def plot_selection_strength2(ranks, weights, lambdas, M=100):
    """
    Plots the relationship between rank and selection probability.

    Args:
        ranks: Not directly used for plotting the x-axis but assumed to be
               structured similarly to weights.
        weights (np.array): A multi-dimensional array, likely shaped
                            (num_lambdas, num_gens, num_sims, M).
        lambdas (list): A list of the lambda values used.
        M (int): The number of individuals in the population.
    """    
    print("Plotting selection probability...")
    M=100
    ranks = np.arange(1,M+1)
    lambdas = [0,1,3,10]
    
    
    f,ax = plt.subplots()
    ax.set_title("Roulette Wheel Selection Probability vs Rank")
    ax.set_xlabel("Individual Rank (1=Best, 100=Worst)")
    ax.set_ylabel("Probability of Selection")
    for lam in lambdas:
        weights = [((M-r)/M)**lam for r in  ranks]
        ax.plot(ranks,weights/np.sum(weights),label=f"λ = {lam}")
    ax.legend(loc='best')

    f.tight_layout()
    plt.show()




def plot_grouped_bars(sccs_for_specific_mutation_probability, keys, mu, lam):
    # Data processing remains the same
    mean_number_sccs_gen0 = []
    mean_number_sccs_genx = []
    std_number_sccs_gen0 = []
    std_number_sccs_genx = []
    
    for i, scc_mat in enumerate(sccs_for_specific_mutation_probability):
        avg_arr = np.mean(scc_mat, axis=(0, 2))
        std_arr = np.std(scc_mat, axis=(0, 2))  # Corrected from mean to std
        
        mean_number_sccs_gen0.append(avg_arr[0])
        mean_number_sccs_genx.append(avg_arr[-1])
        std_number_sccs_gen0.append(std_arr[0])
        std_number_sccs_genx.append(std_arr[-1])
    
    # Plotting setup
    width = 0.35
    x = np.arange(len(sccs_for_specific_mutation_probability))
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Prepare data for plotting
    mean_number_sccs = [mean_number_sccs_gen0, mean_number_sccs_genx]
    std_number_sccs = [std_number_sccs_gen0, std_number_sccs_genx]
    
    # Generation label helper
    def get_generation_label(index):
        if index > 0:
            return f'Gen {len(avg_arr)-1}'
        return f'Gen {index}'
    
    # Plot bars
    for i, (mean, std) in enumerate(zip(mean_number_sccs, std_number_sccs)):
        ax.bar(x + (i - 0.5)*width, mean, width=width, 
               label=get_generation_label(i), yerr=std)
    
    
    # for i, mean in enumerate(mean_number_sccs):
    #     ax.bar(x + (i - 0.5)*width, mean, width=width, 
    #            label=get_generation_label(i))
    
    # Add mu annotation
    # ax.text(0.02, 0.98, f'μ = {mu}', transform=ax.transAxes,
    #         fontsize=12, verticalalignment='top',
    #         bbox=dict(facecolor='white', alpha=0.8))
    ax.text(0.02, 0.98, f'μ = {mu}, λ = {lam}',
        transform=ax.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8))
    
    # Axis labels and title
    ax.set_ylabel('Average SCC')
    ax.set_xlabel('Alpha Combinations')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45)
    # ax.set_title(f'Emergence of Modularity {population_size} rBNs of {N} nodes {n_sim} iterations', 
    #              fontsize=14, pad=20)
    ax.set_title(f'Emergence of Modularity 100 rBNs of 10 nodes 100 iterations', 
                 fontsize=14, pad=20)
    ax.legend(title='Generations')
    
    plt.tight_layout()
    plt.show()