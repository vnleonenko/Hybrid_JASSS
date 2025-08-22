import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.stats import uniform, norm, multivariate_normal
from main_pool import Main
import multiprocessing as mp
from functools import partial
import seaborn as sns 
import plot_funcs


def plot_results(observed_data, abc_results, 
                 method_name='', n_trajectories=5,
                 true_p0=0, true_p1=0, hm_results=[]):
    """
    Plot parameter posterior and time series comparison
    """
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    observed_clm = observed_data.columns[0]
    
    if len(abc_results) == 0:
        axes[0].text(0.5, 0.5, f"No accepted parameter sets for {method_name}", 
                    horizontalalignment='center', verticalalignment='center')
        axes[1].text(0.5, 0.5, f"No accepted parameter sets for {method_name}",
                    horizontalalignment='center', verticalalignment='center')
    else:
        # ____ Accepted parameters ____
        if 'weight' in abc_results.columns:
            s=abc_results["weight"]*100
        else:
            s=30
        #s=30
        
        param_names = abc_results.columns.drop(['distance',
                                                'trajectory']).values   
        
        scatter = axes[0].scatter(abc_results[param_names[0]], 
                                  abc_results[param_names[1]], 
                                  alpha=0.6, s=s, label='Accepted',
                                 color='RoyalBlue')    
        # 'true' parameters    
        axes[0].scatter(true_p0, true_p1, alpha=0.9, 
                        color='OrangeRed', label='Observed', s=50)

        axes[0].set_title(f"Accepted parameters - {method_name}")
        axes[0].set_xlabel(r'$\beta$')
        axes[0].set_ylabel(r'$\alpha$')
        
        if len(hm_results):
            axes[0].set_xlim(hm_results[param_names[0]].quantile(.2),
                            hm_results[param_names[0]].quantile(.8))
            axes[0].set_ylim(hm_results[param_names[1]].quantile(.2),
                            hm_results[param_names[1]].quantile(.8))
        axes[0].legend()
        axes[0].grid()

        # ____ Trajectories from accepted parameters ____
        n_plot = min(n_trajectories, len(abc_results))
        
        for i in range(n_plot):
            traj = abc_results.iloc[i]["trajectory"]
            if i==0:
                label='Accepted trajectory'
            else:
                label=''
            axes[1].plot(traj, alpha=0.1, label=label, color='RoyalBlue')
           
        #print(abc_results.iloc[0]["trajectory"])
        '''
        axes[1].plot(abc_results.iloc[:n_plot]["trajectory"],
                     label='Accepted trajectory', alpha=0.4)
        '''
        # plot time series (of observed)
        axes[1].plot(#np.tile(observed_data["H1N1"], 5), 
                    observed_data[observed_clm],
                     label="Observed", color="OrangeRed", marker='.', linestyle="-")
        axes[1].set_title("Time series comparison")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Infected")
        axes[1].legend()
        axes[1].grid()

        stop = np.array([])#observed_data[observed_data[observed_clm]==0].index
        if stop.shape[0]:
            axes[1].set_xlim(-5, stop[0]+10)
        else:
            axes[1].set_xlim(-5, observed_data.shape[0]+10)

        # ____ Posterior destribution of parameter 1 ____
        plt.style.use("default")
        '''
        axes[2].hist(abc_results[param_names[0]], alpha=0.4, color='gray')
        sns.kdeplot(abc_results[param_names[0]], color="RoyalBlue", 
                    shade=True, ax=axes[2])
        '''
        
        sns.histplot(abc_results[param_names[0]], alpha=0.4, 
                     color='RoyalBlue', bins=min(abc_results.shape[0]+1, 30), 
                     kde=True, stat='probability', edgecolor=None, 
                     ax=axes[2])
        
        axes[2].axvline(true_p0, ls='--', color='OrangeRed',
                        label='Observed')
        axes[2].set_title(r'$\beta$'+", posterior destribution")
        axes[2].set_xlabel(r'$\beta$')
        axes[2].legend()
        axes[2].grid()

        # ____ Posterior destribution of parameter 2 ____
        plt.style.use("default")
        sns.histplot(abc_results[param_names[1]], alpha=0.4, 
                     color='RoyalBlue', bins=min(abc_results.shape[0]+1, 30), 
                     kde=True, stat='probability', edgecolor=None, 
                     ax=axes[3])
        axes[3].axvline(true_p1, ls='--', color='OrangeRed',
                        label='Observed')
        axes[3].set_title(r'$\alpha$'+", posterior destribution")
        axes[3].set_xlabel(r'$\alpha$')
        axes[3].legend()
        axes[3].grid()

    plt.tight_layout()



    
    
def posterior_params(gs, idata, observed_data, true_tau, true_alpha):
    #plt.style.use("default")
    
    fig = plt.figure(figsize=(10,6))
    
    observed_clm = observed_data.columns[0]
    n_chains = idata.sample_stats.chain.shape[0]
    
    #_____________
    ax_i = plt.subplot(gs[0, 1:3])
    for i in range(n_chains):
        ax_i.scatter(idata.posterior.tau[i], 
                      idata.posterior.alpha[i], 
                      alpha=1/(n_chains+1), s=40, label='Accepted',
                     color='RoyalBlue')    
    # 'true' parameters    
    ax_i.scatter(true_tau, true_alpha, alpha=0.9, 
                    color='OrangeRed', label='Observed', s=50)

    ax_i.set_title(f"Accepted parameters")
    ax_i.set_xlabel(r'$\beta$')
    ax_i.set_ylabel(r'$\alpha$')
    
    #_____________
    ax_i = plt.subplot(gs[1, 0:2])
    for i in range(n_chains):
        sns.histplot(idata.posterior.tau[i], alpha=1/(n_chains+1), 
                         color='RoyalBlue', bins=30,
                         kde=True, stat='probability', edgecolor=None, 
                         ax = ax_i)
    ax_i.axvline(true_tau, ls='--', color='OrangeRed',
                        label='Observed')
    ax_i.set_title(r'$\beta$'+", posterior destribution")
    ax_i.set_xlabel(r'$\beta$')
    ax_i.legend()
    ax_i.grid()

    # ____ Posterior destribution of parameter 2 ____
    ax_i = plt.subplot(gs[1, 2:4])
    for i in range(n_chains):
        sns.histplot(idata.posterior.alpha[i], alpha=1/(n_chains+1), 
                     color='RoyalBlue', bins=30,
                     kde=True, stat='probability', edgecolor=None, 
                     ax = ax_i)
    ax_i.axvline(true_alpha, ls='--', color='OrangeRed',
                    label='Observed')
    ax_i.set_title(r'$\alpha$'+", posterior destribution")
    ax_i.set_xlabel(r'$\alpha$')
    ax_i.legend()
    ax_i.grid()
    
    plt.tight_layout()