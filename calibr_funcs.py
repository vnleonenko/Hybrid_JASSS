import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.stats import uniform, norm, multivariate_normal
import multiprocessing as mp
from functools import partial
import seaborn as sns 
import plot_funcs
from networks.model_output import SEIRModelOutput, SEIRParams
from networks.SEIR_network import SEIRNetworkModel

from hybrid_network_to_seir import NetworkSEIR_tuned,\
                                    generate_synthetic_data
import seir_discrete
import predict_Beta_I
#pip install -e git+https://github.com/Mpkosh/Mathematics-of-Epidemics-on-Networks.git@my_changes#egg=eon


def switch_seir(sim_data, gamma=0.1, delta=0.08,
                frac=0.01, modeling_duration=249, 
                method='expanding'):
    
    pop = sim_data.iloc[0,:4].sum()
    '''
    switches = sim_data[sim_data['I'] > pop*frac]
    
    if switches.shape[0]:
        switch_day = switches.index[0]
    else:
        switch_day = 7
    '''
    if sim_data.shape[0]>1:
        switch_day = sim_data.shape[0]-1
    else:
        switch_day = 1
    #print(switch_day)
    y0 = sim_data.iloc[switch_day,:4].values.flatten()
    # FOR FULL OBSERVED DATA
    ts = np.arange(modeling_duration-switch_day)
    if method=='expanding':
        beta = sim_data.iloc[:switch_day]['Beta'
                                ].expanding(1).mean().values[-1]
    elif method=='last':
        beta = sim_data.iloc[:switch_day]['Beta'].values[-1]
        
        S,E,I,R = seir_discrete.seir_model(y0, ts, beta, 
                                           gamma, delta, 
                                           stype='d', 
                                           beta_t=False).T
    elif method=='lstm':
        _, beta, _ = predict_Beta_I.predict_beta('', 
                      sim_data,
                     'lstm', 
                     np.arange(switch_day, 
                               modeling_duration),                                         stochastic=False,
                            count_stoch_line=0, 
                 sigma=0, gamma=0, 
                     model_path='ba100k_lstm_4_001_s10', 
                     window_size=4,
                    modeling_duration=modeling_duration)

        S,E,I,R = seir_discrete.seir_model(y0, ts, beta, 
                                           gamma, delta, 
                                           stype='d', 
                                           beta_t=True).T

    fin = pd.DataFrame([S,E,I,R]).T
    fin.columns = ['S','E','I','R']

    fin['Beta'] = beta
    fin['day'] = ts+switch_day
    fin = pd.concat([sim_data.iloc[:switch_day], fin]
                   ).reset_index(drop=True)
    return fin


def simulation_func(rng, tau, alpha, modeling_duration, 
                    with_switch=False, num_runs=[1], 
                    frac=[0.01], with_df=False,
                    size=None):
    
    tau = np.array(tau).flatten()[0]
    alpha = np.array(alpha).flatten()[0]
    
    modeling_duration = np.array(modeling_duration).flatten()[0]#np.array(modeling_duration).flatten()[0]
    #print(modeling_duration)
    #print(tau, alpha, modeling_duration, frac, size)
    num_runs = np.array(num_runs).flatten()[0] 
    frac = np.array(frac).flatten()[0] 
    
    method='lstm'
    network_type='ba'    
    gamma = 0.3
    delta=0.2
    n_nodes=100000
    
    chosen_seed = np.random.RandomState(42)
    network_model = SEIRNetworkModel(n_nodes, network_type, 
                                     chosen_seed)
    init_inf_frac = 0.0001
    init_rec_frac = 1 - alpha

    all_results = []
    for run in range(np.array(num_runs).flatten()[0]):
        res,rt,ri = network_model.simulate(beta=tau, gamma=gamma, delta=delta, 
                                     init_inf_frac=init_inf_frac, 
                                     init_rec_frac=init_rec_frac,
                                     tmax=modeling_duration,
                                     I_frac_switch=frac,
                                     frac_pop='Infected'
                                    )
        #print(res.I.shape)
        seed_df = pd.DataFrame([res.S, res.E, res.I, res.R]).T
        seed_df.columns = ['S','E','I','R']
        seed_df['day'] = np.arange(seed_df.shape[0])
        # use "values", because "iloc" saves index info 
        # and messes with the calculation
        beta_calc = - seed_df.S.diff().values[1:] / (
                                seed_df.S.values[:-1] * seed_df.I.values[:-1]
                                )
        # the last Beta value cannot be calculated: no S_{t+1}
        seed_df['Beta'] = [*beta_calc, beta_calc[-1]] 
        #print(seed_df.Beta.iloc[-3:].values)
        s = seed_df.shape[0]
        #print(s)
        seed_df.fillna(0, inplace=True)
        
        if res.I.shape[0] < modeling_duration:
            if np.array(with_switch).flatten()[0]:
                seed_df = switch_seir(seed_df, 
                                      gamma=gamma,
                                      delta=delta,
                                      
                                      frac=np.array(frac
                                        ).flatten()[0],
                                          modeling_duration=modeling_duration, 
                                      method=method)
        # sometimes seed_df.shape[0] > modeling duration (is 150, not 149!)
        #print('seed_df ',seed_df.shape)
        # calculating incidence
        temp = seed_df[['E','S']].shift([0,1])
        seed_df['incidence'] = (temp['E_1'] - temp['E_0']) - \
                            (temp['S_0'] - temp['S_1'])
        #print(seed_df.iloc[s-5:s+2])
        seed_df['incidence'].fillna(0, inplace=True)
        all_results.append(seed_df)    

    if all_results:
        combined_results = pd.concat(all_results, 
                                     ignore_index=True)
        seed_df = combined_results.groupby('day').mean().reset_index()
        seed_df[['S','E','I','R',
                    'incidence']] = seed_df[['S','E','I','R',
                                             'incidence']].round()
    
    #print(seed_df['incidence'].astype(int))
    res = seed_df['incidence'
                 ].fillna(0).astype(int).values[:modeling_duration]
    #res = stz(res)
    
    if np.array(with_df).flatten()[0]:
        return res, seed_df
    else:
        return res
    
    

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