import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error as rmse
from tqdm.notebook import tqdm, trange
#from model_complex import Calibration, EpidData, FactoryBRModel
from timeit import default_timer as timer
import pymc as pm
import arviz as az
import seaborn as sns 
import scipy.stats as stats
import calibr_funcs


# _____________________ FOR CALIBRATION    

def plots(idata, data, title, with_trace=False, 
          show_values=True, for_pred=False, return_r2=False, 
          ax=None, p0_mode=0,p1_mode=0,network_params=[]):
    param_names = ['tau','alpha']   
    sim_value = idata.posterior_predictive.sim
    posterior = idata.posterior.stack(samples=("draw", "chain"))
    print(p0_mode,p1_mode)
    
    with_switch,num_runs,frac,gamma,delta,n_nodes = network_params
    
    if with_switch:
    #num_runs=1
        q = calibr_funcs.simulation_func(1, tau=[p0_mode], 
                                    alpha=[p1_mode], 
                                    modeling_duration=[data.shape[0]], 

                                    with_switch=with_switch,
                                    num_runs=num_runs, 
                                    frac=frac, 
                                    size=[data.shape[0]])
    
    q_n = calibr_funcs.simulation_func(1, tau=[p0_mode], 
                                alpha=[p1_mode], 
                                modeling_duration=[data.shape[0]], 

                                with_switch=[False],
                                num_runs=[1], 
                                frac=[1.], 
                                size=[data.shape[0]])
    
    model_time = [data.shape[0]]
    alpha_len = 1
    x = np.arange(model_time[0])
    if ax is None:
        fig, ax = plt.subplots(1,alpha_len, sharex=True, 
                               sharey=True, figsize=(6,3))
    # чтобы работало и при одном ax
    ax = np.array([ax]).flatten()
    # для каждой возрастной группы
    for i in range(1):
        data_part = data[i*model_time[0]:i*model_time[0]+model_time[0]]
        sim_part = sim_value[:,
                             :,
                             i*model_time[0]:i*model_time[0]+model_time[0]]
        
        
        
        # posterior predictive lines
        l0 = ax[i].plot(np.array(sim_part
                           ).reshape(-1,len(data_part)).T, 
                 color='RoyalBlue', alpha=0.1) 
        next_c = 'green'
        
        if with_switch:
            # from mode params
            r2_part = r2_score(data_part, 
                                 q)
            ax[i].plot(q,
                     color='white', lw=4, ls='-',
                     )
            l1=ax[i].plot(q,
                     color='green', lw=2, ls='-',
                     label=r'Hybrid ($R^2$' +f' = {r2_part:.3f})')
            next_c='blue'
        
        # from mode params
        r2_part = r2_score(data_part, 
                             q_n)
        ax[i].plot(q_n,
                 color='white', lw=4, ls='-',
                 )
        l11=ax[i].plot(q_n,
                 color=next_c, lw=2, ls='-',
                 label=r'Network ($R^2$' +f' = {r2_part:.3f})')
        
        
        # real data
        ax[i].plot(data_part, "", ls='-', lw=4, 
                  color='white')
        l2=ax[i].plot(data_part, "", ls='-', color='OrangeRed', lw=2,
                   #markeredgecolor='white', 
                   label='True incidence')
        '''
        ci = .95
        # posterior predictive ci
        ax[i].fill_between(x=x,
                         y1=sim_part.quantile(q=round(0.5 - ci/2, 3), 
                                              dim=['chain', 'draw']), 
                         y2=sim_part.quantile(q=round(0.5 + ci/2, 3), 
                                              dim=['chain', 'draw']),
                         color='skyblue', alpha=.5, label=f'CI {ci*100:.0f}%')
        '''                 
        ax[i].set_title("Time series comparison")
        ax[i].set_xlabel('Time')
        ax[i].set_ylabel('Incidence')
        
        ax[i].set_xlim(-5, data.shape[0])#np.where(data==0)[0][0]*1.1)
        ax[i].grid()
        ax[i].set_title(title)
        ax[i].legend();
    
    if return_r2:
        return r2_part
    
    if with_trace:
        pm.plot_trace(idata);
        az.plot_posterior(idata);
        return az.summary(idata)
    
    
def results_calib(observed_data, idata, 
                 true_tau, true_alpha, network_params,
                  method_name='ABC SMC'
                 ):
    """
    Plot parameter posterior and time series comparison
    """
    
    
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    observed_clm = observed_data.columns[0]
    results = idata.posterior
    n_chains = idata.sample_stats.chain.shape[0]
    
    param_names = ['tau','alpha']   
    fancy_names = [r'$\beta$', r'$\alpha$']
    
    posterior = idata.posterior.stack(samples=("draw", "chain"))
    rr = 2
    p0_mode = stats.mode(posterior[param_names[0]].round(rr))[0]
    p1_mode = stats.mode(posterior[param_names[1]].round(rr))[0]
    
    
    # ____ Accepted parameters ____
    ax_i = axes[0]
    label='Value'
    for i in range(n_chains):
        ax_i.scatter(results[param_names[0]], 
                     results[param_names[1]],
                      alpha=1/(n_chains+1), s=40, label=label,
                     color='RoyalBlue')    
        label=''
    # 'true' parameters   
    ax_i.scatter(true_tau, true_alpha, alpha=0.9, 
                    color='white',
                 s=60)
    ax_i.scatter(true_tau, true_alpha, alpha=0.9, 
                    color='OrangeRed', label='Observed', 
                 #edgecolors='white',
                 s=50)
    
    # __ mode params
    ax_i.scatter(p0_mode, p1_mode, alpha=0.9, 
                    color='white',
                 s=60)
    ax_i.scatter(p0_mode, p1_mode, alpha=0.9, 
                    color='green', label='Param mode', 
                 #markeredgecolor='white',
                 s=50)

    ax_i.set_title(f"Values from posterior distribution - {method_name}")
    ax_i.set_xlabel(fancy_names[0])
    ax_i.set_ylabel(fancy_names[1])

    ax_i.legend()
    ax_i.grid()
    
    # ____ curves
    ax_i = axes[1]
    plots(idata, observed_data['incidence'].values, '', 
           ax=ax_i, p0_mode=p0_mode,
          p1_mode=p1_mode, 
          network_params=network_params)
    ax_i.set_title('Time series comparison')
    
    
    # _______ Posterior distribution 
    for i, pname, fname, pval, pmode in zip(np.arange(2),
                               param_names, fancy_names,
                              [true_tau, true_alpha],
                              [p0_mode,p1_mode]):
        
        ax_i = axes[i+2]
        sns.histplot(posterior[pname], alpha=0.5, 
                     color='RoyalBlue', bins=50, kde=False,
                     stat='probability', edgecolor=None, ax = ax_i)
        ax_i.set_ylabel('Frequency')
        ax_i.axvline(pval, ls='--', color='white',
                            lw=4)
        ax_i.axvline(pval, ls='--', color='OrangeRed',
                      lw=2,label='Observed')
        
        ax_i.axvline(pmode, ls='--', color='white',
                            lw=4)
        ax_i.axvline(pmode, ls='--', color='green',
                      lw=2,  label='Mode')
        ax_i.set_title(fname+", posterior distribution")
        ax_i.set_xlabel(fname)
        ax_i.legend()
        ax_i.grid()

    plt.tight_layout()    
    
