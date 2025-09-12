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
          ax=None, p0_mode=0,p1_mode=0,network_params=[], pred=False):
    param_names = ['tau','alpha']   
    if pred:
        sim_value = idata.predictions.sim
        switchpoint = idata.constant_data.incidence.shape[0]
    else:
        sim_value = idata.posterior_predictive.sim
        switchpoint=0
    posterior = idata.posterior.stack(samples=("draw", "chain"))
    print(p0_mode,p1_mode)
    
    with_switch,num_runs,frac,gamma,delta,n_nodes = network_params
    
    if not pred:
        #num_runs=1
        q = calibr_funcs.simulation_func(1, tau=[p0_mode], 
                                    alpha=[p1_mode], 
                                    modeling_duration=[data.shape[0]], 

                                    with_switch=with_switch,
                                    num_runs=num_runs, 
                                    frac=frac, 
                                    size=[data.shape[0]])

    
    model_time = [data.shape[0]]
    alpha_len = 1
    x = np.arange(model_time[0])
    if ax is None:
        fig, ax = plt.subplots(1,alpha_len, sharex=True, 
                               sharey=True, figsize=(6,3))
    # чтобы работало и при одном ax
    ax = np.array([ax]).flatten()

    data_part = data
    sim_part = sim_value.stack(samples=("draw", "chain"))
    i = 0
    # posterior predictive lines
    l0 = ax[i].plot(sim_part[:switchpoint], 
                    color='gray', alpha=0.05) 
    l00 = ax[i].plot(np.arange(switchpoint, 
                               data.shape[0]),
                     sim_part[switchpoint:], 
                    color='RoyalBlue', alpha=0.05) 
    next_c = 'green'
    
    if not pred:
        # from mode params
        r2_part = r2_score(data_part, 
                             q)
        ax[i].plot(q,
                 color='white', lw=4, ls='-',
                 )
        l1=ax[i].plot(q,
                 color='green', lw=2, ls='-',
                 label=r'Simulation ($R^2$' +f' = {r2_part:.3f})')
        next_c='blue'
    

    # real data
    '''
    ax[i].plot(data_part, "", ls='-', lw=4, 
              color='white')
    '''
    l2=ax[i].scatter(np.arange(switchpoint), 
                     data_part[:switchpoint], 

                  color='green', s=30,
                 edgecolors='white', zorder=99)

    l3=ax[i].scatter(np.arange(switchpoint, 
                               data.shape[0]),
                  data_part[switchpoint:], 
                  color='OrangeRed', s=20,
               label='True incidence',
                 edgecolors='white',
                    alpha=1, zorder=99)

    if switchpoint > 0:
        ax[i].axvline(switchpoint, ls=':', color='gray',
                 lw=3)        
    ax[i].set_title("Time series comparison")
    ax[i].set_xlabel('Time')
    ax[i].set_ylabel('Incidence')

    #ax[i].set_xlim(-5, data.shape[0])#np.where(data==0)[0][0]*1.1)
    ax[i].grid()
    ax[i].set_title(title)
    ax[i].legend();

    if return_r2:
        return r2_part
    
    if with_trace:
        pm.plot_trace(idata);
        az.plot_posterior(idata);
        return az.summary(idata)
    

def calc_stat(posterior, param_names):
    rr = 2
    p0_mode = stats.mode(posterior[param_names[0]
                                  ].round(rr))[0]
    p1_mode = stats.mode(posterior[param_names[1]
                                  ].round(rr))[0]
    '''
    p0_mode = posterior[param_names[0]
                       ].quantile(.5).values
    p1_mode = posterior[param_names[1]
                       ].quantile(.5).values
    
    p0_mode = posterior[param_names[0]
                       ].mean().values
    p1_mode = posterior[param_names[1]
                       ].mean().values
    '''
    return p0_mode, p1_mode


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
    #results = idata.posterior
    n_chains = idata.sample_stats.chain.shape[0]
    
    param_names = ['tau','alpha']   
    fancy_names = [r'$\beta$', r'$\alpha$']
    
    posterior = idata.posterior.stack(samples=("draw", "chain"))
    p0_mode, p1_mode = calc_stat(posterior, param_names)
    
    # ____ Accepted parameters ____
    ax_i = axes[0]
    label='Value'
    #for i in range(n_chains):
    ax_i.scatter(posterior[param_names[0]], 
                 posterior[param_names[1]],
                  alpha=.2, s=40, label='Value',
                 color='RoyalBlue')    
    # 'true' parameters   
    '''
    ax_i.scatter(true_tau, true_alpha, alpha=0.9, 
                    color='white',
                 s=60)
    '''
    ax_i.scatter(true_tau, true_alpha, alpha=0.9, 
                    color='OrangeRed', label='Observed', 
                 edgecolors='white',
                 s=50)
    
    # __ mode params
    '''
    ax_i.scatter(p0_mode, p1_mode, alpha=0.9, 
                    color='white',
                 s=60)
    '''
    ax_i.scatter(p0_mode, p1_mode, alpha=0.9, 
                    color='green', label='Chosen', 
                 edgecolors='white',
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
    
    lims=[]
    # _______ Posterior distribution 
    for i, pname, fname, pval, pmode in zip(np.arange(2),
                               param_names[::-1],
                                fancy_names[::-1],
                              [true_tau, true_alpha][::-1],
                              [p0_mode,p1_mode][::-1]):
        
        ax_i = axes[i+2]
        q = sns.histplot(posterior[pname], alpha=0.5, 
                     color='RoyalBlue', bins=50, kde=False,
                     stat='probability', edgecolor=None, ax = ax_i)
        
        vals = [c.get_height() for c in q.containers[0].patches]
        lims.append(max(vals))

        ax_i.set_ylabel('Frequency')
        ax_i.axvline(pval, ls='--', color='white',
                            lw=4)
        ax_i.axvline(pval, ls='--', color='OrangeRed',
                      lw=2,label='Observed')
        
        ax_i.axvline(pmode, ls='--', color='white',
                            lw=4)
        ax_i.axvline(pmode, ls='--', color='green',
                      lw=2,  label='Chosen')
        ax_i.set_title(fname+", posterior distribution")
        ax_i.set_xlabel(fname)
        ax_i.legend()
        ax_i.grid()
    
    for i in range(2):
        ax_i = axes[i+2]
        ax_i.set_ylim(0, max(lims)*1.1)
        
    plt.tight_layout()
    

def pred_calib(observed_data, idata, 
                 true_tau, true_alpha, network_params,
                  method_name='ABC SMC'
                 ):
    """
    Plot parameter posterior and time series comparison
    """
    
    pred=True 
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    observed_clm = observed_data.columns[0]
    results = idata.posterior
    n_chains = idata.sample_stats.chain.shape[0]
    
    param_names = ['tau','alpha']   
    fancy_names = [r'$\beta$', r'$\alpha$']
    
    posterior = idata.posterior.stack(samples=("draw", "chain"))
    p0_mode, p1_mode = calc_stat(posterior, param_names)
    
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
          network_params=network_params, pred=True)
    ax_i.set_title('Time series comparison')
    
    
    # _______ Posterior distribution 
    lims = []
    for i, pname, fname, pval, pmode in zip(np.arange(2),
                               param_names[::-1], 
                                fancy_names[::-1],
                              [true_tau, true_alpha][::-1],
                              [p0_mode,p1_mode][::-1]):
        
        ax_i = axes[i+2]
        q = sns.histplot(posterior[pname], alpha=0.5, 
                     color='RoyalBlue', bins=50, kde=False,
                     stat='probability', edgecolor=None, ax = ax_i)
        
        vals = [c.get_height() for c in q.containers[0].patches]
        lims.append(max(vals))

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
    
    for i in range(2):
        ax_i.set_ylim(max(lims)*1.1)
        
    plt.tight_layout()     