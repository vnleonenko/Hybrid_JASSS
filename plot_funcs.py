import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib as mpl
import matplotlib.patches as mpatches

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
from arviz.stats.density_utils import _fast_kde_2d, \
                                      _find_hdi_contours
from shapely.geometry import Polygon


# _____________________ FOR CALIBRATION    

def plots(idata, data, title, with_trace=False, 
          show_values=True, return_r2=False, 
          ax=None, p0_mode=0,p1_mode=0,network_params=[], pred=False):
    param_names = ['tau','alpha']   
    if pred:
        sim_value = idata.predictions.sim
        switchpoint = idata.constant_data.incidence.shape[0]
        fin_size = switchpoint+7
    else:
        sim_value = idata.posterior_predictive.sim
        switchpoint=0
        fin_size = data.shape[0]
        
    posterior = idata.posterior.stack(samples=("draw", "chain"))
    print(p0_mode,p1_mode)
    
    with_switch,num_runs,frac,gamma,delta,n_nodes = network_params
    data_part = data
    model_time = [data.shape[0]]
    alpha_len = 1
    x = np.arange(model_time[0])
    if ax is None:
        fig, ax = plt.subplots(1,alpha_len, sharex=True, 
                               sharey=True, figsize=(6,3))
    # чтобы работало и при одном ax
    ax = np.array([ax]).flatten()
    i = 0
   
    
    sim_part = sim_value.stack(samples=("draw", "chain"))
   
    # posterior predictive lines
    l0 = ax[i].plot(sim_part[:switchpoint], 
                    color='gray', alpha=0.05) 
    l00 = ax[i].plot(np.arange(switchpoint, 
                               fin_size),
                     sim_part[switchpoint:fin_size], 
                    color='RoyalBlue', alpha=0.05) 
    if pred:
        label='Forecast'
    else:
        label='Simulation'
        
    ax[i].plot(np.arange(switchpoint, fin_size),
            sim_part[switchpoint:fin_size,0], 
                   label=label, alpha=0.5)
    next_c = 'green'
    
    all_r = []
    all_q = []    
    if not pred:
        for j in range(1):
        #num_runs=1
            q = calibr_funcs.simulation_func(1, tau=[p0_mode], 
                                    alpha=[p1_mode], 
                                    modeling_duration=[data.shape[0]], 

                                    with_switch=with_switch,
                                    num_runs=num_runs, 
                                    frac=frac, 
                                    size=[data.shape[0]])
            all_q.append(q)
            r2_part = r2_score(data_part, q)
            all_r.append(r2_part)
            
            #l1=ax[i].plot(q,color='tab:green', lw=2, ls='-')
    
        ax[i].fill_between(x=np.arange(data.shape[0]),
                   y1 = np.array(all_q).min(axis=0),
                   y2 = np.array(all_q).max(axis=0),
                   color='tab:green', alpha=.5, zorder=99,
                   label='Simulations with selected params')

        best_idx = np.argmax(all_r)        
        ax[i].plot(all_q[best_idx],
                     color='white', lw=4, ls='-',
                     zorder=98)
        l1=ax[i].plot(all_q[best_idx],
                     color='tab:green', lw=2, ls='-',
                     label=r'Best simulation ($R^2$' +\
                      f' = {all_r[best_idx]:.3f})',
                     zorder=99)
        
    next_c='blue'

    # real data
    '''
    ax[i].plot(data_part, "", ls='-', lw=4, 
              color='white')
    '''
    if pred:
        l2=ax[i].scatter(np.arange(switchpoint), 
                         data_part[:switchpoint], 
                      color='tab:green', s=20,
                     edgecolors='white', zorder=100,
                        label='Known data')

        l3=ax[i].scatter(np.arange(switchpoint, 
                                   data.shape[0]),
                      data_part[switchpoint:], 
                      color='gray', s=20,
                   label='Unknown data',
                     edgecolors='white',
                        alpha=1, zorder=100)
        ax[i].axvline(switchpoint, ls='--', color='gray',
                      lw=2#, label='Forecast starts'
                     )       
        
    else:
        l3=ax[i].scatter(np.arange(switchpoint, 
                                   data.shape[0]),
                      data_part[switchpoint:], 
                      color='OrangeRed', s=20,
                   label='Observed incidence',
                     edgecolors='white',
                        alpha=1, zorder=100)

         
    #ax[i].set_title("Time series comparison")
    ax[i].set_xlabel('Time, days')
    ax[i].set_ylabel('Incidence, cases')

    #ax[i].set_xlim(-5, data.shape[0])#np.where(data==0)[0][0]*1.1)
    ax[i].grid()
    #ax[i].set_title(title)
    ax[i].legend();
    
    #return all_q, all_r

    if return_r2:
        return r2_part
    
    if with_trace:
        pm.plot_trace(idata);
        az.plot_posterior(idata);
        return az.summary(idata)
    
    

def calc_stat(idata, param_names):
    '''
    p0_mode = az.hdi(idata.posterior[param_names[0]], 
                     hdi_prob=0.01)[param_names[0]].mean()
    p1_mode = az.hdi(idata.posterior[param_names[1]], 
                     hdi_prob=0.01)[param_names[1]].mean() 
    '''    
    gridsize = (128, 128)
    density, xmin, xmax, ymin, ymax = _fast_kde_2d(idata.posterior[
                                                        param_names[0]],
                                                   idata.posterior[
                                                        param_names[1]],
                                                   gridsize=gridsize)
    hdi_probs=[0.2]
    # Calculate contour levels and sort for matplotlib
    contour_levels = _find_hdi_contours(density, hdi_probs)
    #contour_levels.sort()

    contour_level_list = list(contour_levels) + [density.max()]
    contour_kwargs = {'levels':contour_level_list}
    
    g_s = complex(gridsize[0])
    x_x, y_y = np.mgrid[xmin:xmax:g_s, ymin:ymax:g_s]
    fig, ax = plt.subplots(1,1)

    cs = ax.contour(x_x, y_y, density, **contour_kwargs)
    plt.close()
    
    n_different_areas = len(cs.allsegs[0])
    
    centroid = Polygon(np.array(cs.allsegs[0]).reshape(-1,2)
                      ).centroid
    p0_mode, p1_mode = centroid.x, centroid.y
     

    return p0_mode, p1_mode


def results_calib(observed_data, idata, 
                 true_tau, true_alpha, network_params,
                  method_name='ABC SMC'
                 ):
    """
    Plot parameter posterior and time series comparison
    """
    
    
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()
    observed_clm = observed_data.columns[0]
    #results = idata.posterior
    n_chains = idata.sample_stats.chain.shape[0]
    66
    param_names = ['tau','alpha']   
    fancy_names = [r'$\beta_n$', r'$\alpha$']
    
    posterior = idata.posterior.stack(samples=("draw", "chain"))
    p0_mode, p1_mode = calc_stat(idata, 
                                 param_names)
    
    # ____ Accepted parameters ____
    ax_i = axes[0]
    label='Value'
    #for i in range(n_chains):
    ax_i.scatter(posterior[param_names[0]], 
                 posterior[param_names[1]],
                  alpha=.05, s=40, label='Value',
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

    #ax_i.set_title(f"Values from posterior distribution - {method_name}")
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
    #ax_i.set_title('Time series comparison')
    
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
        #ax_i.set_title(fname+", posterior distribution")
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
    p0_mode, p1_mode = calc_stat(idata, param_names)
    
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
    

def plot_calib(observed_data, idata, 
               true_tau, true_alpha, 
               network_params, pred=False):
    cmap = mpl.colormaps['viridis']
    hdi_list = [0.2,0.5,0.8,0.9]
    colors_l = cmap(np.linspace(0, 1, len(hdi_list)))

    # for edgecolor to have alpha
    fc=to_rgba('RoyalBlue', 0.5)
    param_names = ['tau','alpha']  
    fancy_names = [r'$\beta_n$', r'$\alpha$']

    fig = plt.figure(figsize=(10,5))
    # adding gridspec
    gs = fig.add_gridspec(1, 2, hspace=0.6, width_ratios=[1,1.25])
    # dividing it even further!
    gs0 = gs[0].subgridspec(2, 1, height_ratios=[1, 4], hspace=0.)
    gs1 = gs[1].subgridspec(2, 2, wspace=0, hspace=0.,
                            width_ratios=[5, 1],
                            height_ratios=[1, 4])

    # creating subplots, ignoring useless corners
    ax_curves = fig.add_subplot(gs0[1])

    ax_up = fig.add_subplot(gs1[0])
    ax_scatter = fig.add_subplot(gs1[2], sharex=ax_up)
    ax_right = fig.add_subplot(gs1[3], sharey=ax_scatter)

    # plotting UFO
    q = az.plot_pair(
        idata,
        var_names=["tau", "alpha"],
        kind=["scatter", "kde"],
        kde_kwargs={"fill_last": False, 
                    'hdi_probs':hdi_list,
                    'fill_kwargs':{'alpha': .1},
                    'contour_kwargs':{"colors":None},
                    'contourf_kwargs':{"alpha":0.3, 
                                       'colors':[colors_l[-1],
                                                 *colors_l[:-1]]
                                      }},
        marginals=True,
        #point_estimate="mode",
        #reference_values={'tau':true_tau, 'alpha': true_alpha},
        #reference_values_kwargs={'color':'red'},
        marginal_kwargs={'kind':'hist','hist_kwargs':{'bins':50,
                                                      'color':fc,
                                                     #'alpha':.5,
                                                     'ec':fc}},
        scatter_kwargs={'color':fc},
        ax=np.array([[ax_up,None],[ax_scatter,ax_right]])
    )
    print()
    # removing ticks from small plots
    for a in [ax_up, ax_right]:
        plt.setp(a.get_xticklabels(), visible=False)
        plt.setp(a.get_yticklabels(), visible=False)
    # setting normal ticks for a scatterplot
    min_x = idata.posterior[param_names[0]].min().round(1)
    min_y = idata.posterior[param_names[1]].min().round(1)
    
    ax_scatter.set_xticks(np.arange(min_x,1,0.2), 
                          np.arange(min_x,1,0.2).round(1),
                          fontsize=10)
    ax_scatter.set_yticks(np.arange(min_y,1,0.1),
                         np.arange(min_y,1,0.1).round(1),
                          fontsize=10)


    p0_mode, p1_mode = calc_stat(idata,
                                 param_names)

    # i don't know if there can be multiple ref points, so it's easier    
    ls1 = ax_scatter.scatter(true_tau, true_alpha, alpha=0.9, 
                        color='OrangeRed', label='Observed', 
                     edgecolors='white',
                     s=50, zorder=99)

    ls2 = ax_scatter.scatter(p0_mode, p1_mode, alpha=0.9, 
                        color='tab:green', label='Selected', 
                     edgecolors='white',
                     s=50, zorder=99) 
    ls3 = ax_scatter.scatter(true_tau, true_alpha, zorder=0, s=5,
               color=fc, label='Simulation')
    
    ax_scatter.set_xlabel(fancy_names[0], fontsize=12)
    ax_scatter.set_ylabel(fancy_names[1], fontsize=12)
    
    
    legend_elements= []
    for c, val in zip(colors_l[::-1], hdi_list):
        legend_elements.append(mpatches.Patch(color=c,
                                              label=f'HDR {int(val*100)}%',
                                             alpha=.9)
                              )

    ax_scatter.legend(handles=[ls1,ls2,ls3,
                               *legend_elements])
    ax_scatter.grid()

    ax_up.axvline(p0_mode, ls='-', color='white', lw=3)
    ax_up.axvline(p0_mode, ls='--', color='tab:green')
    ax_up.axvline(true_tau, ls='-', color='white', lw=3)
    ax_up.axvline(true_tau, ls='--', color='OrangeRed')

    ax_right.axhline(p1_mode, ls='-', color='white', lw=3)
    ax_right.axhline(p1_mode, ls='--', color='tab:green')
    ax_right.axhline(true_alpha, ls='-', color='white', lw=3)
    ax_right.axhline(true_alpha, ls='--', color='OrangeRed')

    plots(idata, observed_data, '',  
          ax=ax_curves, p0_mode=p0_mode,p1_mode=p1_mode,
          network_params=network_params,
         pred=pred)

    plt.tight_layout()