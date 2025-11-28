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
from shapely.geometry import Polygon, MultiPolygon 



# _____________________ FOR CALIBRATION    

def plots(idata, data, title, with_trace=False, 
          show_values=True, return_r2=False, 
          ax=None, p0_mode=0,p1_mode=0,network_params=[], pred=False):
    param_names = ['tau','alpha']   
    with_switch,num_runs,frac,gamma,delta,n_nodes = network_params
    
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
    timespace = np.arange(len(data))
    
    
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
    '''
    l0 = ax[i].plot(sim_part[:switchpoint], 
                    color='gray', alpha=0.05) 
    '''
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
    print(num_runs)
    if (num_runs[0]>0):
        if pred:
            data_appr = data.iloc[:fin_size]
        else:
            data_appr = data
        for j in range(50):
            
        #num_runs=1
            q = calibr_funcs.simulation_func(1, tau=[p0_mode], 
                                    alpha=[p1_mode], 
                                    modeling_duration=[data_appr.shape[0]], 

                                     with_switch=with_switch,
                                    num_runs=num_runs, 
                                    frac=frac, 
                                    size=[data_appr.shape[0]])
            
            all_q.append(q)
            r2_part = r2_score(data_appr, q)
            all_r.append(r2_part)
            
            #l1=ax[i].plot(q,color='tab:green', lw=2, ls='-')
        
        ax[i].fill_between(x=np.arange(switchpoint,
                                       fin_size),
                   y1 = np.array(all_q).min(axis=0)[switchpoint:],
                   y2 = np.array(all_q).max(axis=0)[switchpoint:],
                   color='tab:green', alpha=.5, zorder=99,
                   #label='Simulations with selected params'
                          )

        best_idx = np.argmax(all_r)        
        ax[i].plot(np.arange(switchpoint,
                                       fin_size),
                   all_q[best_idx][switchpoint:],
                     color='white', lw=4, ls='-',
                     zorder=980)
        l1=ax[i].plot(np.arange(switchpoint,
                                       fin_size),
                      all_q[best_idx][switchpoint:],
                     color='ForestGreen', lw=2, ls='-',
                     label=r'Best simulation ($R^2$' +\
                      f' = {all_r[best_idx]:.3f})',
                     zorder=990)
        
    elif num_runs[0]==0:
        model = AESurrogateModel(10**5)
        q = model.simulate(p1_mode,p0_mode)[:data.shape[0]]
        r2_part = r2_score(data_part, q)
            
        ax[i].plot(q,
                     color='white', lw=4, ls='-',
                     zorder=980)
        l1=ax[i].plot(q,
                     color='ForestGreen', lw=2, ls='-',
                     label=r'Best simulation ($R^2$' +\
                      f' = {r2_part:.3f})',
                     zorder=990)
    next_c='blue'

    # real data
    '''
    ax[i].plot(data_part, "", ls='-', lw=4, 
              color='white')
    '''
    if pred:
        l2=ax[i].scatter(np.arange(switchpoint), 
                         data[:switchpoint], 
                      color='LimeGreen', 
                         edgecolors='k',
                         s=50, zorder=1000,
                        label='Known data',
                        )

        l3=ax[i].scatter(np.arange(switchpoint, 
                                   data.shape[0]),
                      data[switchpoint:], 
                      color='Grey', s=50,
                   label='Unknown data',
                     edgecolors='k',
                        alpha=.5, zorder=1000)
        
       
        ax[i].axvline(switchpoint, ls='--', color='k',
                      lw=1#, label='Forecast starts'
                     )       
        
    
    else:
        l3=ax[i].scatter(np.arange(switchpoint, 
                                   data.shape[0]),
                      data_part[switchpoint:], 
                      color='white', s=20,
                        alpha=1, zorder=1000)
        l3=ax[i].scatter(np.arange(switchpoint, 
                                   data.shape[0]),
                      data_part[switchpoint:], 
                      color='OrangeRed', s=10,
                   label='Observed incidence',
                     
                        alpha=1, zorder=1000)
    fontsize = 14
    #ax[i].set_title("Time series comparison")
    ax[i].set_xlabel('Time, days', fontsize=1.2*fontsize)
    ax[i].set_ylabel('Incidence, cases', fontsize=1.2*fontsize)
    '''
    ax[i].set_xticks(np.arange(0,100,20), 
                     np.arange(0,100,20),
                     fontsize=flabel)
    ax[i].set_yticks(np.arange(0,
                               int(data['incidence'].max()*1.2),
                               1000), 
                     np.arange(0,
                               int(data['incidence'].max()*1.2),
                               1000),
                     fontsize=flabel)
    
    ax[i].set_ylim(0, data['incidence'].max()*1.2)
    #ax[i].set_xlim(-5, data.shape[0])#np.where(data==0)[0][0]*1.1)
    '''
    
    ax[i].tick_params(axis='both', which='major', labelsize=fontsize)
    if pred:
        ymax = 3300
        ax[i].set_ylim(0, ymax)
    
    ax[i].grid()
    #ax[i].set_title(title)
    if pred:
        flabel_p = 10
    else:
        flabel_p = 10
        
    ax[i].legend(fontsize=flabel_p).set_zorder(9999);
    
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
    
    pols = [Polygon(cs.allsegs[0][i]) for i in range(n_different_areas)]
    centroid = MultiPolygon(pols).centroid
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
               network_params, pred=False,
              ax_curves=[], ax_kde=[]):
    cmap = mpl.colormaps['viridis']
    hdi_list = [0.2,0.5,0.8,0.9]
    colors_l = cmap(np.linspace(0, 1, len(hdi_list)))
    fontsize = 14
    
    # for edgecolor to have alpha
    fc=to_rgba('RoyalBlue', 0.5)
    param_names = ['tau','alpha']  
    fancy_names = [r'$\beta_n$', r'$\alpha$']
    
    if not ax_curves:
        fig = plt.figure(figsize=(12,5))
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
    else:
        ax_curves = ax_curves[0]
        ax_up,ax_scatter,ax_right = ax_kde
        
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
                          #fontsize=flabel
                         )
    ax_scatter.set_yticks(np.arange(min_y,1,0.1),
                         np.arange(min_y,1,0.1).round(1),
                          #fontsize=flabel
                         )


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
    
    ax_scatter.set_xlabel(fancy_names[0], #fontsize=flabel
                         )
    ax_scatter.set_ylabel(fancy_names[1], #fontsize=flabel
                         )
                          
    ax_scatter.set_xlim(0, 1)
    ax_scatter.set_ylim(0.5, 1)
    ticks_x = [0.1, 0.3, 0.5, 0.7, 0.9]
    ticks_y = [0.5, 0.6, 0.7, 0.8, 0.9]
    ax_scatter.set_xticks(ticks_x, list(map(str, ticks_x)))
    ax_scatter.set_yticks(ticks_y, list(map(str, ticks_y)))
    ax_scatter.tick_params(axis='both', which='major', labelsize=fontsize)
    ax_scatter.set_xlabel(fancy_names[0], fontsize=1.2*fontsize)
    ax_scatter.set_ylabel(fancy_names[1], fontsize=1.2*fontsize)
    
    legend_elements= []
    for c, val in zip(colors_l[::-1], hdi_list):
        legend_elements.append(mpatches.Patch(color=c,
                                              label=f'HDR {int(val*100)}%',
                                             alpha=.9)
                              )
    
    if pred:
        flabel_p = 12
    else:
        flabel_p = 10
    ax_scatter.legend(handles=[ls1,ls2,ls3,
                               *legend_elements],
                     fontsize=10)
    ax_scatter.grid()
    '''
    ax_scatter.plot([p0_mode,p0_mode], [p1_mode,1],
                    color='tab:green')
    ax_scatter.plot([p0_mode,1], [p1_mode,p1_mode],
                   color='tab:green')
    '''
    ax_up.axvline(p0_mode, ls='-', color='white', lw=3)
    ax_up.axvline(p0_mode, ls='--', color='tab:green')
    ax_up.axvline(true_tau, ls='-', color='white', lw=3)
    ax_up.axvline(true_tau, ls='--', color='OrangeRed')

    ax_right.axhline(p1_mode, ls='-', color='white', lw=3)
    ax_right.axhline(p1_mode, ls='--', color='tab:green')
    ax_right.axhline(true_alpha, ls='-', color='white', lw=3)
    ax_right.axhline(true_alpha, ls='--', color='OrangeRed')

    #ax_scatter.set_ylim(min_y*.9,1)
    #ax_scatter.set_xlim(min_x*.9,1)
    
    #ax_up.legend()
    
    plots(idata, observed_data, '',  
          ax=ax_curves, p0_mode=p0_mode,p1_mode=p1_mode,
          network_params=network_params,
         pred=pred)

    plt.tight_layout()