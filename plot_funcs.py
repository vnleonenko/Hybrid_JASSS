import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error as rmse
from tqdm.notebook import tqdm, trange
#from model_complex import Calibration, EpidData, FactoryBRModel
from timeit import default_timer as timer
import pymc as pm
import arviz as az


def count_4_metrics(idata, max_data_len, 
                    r2_threshold=0.65, with_pred=False,
                    full_data=np.array([]), 
                    peak_height=False, peak_time=False):
    if with_pred:
        sim_value = idata.predictions.sim_forecast
        real_data = full_data
    else:
        sim_value = idata.posterior_predictive.sim
        real_data = idata.observed_data.sim.values
    
    model_time = real_data.shape[0]
    x = np.arange(model_time)
    
    # shape = (n_chain*n_draws, len)
    sim_all = sim_value.values.reshape(-1, model_time)
    # добавляем до макс.длины
    to_add = max_data_len - real_data.shape[0]
    sim_fin = np.concatenate([np.zeros((sim_all.shape[0], to_add)), 
                                  sim_all], axis=1)

    # реальные данные; добавляем до макс.длины
    real_data_max_len = np.concatenate([np.zeros(to_add), real_data])
    # повторяем n_chain*n_draws раз
    real_data_fin = np.array([real_data_max_len] * sim_all.shape[0])

    # считаем r2 для всех линий
    r2_all = r2_score(real_data_fin.T, sim_fin.T, 
                      multioutput='raw_values')
    # % линий с r2 выше порога
    perc_higher_thresh = r2_all[r2_all > r2_threshold
                               ].shape[0] / r2_all.shape[0]
    # лучший r2
    r2_max = r2_all.max()

    rmse_peak_height, rmse_peak_time = [np.inf], [np.inf]
    if peak_height:
        rmse_peak_height = rmse(np.array([real_data] * sim_all.shape[0]).max(1), 
                                 sim_all.max(1),
                                 multioutput='raw_values')
    
    if peak_time:
        rmse_peak_time = rmse(np.array([real_data] * sim_all.shape[0]).argmax(1), 
                                 sim_all.argmax(1),
                                 multioutput='raw_values')
    
    return perc_higher_thresh, r2_max, rmse_peak_height, rmse_peak_time

    

def plot_metric_zeros(ax, idata, title, 
                      max_data_len, r2_threshold=0.65, 
                      peak_height=False, peak_time=False):

    sim_value = idata.posterior_predictive.sim
    real_data = idata.observed_data.sim.values
    
    model_time = real_data.shape[0]
    x = np.arange(model_time)

    perc_higher_thresh, r2_max, \
        rmse_peak_height, rmse_peak_time = count_4_metrics(idata, max_data_len, 
                                                            r2_threshold, with_pred=False,
                                                            peak_height=peak_height, 
                                                            peak_time=peak_time)
    
    # shape = (n_chain*n_draws, len)
    sim_all = sim_value.values.reshape(-1, model_time)
    
    # posterior predictive lines
    ax.plot(sim_all.T, color='blue', alpha=0.005) 
    # ppc median
    ax.plot(sim_value.quantile(q=.5, dim=['chain','draw']),
             color='lightgray', lw=2, label='median')
    # real data
    ax.plot(real_data, "o", ls='', color='brown', 
               markeredgecolor='white', label='real data')

    ci = .95
    # posterior predictive ci
    ax.fill_between(x=x,
                     y1=sim_value.quantile(q=round(0.5 - ci/2, 3), 
                                         dim=['chain','draw']), 
                     y2=sim_value.quantile(q=round(0.5 + ci/2, 3), 
                                         dim=['chain','draw']),
                     color='skyblue', alpha=.5, label=f'CI {ci*100:.0f}%')

    p_title = f''
    if peak_height:
        p_title = p_title + f'RMSE peak height: {rmse_peak_height[0]:.1f}\n' 
    if peak_time:
        p_title += f'RMSE peak time: {rmse_peak_time[0]:.2f}'     
    
    ax.set_title(f'{title}.\n' + 
                 f'"Good" lines: {perc_higher_thresh*100:.1f}%. ' + 
                 f'Max r2: {r2_max:.2f}\n' + p_title)
    ax.grid()
    
    if peak_height & peak_time:
        return perc_higher_thresh, r2_max, rmse_peak_height, rmse_peak_time
    else:
        return perc_higher_thresh, r2_max
    


def pred_plot(year, data, data_mtest, 
              idata, with_ylim=False):
    
    alpha_len = idata.posterior.a.shape[-1]
    train_size = len(data)//alpha_len
    test_size = len(data_mtest)//alpha_len
    
    train_steps = np.arange(train_size)
    test_steps = np.arange(train_size+test_size)
    
    # 'sim_forecast' chain: 4 draw: 500 sim_forecast_dim_2: ...
    values = idata.predictions["sim_forecast"]
    sim_value = idata.posterior_predictive.sim
     
    
    fig, ax = plt.subplots(1,alpha_len, sharex=True, 
                           sharey=True, figsize=(9,5))
    # чтобы работало и при одном ax
    ax = np.array([ax]).flatten()
    
    # для каждой возрастной группы
    for i in range(alpha_len):
        model_time = [train_size]
        data_part = data[i*model_time[0]:i*model_time[0]+model_time[0]]  
        
        # _____calibration plot part
        sim_part = sim_value[:,
                             :,
                             i*model_time[0]:i*model_time[0]+model_time[0]]
        # ci
        ci = .95
        ax[i].fill_between(x=train_steps,
                         y1=sim_part.quantile(q=round(0.5 - ci/2, 3), dim=['chain', 'draw']), 
                         y2=sim_part.quantile(q=round(0.5 + ci/2, 3), dim=['chain', 'draw']),
                         color='lightgray', alpha=.5)
        # median calibration line
        ax[i].plot(train_steps, sim_part.quantile(q=.5, dim=['chain', 'draw']), color='lightgray', lw=2)
        # all lines
        calib_lines = np.array(sim_part).reshape(-1,train_size).T
        ax[i].plot(train_steps, calib_lines, color='gray', alpha=0.005) 
        
        # _____prediction plot part
        
        model_time = [train_size+test_size]
        #data_part = data[i*model_time[0]:i*model_time[0]+model_time[0]] 
        values_part = values[:,
                             :,
                             i*model_time[0]:i*model_time[0]+model_time[0]]
        preds_lower = values_part.quantile(q=round(0.5 - ci/2, 3), 
                                           dim=['chain', 'draw'])
        pred_higher = values_part.quantile(q=round(0.5 + ci/2, 3), 
                                           dim=['chain', 'draw'])
        # ci
        ax[i].fill_between(x=test_steps[train_size:],
                         y1=preds_lower[train_size:], 
                         y2=pred_higher[train_size:],
                         color='skyblue', alpha=.5, label=f'CI {ci*100:.0f}%')
        # predictive lines
        preds = np.array(values_part).reshape(-1,train_size+test_size).T
        ax[i].plot(test_steps[train_size:], 
                 preds[train_size:], 
                 color='blue', alpha=0.005);    
        # median predictive line
        median_pred = np.array(values_part.quantile(q=.5, 
                                                    dim=['chain', 'draw']))
        ax[i].plot(test_steps[train_size:], 
                 median_pred[train_size:], 
                 color='gray', lw=2, label=f'median')

        ax[i].axvline(train_size-0.5, ls="--", color='gray', lw=1)


        # data
        ax[i].plot(train_steps, 
                   data[i*train_size:i*train_size+train_size], 
                   "o", ls='', color='brown', 
                   markeredgecolor='white', label='train data')
        ax[i].plot(test_steps[train_size:], 
                   data_mtest[i*test_size:i*test_size+test_size],
                   "o", ls='', color='forestgreen', 
                   markeredgecolor='white', label='test data')

        if with_ylim:
            high_lim = np.max([median_pred.max()*1.25,
                              np.max(data[i*train_size:i*train_size+train_size
                                  ])*1.25,
                              np.max(data_mtest[i*test_size:i*test_size+test_size
                                        ])*1.25])
            ax[i].set_ylim(None, high_lim)
            
        ax[i].set_xlabel('День')
        ax[i].set_ylabel('Новые заболевшие (чел.)')
        ax[i].grid()
        ax[i].set_title(f'{year}-{year+1} г.г.')
        ax[i].legend();

    
def pred_plot_ax(ax, title, data_train, data_test, 
                 idata, with_ylim=False, time=0, 
                 with_4_metrics=False, max_data_len=50):
    '''
    График прогноза на одном ax
    '''
    
    alpha_len = idata.posterior.a.shape[-1]
    train_size = len(data_train)//alpha_len
    test_size = len(data_test)//alpha_len
    
    train_steps = np.arange(train_size)
    test_steps = np.arange(train_size+test_size)
    
    # 'sim_forecast' chain: 4, draw: 500, sim_forecast_dim_2: ...
    values = idata.predictions["sim_forecast"]
    sim_value = idata.posterior_predictive.sim
    
    model_time = [train_size]
    # _____calibration plot part

    # ci
    ci = .95
    ax.fill_between(x=train_steps,
                    y1=sim_value.quantile(q=round(0.5 - ci/2, 3), 
                                          dim=['chain', 'draw']), 
                    y2=sim_value.quantile(q=round(0.5 + ci/2, 3), 
                                          dim=['chain', 'draw']),
                    color='lightgray', alpha=.5)
    # median calibration line
    ax.plot(train_steps, sim_value.quantile(q=.5, dim=['chain', 'draw']), 
            color='lightgray', lw=2)
    # all lines
    calib_lines = np.array(sim_value).reshape(-1,train_size).T
    ax.plot(train_steps, calib_lines, color='gray', alpha=0.005) 

    # _____prediction plot part

    model_time = [train_size+test_size]
    preds_lower = values.quantile(q=round(0.5 - ci/2, 3), 
                                  dim=['chain', 'draw'])
    pred_higher = values.quantile(q=round(0.5 + ci/2, 3), 
                                  dim=['chain', 'draw'])
    # ci
    ax.fill_between(x=test_steps[train_size:],
                     y1=preds_lower[train_size:], 
                     y2=pred_higher[train_size:],
                     color='skyblue', alpha=.5, label=f'CI {ci*100:.0f}%')
    # predictive lines
    preds = np.array(values).reshape(-1,train_size+test_size).T
    ax.plot(test_steps[train_size:], 
             preds[train_size:], 
             color='blue', alpha=0.005);    
    # median predictive line
    median_pred = np.array(values.quantile(q=.5, dim=['chain', 'draw']))
    ax.plot(test_steps[train_size:], 
             median_pred[train_size:], 
             color='gray', lw=2, label=f'median')

    ax.axvline(train_size-0.5, ls="--", color='gray', lw=1)

    # data
    ax.plot(train_steps, 
               data_train, "o", ls='', color='brown', 
               markeredgecolor='white', label='train data')
    ax.plot(test_steps[train_size:], data_test,
               "o", ls='', color='forestgreen', 
               markeredgecolor='white', label='test data')

    if with_ylim:
        high_lim = np.max([median_pred.max()*1.25,
                          np.max(data_train)*1.25,
                          np.max(data_test)*1.25])
        ax.set_ylim(None, high_lim)
 
    if with_4_metrics:
        perc_higher_thresh, r2_max, \
        rmse_peak_height, rmse_peak_time = count_4_metrics(idata, max_data_len, 
                                        r2_threshold=0.65, 
                                        with_pred=True,
                                        full_data=np.concatenate([data_train,data_test]),
                                        peak_height=True, peak_time=True)
        ax.set_title(f'{title}.\n' + 
                     f'"Good" lines: {perc_higher_thresh*100:.1f}%. ' + 
                     f'Max r2: {r2_max:.2f}\n' +
                     f'RMSE peak height: {rmse_peak_height[0]:.1f}\n' +
                     f'RMSE peak time: {rmse_peak_time[0]:.2f}')
    else:
        ax.set_title(f'{title}')
        
    #ax.set_xlabel('День')
    #ax.set_ylabel('Новые заболевшие (чел.)')
    ax.grid();
    
    if with_4_metrics:
        return  perc_higher_thresh, r2_max, rmse_peak_height, rmse_peak_time
    

# _____________________ FOR CALIBRATION    

def plots(idata, data, title, simulation_func, with_trace=True, 
          show_values=True, for_pred=False, return_r2=False):

    sim_value = idata.posterior_predictive.sim
    posterior = idata.posterior.stack(samples=("draw", "chain"))

    model_time = [data.shape[0]]
    alpha_len = 1
    x = np.arange(model_time[0])
    
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
        r2_part = r2_score(data_part, 
                             sim_part.quantile(q=.5, 
                                               dim=['chain', 'draw']
                                              ))
        
        # posterior predictive lines
        ax[i].plot(np.array(sim_part).reshape(-1,len(data_part)).T, 
                 color='tab:blue', alpha=0.1) 
        # ppc median
        ax[i].plot(sim_part.quantile(q=.5, dim=['chain', 'draw']),
                 color='lightgray', lw=2,
                 label=f'median (R^2 = {r2_part:.3f})')

        # real data
        ax[i].plot(data_part, "o", ls='', color='brown', 
                   markeredgecolor='white', label='real data')

        ci = .95
        # posterior predictive ci
        ax[i].fill_between(x=x,
                         y1=sim_part.quantile(q=round(0.5 - ci/2, 3), 
                                              dim=['chain', 'draw']), 
                         y2=sim_part.quantile(q=round(0.5 + ci/2, 3), 
                                              dim=['chain', 'draw']),
                         color='skyblue', alpha=.5, label=f'CI {ci*100:.0f}%')

        ax[i].set_xlabel('Day')
        ax[i].set_ylabel('Newly infected')
        
        ax[i].set_xlim(-5, np.where(data==0)[0][0]*1.1)
        ax[i].grid()
        ax[i].set_title(title)
        ax[i].legend();
    
    if return_r2:
        return r2_part
    
    if with_trace:
        pm.plot_trace(idata);
        az.plot_posterior(idata);
        return az.summary(idata)

    
def fig_for_subplot(ax, idata, data, year, simulation_func,
                    cal_model, with_rho=False, 
                    with_initi=False, time=0):
    col = 0
    if year >= 2015:
        col = 2
    
    sim_value = idata.posterior_predictive.sim
    
    alpha_len = idata.posterior.a.shape[-1]    
    model_time = [len(data)//alpha_len]
    x = np.arange(model_time[0])

    
    # для каждой возрастной группы
    for i in range(alpha_len):
        
        data_part = data[i*model_time[0]:i*model_time[0]+model_time[0]]
        sim_part = sim_value[:,
                             :,
                             i*model_time[0]:i*model_time[0]+model_time[0]]
        r2_part = r2_score(data_part[len(data_part)//2:], 
                             sim_part.quantile(q=.5, 
                                               dim=['chain', 'draw']
                                              )[sim_part.shape[-1]//2:])
        ci = .95
        # posterior predictive ci
        ax[col+i].fill_between(x=x,
             y1=sim_part.quantile(q=round(0.5 - ci/2, 3), 
                                  dim=['chain', 'draw']), 
             y2=sim_part.quantile(q=round(0.5 + ci/2, 3), 
                                  dim=['chain', 'draw']),
             color='skyblue', alpha=.5, label=f'CI {ci*100:.0f}%'
                              )
        
        # calibration lines
        ax[col+i].plot(np.array(sim_part).reshape(-1,len(data_part)).T, 
                 color='blue', alpha=0.005) 
        #  median
        ax[col+i].plot(sim_part.quantile(q=.5, dim=['chain', 'draw']),
                 color='lightgray', lw=2,
                 label=f'median (R^2 = {r2_part:.3f})')

        # real data
        ax[col+i].plot(data_part, "o", ls='', color='brown', 
                   markeredgecolor='white', label='real data')
        
        ax[col+i].text(x=0.01,y=0.95, s=f'R\N{SUPERSCRIPT TWO}(median) = {r2_part:.3f}', 
                ha='left', va='top', transform=ax[col+i].transAxes)
        ax[col+i].text(x=0.01,y=0.85, s=f'Time = {time:.3f}', 
                ha='left', va='top', transform=ax[col+i].transAxes)
        ax[col+i].set_title(f'{year}-{year+1} г.г., age: {i}')
        ax[col+i].grid()



def grid_plots(with_rho=True, with_initi=True, weeks_bfr = 3,
               years = [2010,2011,2012,2013,2014,2015,
                        2016,2017,2018,2019],
              tune=2500, draws=500, chains=4):
    fig, ax = plt.subplots(5,2, sharex=True, sharey=True, figsize=(9,11))
    init_infect = [100]

    for idx, year in enumerate(tqdm(years)):
        data_orig = EpidData('spb', './', f'7-01-{year}', f'6-20-{year+1}')
        peak_idx = data_orig.reset_index(drop=True)['total'].nlargest(1).index[0]
        data_train, data_test = data_orig.iloc[:peak_idx-weeks_bfr+1], \
                                    data_orig.iloc[peak_idx-weeks_bfr+1:]

        model = TotalBRModel()

        d = Calibration(init_infect, model, data_train)
        start = timer()
        idata, data, simulation_func, \
            pm_model,rho = d.mcmc_calibration(with_rho=with_rho,
                                              with_initi=with_initi,
                                              tune=tune, draws=draws, chains=chains)
        pp = predictions(simulation_func, data, idata, duration=data_orig.shape[0], 
                         rho=rho, alpha_len=d.model.alpha_len,
                         with_rho=with_rho, with_initi=with_initi)  
        end = timer()
        
        col = 0
        if year >= 2015:
            col = 1
            idx -= 5

        pred_plot_ax(ax[idx, col], year, data_train, data_test,
                     pp, idata, end-start, with_ylim=True)

    fig.supxlabel('День')   
    fig.supylabel('Новые заболевшие (чел.)')

    handles, labels = ax[-1,0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, bbox_to_anchor=(1.1, 1));
    
    
def predictions(simulation_func, idata, model, duration, 
                with_rho=True, with_initi=True):
    
    alpha_len = model.alpha_len
    beta_len = model.beta_len
    
    with pm.Model() as forecast_m:
        alpha = pm.Uniform(name="a", shape=alpha_len)
        beta = pm.Uniform(name="b", shape=beta_len)
        rho = model.rho
        init_inf = [100]
        
        pm.Simulator("sim_forecast", simulation_func, alpha, beta,
                 rho, init_inf, [duration], epsilon=10000,
                     ndims_params=[alpha_len,beta_len,
                                   1,alpha_len,1])
        idata.extend(pm.sample_posterior_predictive(idata, 
                                            var_names=["sim_forecast"],
                                            predictions=True,
                                            progressbar=False))
    return idata