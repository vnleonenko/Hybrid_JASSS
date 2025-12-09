import numpy as np
import pandas as pd
import pymc as pm
import time
from source.autoencoder import AESurrogateModel


def surr_sim(rng, alpha, beta, modeling_duration, 
             n_nodes, top, koeff, shift,
             size=None):
    modeling_duration = modeling_duration[0]
    n_nodes =n_nodes[0]
    top =top[0]
    koeff = koeff[0]
    
    if top:
        top_str = 'ba'
    else:
        top_str = 'sw'
    model = AESurrogateModel(n_nodes, top_str)
    #alpha, beta
    q = model.simulate(alpha,beta)
    
    q[q<0] = 0
    week_data = pd.Series(q).groupby(pd.Series(q).index // 7).sum().values
    
    diff_w = week_data.shape[0] - modeling_duration
    # adding zeroes, if AE gives less datapoints than the original data
    if diff_w < 0:
        week_data = [*week_data,*[0]*abs(diff_w)]
    else:
        week_data = week_data[abs(diff_w):]#week_data[:modeling_duration]
        
    # shifting left or right horizontally    
    week_data= pd.Series(week_data).shift(shift).fillna(0).values
    return [i*koeff for i in np.squeeze(week_data)]


def calibr(draws=200, chains = 4, epsilon=500, 
           shift=[0], incidence=[], koeff = [1]):
    progressbar = True

    gamma = [0.3]
    delta=[0.2]
    n_nodes=[100000]
    top = [False] #is BA?
    #koeff = [int(pop/ n_nodes)]
    modeling_duration = [incidence.shape[0]]

    new_observed = incidence

    with pm.Model() as pm_modelp:
        real_data = pm.Data("incidence", new_observed)
        alpha = pm.Uniform(name="alpha", lower=10/n_nodes[0],
                           upper=1.)
        beta = pm.Uniform(name="tau", lower=0., upper=1.)

        # вынесем, тк для прогноза нужно будет задавать размер

        sim = pm.Simulator("sim", surr_sim, 
                            params = (alpha, beta, 
                                      modeling_duration, 
                                      n_nodes, top, 
                                      koeff,shift
                                     ),   

                            epsilon=epsilon, 
                            #ndims_params=[1,1,1,1,1,1],
                            observed=new_observed
                          )

        # maybe parameter start == mean of HM ?
        #step=pm.DEMetropolisZ()

        start_time = time.time()
        idatap = pm.sample_smc(#tune=tune, 
                              draws=draws, 
            chains=chains,
                              #step=step, 
            return_inferencedata=True,
                              progressbar=progressbar)
        end_time = time.time()
        idatap.extend(pm.sample_posterior_predictive(idatap, 
                                                     progressbar=progressbar))

    # It took  32616.331862449646 sec
    print('It took ', end_time-start_time, 'sec')

    idata = idatap.copy()

    idata.sample_stats = idata.sample_stats.drop('beta')
    idata.sample_stats = idata.sample_stats.drop('log_marginal_likelihood')
    idata.sample_stats = idata.sample_stats.drop('accept_rate')
    idata.to_netcdf(f"surr_on_real/sw_surr_{draws}_{chains}_eps{epsilon}_g{gamma[0]}_d{delta[0]}.nc")
    
    return idata