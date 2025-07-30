import pandas as pd
import optuna
from sklearn.metrics import r2_score
import numpy as np
from scipy.optimize import dual_annealing

from TotalBRModel import TotalBRModel

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, 
                        message=r".*ndims_params is deprecated.*")
warnings.filterwarnings(action='ignore', category=UserWarning, 
                        message=r".*fusion failed.*")
import pymc as pm

import logging
logger = logging.getLogger("pymc")
logger.setLevel(logging.WARNING)

optuna.logging.set_verbosity(optuna.logging.ERROR)

class Calibration:

    def __init__(        
        self,
        init_infectious: list[int]|int,
        model: TotalBRModel,
        data: pd.DataFrame,
        ) -> None:
        """
        Calibration class

        TODO

        :param init_infectious: Number of initial infected people  
        :param model: Model for calibration  
        :param data: Observed data for calibrating process  
        """
        self.rho = 4600000 #data['population_age_0-14'].iloc[-1] + data['population_age_15+'].iloc[-1]
        self.init_infectious = init_infectious
        self.model = model
        self.data = data
    


    def optuna_calibration(self, n_trials=1000):
        """
        TODO

        """

        data, alpha_len, beta_len = self.model.params(self.data)

        def model(trial):

            alpha = [trial.suggest_float(f'alpha_{i}', 0, 1) for i in range(alpha_len)]
            beta = [trial.suggest_float(f'beta_{i}', 0, 1) for i in range(beta_len)]


            self.model.simulate(
                alpha=alpha, 
                beta=beta, 
                initial_infectious=self.init_infectious, 
                rho=round(self.rho/10), 
                modeling_duration=int(len(data)/alpha_len)
            )

            return r2_score(data, self.model.newly_infected)


        study = optuna.create_study(direction="maximize")
        study.optimize(model, n_trials=n_trials)

        alpha = [study.best_params[f'alpha_{i}'] for i in range(alpha_len)]
        beta = [study.best_params[f'beta_{i}'] for i in range(beta_len)]

        return alpha, beta, round(self.rho/10)



    def annealing_calibration(self):
        """
        TODO

        """
        
        data, alpha_len, beta_len = self.model.params(self.data)

        lw_alpha = [0] * alpha_len  # нижние границы для alpha
        up_alpha = [25] * alpha_len  # верхние границы для alpha

        lw_beta = [0] * beta_len  # нижние границы для beta
        up_beta = [4] * beta_len  # верхние границы для beta

        lw = lw_alpha + lw_beta
        up = up_alpha + up_beta

        def model(x):
 
            alpha = x[:alpha_len]
            beta = x[alpha_len:]

            self.model.simulate(
                alpha=alpha, 
                beta=beta, 
                initial_infectious=self.init_infectious, 
                rho=round(self.rho/10), 
                modeling_duration=int(len(data)/alpha_len)
            )

            return -r2_score(data, self.model.newly_infected)

        ret = dual_annealing(model, bounds=list(zip(lw, up)))

        alpha = ret.x[:alpha_len]
        beta = ret.x[alpha_len:]

        return alpha, beta, round(self.rho/10)
    
    
    def mcmc_calibration(self, verbose=False, with_rho=False, with_initi=False,
                         tune=2500, draws=500, chains=4, beta_max=3,
                        rho=0.25*1163645):
        '''
        Parameters:
            - verbose -- show pymc progressbar
            - with_rho -- tune population size
            - with_initi -- tune initial infected
            - tune -- number of mcmc warmup samples
            - draws -- number of mcmc draws
            - chains -- number of chains
        '''
        
        init_inf = self.init_infectious
        self.model.rho = rho
        rho_fract = 1.0
        data, alpha_len, beta_len = self.model.params(self.data)
        progressbar = verbose
        data = np.nan_to_num(data).tolist() 

        def simulation_func(rng, alpha, beta, rho, init_inf, 
                            modeling_duration, size=None):
            
            self.model.simulate(
                alpha=np.array(alpha).flatten(), 
                beta=np.array(beta).flatten(), 
                initial_infectious=np.array(init_inf).flatten(), 
                rho=np.array(rho).flatten(), 
                modeling_duration=np.array(modeling_duration
                                          ).flatten()[0]
            )      
            return self.model.newly_infected
        
        with pm.Model() as pm_model:
            alpha = pm.Uniform(name="a", lower=0, upper=1., 
                               shape=alpha_len)
            beta = pm.Uniform(name="b", lower=0, upper=beta_max,
                              shape=beta_len)
            if with_rho:
                rho_fract = pm.Uniform(name="rho", lower=0.001, 
                                       upper=1.)
            if with_initi:
                init_inf = pm.Uniform(name="init_inf", lower=1, 
                                      upper=1000, shape=alpha_len)

            # вынесем, тк для прогноза нужно будет задавать размер
            modeling_duration = [int(len(data)/alpha_len)] 
            
            sim = pm.Simulator("sim", simulation_func, alpha, beta,
                                rho*rho_fract, init_inf, modeling_duration,
                                epsilon=10000, 
                               ndims_params=[alpha_len,beta_len,1,
                                             alpha_len,1],
                               observed=data)

            step=pm.DEMetropolisZ()
            idata = pm.sample(tune=tune, draws=draws, 
                              cores=6, chains=chains,
                              step=step, progressbar=progressbar)
            idata.extend(pm.sample_posterior_predictive(idata,
                                                    progressbar=progressbar))

        return idata, data, simulation_func, pm_model
