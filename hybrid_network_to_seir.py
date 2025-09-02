import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import os
import glob
import EoN
import networkx as nx

from tqdm import tqdm
from scipy.stats import uniform, norm, multivariate_normal
from scipy.integrate import odeint

import warnings
import shutil
import time
warnings.filterwarnings('ignore')
#from agent_based_model import load_data, preprocess_data
from main_pool import Main

from networks.model_output import SEIRModelOutput, SEIRParams
from networks.SEIR_network import SEIRNetworkModel
import plot_hyb
import predict_Beta_I
import seir_discrete
import choice_start_day


class NetworkSEIR_tuned:
    """
    Network-based SEIR model parameter estimation class
    """
    def __init__(self, observed_data, network_params=None, 
                 fixed_sigma=0.1, fixed_gamma=0.08):
        """Initialization with fixed alpha and gamma"""
        self.observed_data = observed_data
        
        # fixed parameters
        self.fixed_sigma = fixed_sigma 
        self.fixed_gamma = fixed_gamma 
        
        # default network parameters
        if network_params is None:
            network_params = {
                'n_nodes': 10000,
                'network_type': 'ba',
                #'network_params': {'m': 3}
            }
        self.network_params = network_params
        
        # store history matching results
        self.hm_results = None
        
        print(f"Network SEIR - Fixed parameters: "+\
              f"sigma = {self.fixed_sigma}, gamma = {self.fixed_gamma}")
        print("Calibrating parameters")
    
    
    def generate_network(self):
        """Generate network based on specified parameters"""
        population = self.network_params['n_nodes']
        network_type = self.network_params['network_type']
        
        if network_type == 'ba':
            return nx.barabasi_albert_graph(population, 5, 
                                            seed=chosen_seed)
        elif network_type == 'sm':
            return nx.watts_strogatz_graph(population, 5, 0.1, 
                                           seed=chosen_seed)
        elif network_type == 'r':
            return nx.fast_gnp_random_graph(population, 5/population, 
                                            seed=chosen_seed)
        else:
            raise ValueError(f"Unknown network type: {network_type}")

            
    def switch_seir(self, sim_data, method='expanding'):
        pop = sim_data.iloc[0,:4].sum()
        
        '''
        switches = sim_data[sim_data['I'] > pop*self.frac]
        if switches.shape[0]:
            switch_day = switches.index[0]
        else:
            switch_day = 0
        '''    
        switch_day = sim_data.shape[0]-1
        y0 = sim_data.iloc[switch_day,:4].values.flatten()
        # FOR FULL OBSERVED DATA
        
        ts = np.arange(self.tmax-switch_day)
        if method=='expanding':
            beta = sim_data.iloc[:switch_day]['Beta'
                                    ].expanding(1).mean().values[-1]
        elif method=='last':
            beta = sim_data.iloc[:switch_day]['Beta'].values[-1]
        elif method=='rolling':
            beta = sim_data.iloc[:switch_day]['Beta'
                                    ].rolling(7).mean().values[-1]
            
        # Median from HM accepted parameters' trajectories!
        #median_b = pd.read_csv('median_beta.csv')
        #beta = median_b.iloc[switch_day:].values
        S,E,I,R = seir_discrete.seir_model(y0, ts, beta, 
                                           self.fixed_sigma, 
                                           self.fixed_gamma, stype='d', 
                                           beta_t=False).T
        
        fin = pd.DataFrame([S,E,I,R]).T
        fin.columns = ['S','E','I','R']
        
        fin['Beta'] = beta
        fin['day'] = ts+switch_day
        fin = pd.concat([sim_data.iloc[:switch_day], fin]
                       ).reset_index(drop=True)
        return fin
    
    
    def simulator_function(self, tau, alpha, with_switch=False, 
                           num_runs=1, frac=1, 
                           init_inf_frac=0.0001,
                           method='expanding'):
        """Run SEIR network simulation with given tau, rho and fixed alpha, gamma"""
        self.frac = frac
        #try:
        chosen_seed = np.random.RandomState(42)
        network_model = SEIRNetworkModel(self.network_params['n_nodes'],
                                         self.network_params['network_type'], 
                                         chosen_seed)

        self.tmax = len(self.observed_data) - 1
        # fraction of initially recovered
        init_rec_frac = 1 - alpha
        
        all_results = []
        for run in range(num_runs):
            res = network_model.simulate(beta=tau, 
                                         gamma=self.fixed_sigma, 
                                         delta=self.fixed_gamma, 
                                         init_inf_frac=init_inf_frac, 
                                         init_rec_frac=init_rec_frac,
                                         tmax=self.tmax,
                                         I_frac_switch = frac)

            seed_df = pd.DataFrame([res.S, res.E, res.I, res.R]).T
            seed_df.columns = ['S','E','I','R']
            seed_df['day'] = np.arange(seed_df.shape[0])
            # use "values", because "iloc" saves index info 
            # and messes with the calculation
            beta_calc = - seed_df.S.diff().values[1:] / (
                                    seed_df.S.values[:-1] * seed_df.I.values[:-1]
                                    )
            # the last Beta value cannot be calculated: no S_{t+1}
            seed_df['Beta'] = [*beta_calc, 0] 
            seed_df.fillna(0, inplace=True)

            if with_switch:
                if 'Beta' not in seed_df.columns:
                    print(seed_df.columns)
                seed_df = self.switch_seir(seed_df, method=method)
            
            # calculating incidence
            temp = seed_df[['E','S']].shift([0,1])
            seed_df['incidence'] = (temp['E_1'] - temp['E_0']) - \
                                (temp['S_0'] - temp['S_1'])
            seed_df['incidence'].fillna(0, inplace=True)
            all_results.append(seed_df)  
            
        if all_results:
            combined_results = pd.concat(all_results, 
                                         ignore_index=True)
            seed_df = combined_results.groupby('day').mean().reset_index()
            seed_df[['S','E','I','R',
                    'incidence']] = seed_df[['S','E','I','R',
                                             'incidence']].round()
            
        return seed_df
        '''    
        except Exception as e:
            print(f"Simulation error: {e}")
            return None
        '''
    
    
    def calculate_distance(self, sim_data):
        """Calculate distance between simulated and observed data"""
        if sim_data is None:
            return np.inf
        
        try:
            
            min_len = min(len(self.observed_data), len(sim_data))
            
            obs = self.observed_data['incidence'].values[:min_len]
            
            sim = sim_data['incidence'].values[:min_len]
            # mse
            distance = np.mean((obs - sim)**2)
            
            distance = np.mean(abs((obs - sim))) 
            return distance
        
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return np.inf
    
    
    def history_matching(self, prior_ranges, n_samples=100, 
                         epsilon=1000, adaptive=False, accept_ratio=0.2,
                         prepared=True, folder='../new_sw_100000/',
                        file='../sim_data/data_sw_100000.csv'):
        """
        History matching to find plausible parameter regions 
        for tau and rho only
        """
        print(f"Running Network SEIR history matching with {n_samples} samples...")
        
        p0, p1 = ['tau', 'alpha']
        
        if prepared:
            '''
            # берем файлы по уникальным параметрам
            u_files = glob.glob(f'{folder}*.csv')[::10]
            
            random.seed(42)
            u_files = random.sample(u_files,n_samples)
            results = []
            for file in tqdm(u_files, desc="Network History Matching"):
                # берем все сиды для одного набора параметров и усредняем
                seeds = glob.glob(f'{file[:-4]}*.csv')
                all_results = []
                for seed in seeds:
                    seed_df = pd.read_csv(seed, usecols=['E','S']
                                         ).reset_index()
                    
                    temp = seed_df[['E','S']].shift([0,1])
                    seed_df['incidence'] = (temp['E_1'] - temp['E_0']) - \
                                        (temp['S_0'] - temp['S_1'])
                    seed_df['incidence'].fillna(0, inplace=True)
                    all_results.append(seed_df[['index','incidence']])
                    
                combined_results = pd.concat(all_results, ignore_index=True)
                combined_results.columns=['day', 'incidence']
                sim_data = combined_results.groupby('day').mean().reset_index()  
                distance = self.calculate_distance(sim_data)
                
                # add trajectory to results dictionary
                trajectory = []
                if sim_data is not None:
                    trajectory = sim_data["incidence"].values.tolist()
                    
                # берем значения параметров    
                params = file.split('\\')[-1].split('_')
                sample_tau = float(params[1])
                sample_alpha = float(params[5])
                # store trajectory data
                result = [sample_tau, sample_alpha, distance, trajectory]
                
                results.append(result)
            '''
            
            d = pd.read_csv(file, index_col=0)#.iloc[::10]
            if 'init_rec_frac' in d.columns:
                d.rename(columns={'init_rec_frac':'alpha'}, inplace=True)
            d.drop(columns=['gamma','delta','init_inf_frac'], inplace=True)
            
            beta_arr = np.arange(0.1, 1, 0.01) #np.arange(0.04, 0.09, 0.01)
            alpha_arr = np.arange(0.2, 1, 0.01) #np.arange(0.005, 0.011, 0.001)
            
            i = 0
            results=[]
            for beta in beta_arr:
                beta = round(beta,2)
                for alpha in alpha_arr:
                    alpha = round(alpha,2)
                    #print(chosen.iloc[:,:2].values, beta, alpha)
                    
                    if d.shape[1]>200:
                        s = self.observed_data.shape[0]
                        chosen = d.iloc[i:i+1]#d[(d.beta>=beta)&(d.alpha>=alpha)]
                        incidence = np.array([chosen.iloc[:1,2:2+s].values[0],
                                     chosen.iloc[:1,2+s:2+2*s].values[0],
                                     chosen.iloc[:1,2+2*s:2+3*s].values[0]]).mean(0)
                        
                    else:
                        incidence = d.iloc[i*10:i*10+10,2:].mean()
                    i+=1
                    sim_data = pd.DataFrame(incidence, columns=['incidence'])
                    distance = self.calculate_distance(sim_data)

                    # add trajectory to results dictionary
                    trajectory = []
                    if sim_data is not None:
                        trajectory = sim_data["incidence"].values.tolist()

                    # берем значения параметров    
                    params = file.split('\\')[-1].split('_')
                    sample_tau = beta
                    sample_alpha = alpha
                    # store trajectory data
                    result = [sample_tau, sample_alpha, distance, trajectory]

                    results.append(result)
                    if len(results) > n_samples:
                        break
            
        else:
            samples = []
            for _ in range(n_samples):
                sample = {}
                for param, (min_val, max_val) in prior_ranges.items():
                    if param in [p0, p1]:
                        sample[param] = uniform.rvs(loc=min_val, 
                                                    scale=max_val-min_val)
                samples.append(sample)

            results = []
            for sample in tqdm(samples):
                sim_data = self.simulator_function(sample[p0], sample[p1])
                
                distance = self.calculate_distance(sim_data)
                
                trajectory = []
                if sim_data is not None:
                    trajectory = sim_data["incidence"]

                result = [sample[p0], sample[p1], distance, trajectory]    
                results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df.columns = [p0, p1,'distance','trajectory']
        
        if adaptive:
            n_accept = max(1, int(len(results_df) * accept_ratio))
            accepted = results_df.nsmallest(n_accept, "distance")
        else:
            accepted = results_df[results_df["distance"] < epsilon]
        
        print(f"Accepted {len(accepted)} parameter sets")
        self.hm_results = accepted
        return accepted
    
    
    def smc_abc(self, n_particles=50, n_populations=3, 
                accept_ratio=0.1, frac=0.01, num_runs=1, 
                with_switch=True, method='expanding'):
        """ABC Sequential Monte Carlo"""
        if self.hm_results is None or self.hm_results.empty:
            print("Warning: No history matching results available. Cannot run SMC ABC.")
            return pd.DataFrame()
        
        print(f"Running ABC-SMC with {n_populations} populations...")
        initial_epsilon = self.hm_results['distance'].quantile(0.9)
        final_epsilon = self.hm_results['distance'].quantile(0.1)
        epsilons = np.geomspace(initial_epsilon, final_epsilon, n_populations)
        
        param_names = self.hm_results.columns.drop(['distance',
                                                'trajectory']).values  
        p0 = param_names[0]
        p1 = param_names[1]
        
        param_bounds = {
            p0: (self.hm_results[p0].quantile(.1), 
                      self.hm_results[p0].quantile(.9)),
            p1: (self.hm_results[p1].quantile(.1), 
                     self.hm_results[p1].quantile(.9))
        }
        
        particles = []
        for _ in range(n_particles):
            hm_idx = np.random.randint(0, len(self.hm_results))
            hm_sample = self.hm_results.iloc[hm_idx]
            particle = {param: hm_sample[param] for param in [p0, p1]}
            particles.append(particle)
        weights = np.ones(n_particles) / n_particles
        
        
        for t in range(n_populations):
            epsilon = epsilons[t]
            print(f"SMC Population {t+1}/{n_populations}, epsilon = {epsilon:.2f}")
            distances = []
            trajectories = []
            
            for particle in tqdm(particles, desc=f"SMC Population {t+1}"):
                # ABM switch (1% of Infected)
                
                sim_data = self.simulator_function(particle[p0], particle[p1],
                                                   with_switch=with_switch, 
                                                   num_runs=num_runs,
                                                   frac=frac, method=method)
                
                #sim_data = pd.DataFrame(sim_data, columns=['H1N1'])
                distance = self.calculate_distance(sim_data[['I']])
                distances.append(distance)
                #print(distance, sim_data['I'][:5])
                if sim_data is not None:
                    trajectories.append(sim_data['I'].values)
                else:
                    trajectories.append(None)
                    
            new_weights = np.zeros(n_particles)
            
            for i, distance in enumerate(distances):
                if distance < epsilon:
                    new_weights[i] = weights[i]
                    
            if np.sum(new_weights) > 0:
                new_weights = new_weights / np.sum(new_weights)
            else:
                sorted_indices = np.argsort(distances)
                n_best = max(1, int(n_particles * 0.1))
                for i in range(n_best):
                    new_weights[sorted_indices[i]] = 1.0
                new_weights = new_weights / np.sum(new_weights)
            
                
            ESS = 1.0 / np.sum(new_weights**2)
            print(f"Effective sample size: {ESS:.1f}")
            
            print(particles)
            if ESS < n_particles / 2:
                indices = np.random.choice(n_particles, 
                                           size=n_particles, 
                                           p=new_weights)
                particles = [particles[i] for i in indices]
                trajectories = [trajectories[i] for i in indices]
                weights = np.ones(n_particles) / n_particles
            else:
                weights = new_weights
            
            print(n_particles, n_particles, new_weights)
            print(particles)
            
            #weights = new_weights
            if t < n_populations - 1:
                param_values = np.array([[p[p0], p[p1]] for p in particles])
                cov = np.cov(param_values.T) + np.eye(2) * 1e-6
                new_particles = []
                for particle in particles:
                    attempts = 0
                    while attempts < 50:
                        perturbation = multivariate_normal.rvs(mean=[0, 0], cov=cov)
                        new_particle = {
                            p0: particle[p0] + perturbation[0],
                            p1: particle[p1] + perturbation[1]
                        }
                        alpha_valid = param_bounds[p0][0] <= new_particle[p0] <= param_bounds[p0][1]
                        lmbd_valid = param_bounds[p1][0] <= new_particle[p1] <= param_bounds[p1][1]
                        if alpha_valid and lmbd_valid:
                            new_particles.append(new_particle)
                            break
                        attempts += 1
                    if attempts >= 50:
                        new_particles.append(particle)
                particles = new_particles
                
        final_results = []
        for i, particle in enumerate(particles):
            final_results.append({
                p0: particle[p0],
                p1: particle[p1],
                "weight": weights[i],
                "distance": distances[i],
                "trajectory": trajectories[i]
            })
        results_df = pd.DataFrame(final_results)
        print(f"ABC-SMC completed with {len(results_df)} particles")
        
        return results_df.sort_values('distance')
    
    
    
def generate_synthetic_data(tau=0.3, sigma=0.1, gamma=0.08, 
                                 alpha=0.01, tmax=99, 
                                 network_params=None, model_type='network'):
    """
    Generate synthetic SEIR epidemic data with known parameters
    """
    if model_type == 'network':
        if network_params is None:
            network_params = {
                'n_nodes': 10000,
                'network_type': 'barabasi_albert'
            }
        
        dummy_seir = NetworkSEIR_tuned(pd.DataFrame({'I': [0]}), 
                                       network_params, 
                                       fixed_sigma=sigma, 
                                       fixed_gamma=gamma)
        G = dummy_seir.generate_network()
        days, S, E, I, R = dummy_seir.SEIR_network(G, tau, sigma, 
                                                   gamma, rho, tmax)