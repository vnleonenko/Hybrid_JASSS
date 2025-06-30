import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import random
import os
from tqdm import tqdm
from scipy.stats import uniform, norm, multivariate_normal
import copy
import warnings
import shutil
import time
warnings.filterwarnings('ignore')
from agent_based_model import load_data, preprocess_data, set_initial_values, main_function
from main_pool import Main

class ABC_Agent:
    """
    Approximate Bayesian Computation for agent-based model parameter estimation
    """
    def __init__(self, observed_data, data_path="./chelyabinsk_10/", days=range(1, 100)):
        self.observed_data = observed_data
        self.data_path = data_path
        self.days = days
        
        # define strain keys
        self.strains_keys = ['H1N1', 'H3N2', 'B']
        
        # prepare data only once
        print("Loading and preprocessing data...")
        self.data, self.households, self.dict_school_id = load_data(data_path)
        self.data, self.households, self.dict_school_id = preprocess_data(self.data, self.households, self.dict_school_id)
        self.dict_school_len = [len(self.dict_school_id[i]) for i in self.dict_school_id.keys()]
        
        # store history matching results
        self.hm_results = None
        
    def run_simulation(self, params):
        alpha = params['alpha']
        lmbd = params['lmbd']
        
        try:
            pool = Main(
                strains_keys=self.strains_keys,
                infected_init=[10, 0, 0],
                alpha=[alpha, alpha, alpha],
                lmbd=lmbd
            )
            
            num_runs = 5
            # configure runs
            pool.runs_params(
                num_runs=num_runs,
                days=[1, len(self.days)],
                data_folder=self.data_path
            )
            
            # define age groups
            pool.age_groups_params(
                age_groups=['0-10', '11-17', '18-59', '60-150'],
                vaccined_fraction=[0, 0, 0, 0]
            )
            
            # run simulation
            pool.start(with_seirb=True)
            
            # load ALL results from different seeds
            all_results = []
            for run_number in range(num_runs):
                results_path = os.path.join(pool.results_dir, f"prevalence_seed_{run_number}.csv")
                if os.path.exists(results_path):
                    sim_results = pd.read_csv(results_path, sep='\t')
                    sim_results['run'] = run_number
                    all_results.append(sim_results)
            
            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)
                # average across runs
                avg_results = combined_results.groupby('day').mean().reset_index()
                return avg_results
            else:
                return None
        
        except Exception as e:
            print(f"Simulation error: {e}")
            return None
    
    def calculate_distance(self, sim_data):
        if sim_data is None:
            return np.inf
        
        try:
            # use H1N1 strain ONLY
            min_len = min(len(self.observed_data), len(sim_data))
            obs = self.observed_data['H1N1'].values[:min_len]
            sim = sim_data['H1N1'].values[:min_len]
            
            # mean squared error
            distance = np.mean((obs - sim)**2)
            return distance
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return np.inf
    
    def history_matching(self, prior_ranges, n_samples=100, accept_ratio=0.2):
        """
        Perform history matching to find plausible parameter regions
        """
        print(f"Running history matching with {n_samples} samples...")
        
        # generate samples from prior ranges
        samples = []
        for _ in range(n_samples):
            sample = {}
            for param, (min_val, max_val) in prior_ranges.items():
                sample[param] = uniform.rvs(loc=min_val, scale=max_val-min_val)
            samples.append(sample)
        
        # run simulations and calculate distances
        results = []
        for sample in tqdm(samples, desc="History Matching"):
            sim_data = self.run_simulation(sample)
            distance = self.calculate_distance(sim_data)
            
            # store trajectory data
            result_dict = {
                "alpha": sample["alpha"],
                "lmbd": sample["lmbd"],
                "distance": distance
            }
            
            # add trajectory to results dictionary
            if sim_data is not None:
                result_dict["trajectory"] = sim_data["H1N1"].copy()
            else:
                result_dict["trajectory"] = None
            
            results.append(result_dict)

        results_df = pd.DataFrame(results)
        
        # print distance statistics for debugging
        if not results_df.empty and 'distance' in results_df.columns:
            print(f"Distance stats: min={results_df['distance'].min()}, max={results_df['distance'].max()}, mean={results_df['distance'].mean()}")
        
            # filter results (accept)
            n_accept = max(1, int(len(results_df) * accept_ratio))
            accepted = results_df.nsmallest(n_accept, "distance")
        else:
            print("No valid results from history matching")
            accepted = pd.DataFrame()
        
        print(f"Accepted {len(accepted)} parameter sets")
        
        # store results for other methods to use
        self.hm_results = accepted
        
        return accepted

    def rejection_abc(self, n_samples=100, accept_ratio=0.1):
        if self.hm_results is None or self.hm_results.empty:
            print("Warning: No history matching results available. Cannot run rejection ABC.")
            return pd.DataFrame()
        
        print(f"Running ABC rejection with {n_samples} samples from History Matching...")
        
        # use history matching results to define parameter ranges
        alpha_min = self.hm_results['alpha'].min()
        alpha_max = self.hm_results['alpha'].max()
        lmbd_min = self.hm_results['lmbd'].min()
        lmbd_max = self.hm_results['lmbd'].max()
        
        # sample from history matching parameter space
        samples = []
        for _ in range(n_samples):
            # randomly select a parameter set from history matching results
            hm_idx = np.random.randint(0, len(self.hm_results))
            hm_sample = self.hm_results.iloc[hm_idx]
            
            # add small perturbation to create a new sample
            alpha_perturb = uniform.rvs(loc=-0.05, scale=0.1) # +- 0.05
            lmbd_perturb = uniform.rvs(loc=-0.05, scale=0.1) # +- 0.05
            
            alpha = np.clip(hm_sample['alpha'] + alpha_perturb, alpha_min, alpha_max)
            lmbd = np.clip(hm_sample['lmbd'] + lmbd_perturb, lmbd_min, lmbd_max)
            
            samples.append({"alpha": alpha, "lmbd": lmbd})
        
        # run simulations and calculate distances
        results = []
        for sample in tqdm(samples, desc="ABC Rejection"):
            sim_data = self.run_simulation(sample)
            distance = self.calculate_distance(sim_data)
            
            result_dict = {
                "alpha": sample["alpha"],
                "lmbd": sample["lmbd"],
                "distance": distance
            }
            
            if sim_data is not None:
                result_dict["trajectory"] = sim_data["H1N1"].copy()
            else:
                result_dict["trajectory"] = None
            
            results.append(result_dict)
        
        results_df = pd.DataFrame(results)
        
        if not results_df.empty and 'distance' in results_df.columns:
            # select best samples
            n_accept = max(1, int(len(results_df) * accept_ratio))
            accepted = results_df.nsmallest(n_accept, "distance")
        else:
            accepted = pd.DataFrame()
        
        print(f"ABC rejection accepted {len(accepted)} parameter sets")
        return accepted
    
    def annealing_abc(self, n_samples=100, cooling_steps=3, accept_ratio=0.1):
        if self.hm_results is None or self.hm_results.empty:
            print("Warning: No history matching results available. Cannot run annealing ABC.")
            return pd.DataFrame()
        
        print(f"Running ABC annealing with {cooling_steps} cooling steps...")
        
        # calculate epsilon values for each step
        initial_epsilon = self.hm_results['distance'].quantile(0.5)
        final_epsilon = self.hm_results['distance'].quantile(0.1)
        epsilons = np.geomspace(initial_epsilon, final_epsilon, cooling_steps)
        
        # get parameter bounds from history matching
        alpha_min = self.hm_results['alpha'].min()
        alpha_max = self.hm_results['alpha'].max()
        lmbd_min = self.hm_results['lmbd'].min()
        lmbd_max = self.hm_results['lmbd'].max()
        
        # initial samples from history matching results
        current_samples = []
        for _ in range(n_samples):
            hm_idx = np.random.randint(0, len(self.hm_results))
            hm_sample = self.hm_results.iloc[hm_idx]
            current_samples.append({
                "alpha": hm_sample["alpha"], 
                "lmbd": hm_sample["lmbd"]
            })
        
        # run annealing process
        for step, epsilon in enumerate(epsilons):
            print(f"Annealing step {step+1}/{cooling_steps}, epsilon = {epsilon:.6f}")
            
            # evaluate current samples
            results = []
            for sample in tqdm(current_samples, desc=f"Annealing Step {step+1}"):
                sim_data = self.run_simulation(sample)
                distance = self.calculate_distance(sim_data)
                
                result_dict = {
                    "alpha": sample["alpha"],
                    "lmbd": sample["lmbd"],
                    "distance": distance
                }
                
                if sim_data is not None:
                    result_dict["trajectory"] = sim_data["H1N1"].copy()
                else:
                    result_dict["trajectory"] = None
                
                results.append(result_dict)
            
            # filter accepted samples
            results_df = pd.DataFrame(results)
            
            if not results_df.empty and 'distance' in results_df.columns:
                n_accept = max(1, int(len(results_df) * accept_ratio))
                accepted = results_df.nsmallest(n_accept, "distance")
            else:
                accepted = pd.DataFrame()
            
            # generate new samples for next iteration
            if step < cooling_steps - 1 and not accepted.empty:
                # calculate mean and variance of accepted parameters
                alpha_mean = accepted["alpha"].mean()
                lmbd_mean = accepted["lmbd"].mean()
                
                alpha_var = max(accepted["alpha"].var(), 1e-4)
                lmbd_var = max(accepted["lmbd"].var(), 1e-4)
                
                # generate new samples from normal distribution around accepted values
                current_samples = []
                for _ in range(n_samples):
                    try:
                        alpha = norm.rvs(loc=alpha_mean, scale=np.sqrt(alpha_var))
                        lmbd = norm.rvs(loc=lmbd_mean, scale=np.sqrt(lmbd_var))
                    except ValueError:
                        # fallback if there's an error
                        perturb_scale = 0.05
                        alpha = alpha_mean + np.random.uniform(-perturb_scale, perturb_scale) * (alpha_max - alpha_min)
                        lmbd = lmbd_mean + np.random.uniform(-perturb_scale, perturb_scale) * (lmbd_max - lmbd_min)
                    
                    # ensure parameters are within bounds
                    alpha = max(alpha_min, min(alpha, alpha_max))
                    lmbd = max(lmbd_min, min(lmbd, lmbd_max))
                    
                    current_samples.append({"alpha": alpha, "lmbd": lmbd})
        
        print(f"ABC annealing accepted {len(accepted)} parameter sets")
        return accepted

    def smc_abc(self, n_particles=100, n_populations=3, accept_ratio=0.1):
        if self.hm_results is None or self.hm_results.empty:
            print("Warning: No history matching results available. Cannot run SMC ABC.")
            return pd.DataFrame()
        
        print(f"Running ABC-SMC with {n_populations} populations...")
        
        # calculate epsilon sequence
        initial_epsilon = self.hm_results['distance'].quantile(0.7)
        final_epsilon = self.hm_results['distance'].quantile(0.05)
        epsilons = np.geomspace(initial_epsilon, final_epsilon, n_populations)
        
        # get parameter bounds from history matching
        alpha_min = self.hm_results['alpha'].min()
        alpha_max = self.hm_results['alpha'].max()
        lmbd_min = self.hm_results['lmbd'].min()
        lmbd_max = self.hm_results['lmbd'].max()
        
        # initialize first population from history matching results
        particles = []
        for _ in range(n_particles):
            hm_idx = np.random.randint(0, len(self.hm_results))
            hm_sample = self.hm_results.iloc[hm_idx]
            particles.append({
                "alpha": hm_sample["alpha"], 
                "lmbd": hm_sample["lmbd"]
            })
        
        # equal weights for first population
        weights = np.ones(n_particles) / n_particles
        
        # run SMC process
        for t in range(n_populations):
            epsilon = epsilons[t]
            print(f"SMC Population {t+1}/{n_populations}, epsilon = {epsilon:.6f}")
            
            # evaluate particles and calculate distances
            distances = []
            trajectories = []
            for particle in tqdm(particles, desc=f"SMC Population {t+1}"):
                sim_data = self.run_simulation(particle)
                distance = self.calculate_distance(sim_data)
                distances.append(distance)
                if sim_data is not None:
                    trajectories.append(sim_data["H1N1"].copy())
                else:
                    trajectories.append(None)
            
            # update weights based on epsilon
            new_weights = np.zeros(n_particles)
            for i, distance in enumerate(distances):
                if distance < epsilon:
                    new_weights[i] = weights[i]
            
            # normalize weights
            if np.sum(new_weights) > 0:
                new_weights = new_weights / np.sum(new_weights)
            else:
                print(f"No particles accepted at epsilon = {epsilon}. Taking best particles.")
                sorted_indices = np.argsort(distances)
                for i in range(max(1, int(n_particles * 0.1))):
                    new_weights[sorted_indices[i]] = 1.0
                new_weights = new_weights / np.sum(new_weights)
            
            # calculate effective sample size
            ESS = 1.0 / np.sum(new_weights**2)
            print(f"Effective Sample Size: {ESS:.2f}")
            
            # resample if needed
            if ESS < n_particles / 2 or t == n_populations - 1:
                indices = np.random.choice(n_particles, size=n_particles, p=new_weights)
                resampled_particles = [particles[i] for i in indices]
                resampled_trajectories = [trajectories[i] for i in indices]
                particles = resampled_particles
                trajectories = resampled_trajectories
                weights = np.ones(n_particles) / n_particles
            else:
                weights = new_weights
            
            # if not final iteration, perturb particles
            if t < n_populations - 1:
                # calculate kernel covariance
                alpha_values = np.array([p["alpha"] for p in particles])
                lmbd_values = np.array([p["lmbd"] for p in particles])
                
                params = np.vstack([alpha_values, lmbd_values]).T
                cov = np.cov(params.T) + np.eye(2) * 1e-6
                
                # perturb particles
                new_particles = []
                for i, particle in enumerate(particles):
                    accepted = False
                    attempts = 0
                    while not accepted and attempts < 100:
                        attempts += 1
                        perturbation = multivariate_normal.rvs(mean=[0, 0], cov=cov)
                        new_alpha = particle["alpha"] + perturbation[0]
                        new_lmbd = particle["lmbd"] + perturbation[1]
                        
                        # check if within history matching bounds
                        alpha_in_bounds = alpha_min <= new_alpha <= alpha_max
                        lmbd_in_bounds = lmbd_min <= new_lmbd <= lmbd_max
                        
                        if alpha_in_bounds and lmbd_in_bounds:
                            accepted = True
                            new_particles.append({"alpha": new_alpha, "lmbd": new_lmbd})
                    
                    if not accepted:
                        new_particles.append(particle)
                
                particles = new_particles
        
        # return final particles and weights
        final_results = []
        for i, particle in enumerate(particles):
            final_results.append({
                "alpha": particle["alpha"],
                "lmbd": particle["lmbd"],
                "weight": weights[i],
                "distance": distances[i],
                "trajectory": trajectories[i]
            })
        
        return pd.DataFrame(final_results)

def generate_synthetic_data(alpha=0.78, lmbd=0.4, days=range(1, 100), data_path="./chelyabinsk_10/"):
    """
    Generate synthetic data with the actual agent-based model.
    """
    print(f"Generating ABM data with alpha={alpha}, lmbd={lmbd}")
    
    # create a Main instance with known parameters
    pool = Main(
        strains_keys=['H1N1', 'H3N2', 'B'],
        infected_init=[10, 0, 0], 
        alpha=[alpha, alpha, alpha],
        lmbd=lmbd
    )
    
    # configure the simulation
    num_runs = 1
    pool.runs_params(
        num_runs=num_runs,
        days=[1, len(days)],
        data_folder=data_path
    )
    
    # configure age groups
    pool.age_groups_params(
        age_groups=['0-10', '11-17', '18-59', '60-150'],
        vaccined_fraction=[0, 0, 0, 0]
    )
    
    # run the simulation
    pool.start(with_seirb=True)
    
    # load results
    results_path = os.path.join(pool.results_dir, "prevalence_seed_0.csv")
    data = pd.read_csv(results_path, sep='\t')
    
    return data

def ensure_day_column(data):
    """
    Ensure the data has a 'day' column for plotting
    """
    if 'day' not in data.columns:
        # create day column starting from 1
        data['day'] = range(1, len(data) + 1)
        print("Added 'day' column to data")
    return data

def print_data_info(data, name="Data"):
    """
    Print information about the data structure for debugging
    """
    print(f"\n{name} Info:")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    if len(data) > 0:
        print(f"First few rows:\n{data.head()}")
    print("-" * 50)

class NetworkSEIR_Calibrator:
    """
    Network SEIR model with ABC calibration methods
    """
    def __init__(self, observed_data, network_params=None, fixed_alpha=0.2, fixed_gamma=0.1):
        self.observed_data = observed_data
        self.fixed_alpha = fixed_alpha
        self.fixed_gamma = fixed_gamma
        
        if network_params is None:
            network_params = {
                'n_nodes': 3500,
                'network_type': 'barabasi_albert',
                'network_params': {'m': 5}
            }
        self.network_params = network_params
        self.hm_results = None
        
    def generate_network(self):
        """Generate network based on parameters"""
        n_nodes = self.network_params['n_nodes']
        network_type = self.network_params['network_type']
        net_params = self.network_params['network_params']
        
        if network_type == 'barabasi_albert':
            return nx.barabasi_albert_graph(n_nodes, net_params['m'])
        else:
            raise ValueError(f"Unknown network type: {network_type}")
    
    def seir_network_simulation(self, G, tau, alpha, gamma, rho, tmax, initial_infected_count=None):
        """Run SEIR simulation on network"""
        # initialize all nodes as susceptible
        for node in G.nodes():
            G.nodes[node]['state'] = 0  # S=0, E=1, I=2, R=3
        
        # set initial infected nodes
        if initial_infected_count is not None:
            initial_infected = min(initial_infected_count, len(G.nodes()) // 2)
        else:
            initial_infected = max(1, int(rho * len(G.nodes())))
        
        initial_infected_nodes = random.sample(list(G.nodes()), initial_infected)
        for node in initial_infected_nodes:
            G.nodes[node]['state'] = 2
        
        # track counts over time
        counts = {'S': [], 'E': [], 'I': [], 'R': []}
        
        for day in range(tmax + 1):
            # count current states
            for state, state_list in zip([0, 1, 2, 3], ['S', 'E', 'I', 'R']):
                counts[state_list].append(sum(1 for n in G.nodes if G.nodes[n]['state'] == state))
            
            if day == tmax:
                break
                
            # update states
            new_states = {}
            
            for node in G.nodes():
                current_state = G.nodes[node]['state']
                
                if current_state == 2:  # infected
                    # infect neighbors
                    for neighbor in G.neighbors(node):
                        if G.nodes[neighbor]['state'] == 0:  # susceptible
                            if random.random() < tau:
                                new_states[neighbor] = 1  # exposed
                    if random.random() < gamma:
                        new_states[node] = 3  # recovered
                elif current_state == 1:  # exposed
                    if random.random() < alpha:
                        new_states[node] = 2  # become infected
            
            # apply state changes
            for node, new_state in new_states.items():
                G.nodes[node]['state'] = new_state
        
        return counts
    
    def run_simulation(self, tau, rho, initial_infected_count=None):
        """
        Run single SEIR simulation
        """
        try:
            G = self.generate_network()
            tmax = len(self.observed_data) - 1
            counts = self.seir_network_simulation(G, tau, self.fixed_alpha, self.fixed_gamma, rho, tmax, initial_infected_count)
            
            sim_results = pd.DataFrame({
                'day': range(tmax + 1),
                'S': counts['S'],
                'E': counts['E'],
                'I': counts['I'],
                'R': counts['R']
            })
            
            return sim_results
        except Exception as e:
            print(f"SEIR Simulation error: {e}")
            return None
    
    def calculate_distance(self, sim_data):
        """
        Calculate MSE distance between simulated and observed data
        """
        if sim_data is None:
            return np.inf
        
        try:
            min_len = min(len(self.observed_data), len(sim_data))
            obs = self.observed_data['I'].values[:min_len]
            sim = sim_data['I'].values[:min_len]
            distance = np.mean((obs - sim)**2)
            return distance
        except Exception as e:
            return np.inf
    
    def history_matching(self, prior_ranges, n_samples=200, accept_ratio=0.2):
        """
        History matching calibration
        """
        print(f"Running SEIR history matching with {n_samples} samples...")
        
        samples = []
        for _ in range(n_samples):
            sample = {}
            for param, (min_val, max_val) in prior_ranges.items():
                sample[param] = uniform.rvs(loc=min_val, scale=max_val-min_val)
            samples.append(sample)
        
        results = []
        for sample in tqdm(samples, desc="SEIR History Matching"):
            sim_data = self.run_simulation(sample['tau'], sample['rho'])
            distance = self.calculate_distance(sim_data)
            
            result_dict = {
                "tau": sample["tau"],
                "rho": sample["rho"],
                "distance": distance
            }
            
            if sim_data is not None:
                result_dict["trajectory"] = sim_data["I"].values.copy()
            else:
                result_dict["trajectory"] = None
            
            results.append(result_dict)
        
        results_df = pd.DataFrame(results)
        
        if not results_df.empty and 'distance' in results_df.columns:
            n_accept = max(1, int(len(results_df) * accept_ratio))
            accepted = results_df.nsmallest(n_accept, "distance")
        else:
            accepted = pd.DataFrame()
        
        self.hm_results = accepted
        print(f"SEIR History matching accepted {len(accepted)} parameter sets")
        return accepted
    
    def rejection_abc(self, n_samples=100, accept_ratio=0.1):
        """
        ABC rejection sampling
        """
        if self.hm_results is None or self.hm_results.empty:
            print("Warning: No SEIR history matching results available. Cannot run rejection ABC.")
            return pd.DataFrame()
        
        print(f"Running SEIR ABC rejection with {n_samples} samples...")
        
        param_bounds = {
            'tau': (self.hm_results['tau'].min(), self.hm_results['tau'].max()),
            'rho': (self.hm_results['rho'].min(), self.hm_results['rho'].max())
        }
        
        samples = []
        for _ in range(n_samples):
            hm_idx = np.random.randint(0, len(self.hm_results))
            hm_sample = self.hm_results.iloc[hm_idx]
            
            sample = {}
            for param in ['tau', 'rho']:
                perturb = uniform.rvs(loc=-0.02, scale=0.04)
                param_min, param_max = param_bounds[param]
                sample[param] = np.clip(hm_sample[param] + perturb, param_min, param_max)
            samples.append(sample)
        
        results = []
        for sample in tqdm(samples, desc="SEIR ABC Rejection"):
            sim_data = self.run_simulation(sample['tau'], sample['rho'])
            distance = self.calculate_distance(sim_data)
            
            result_dict = {
                "tau": sample["tau"],
                "rho": sample["rho"],
                "distance": distance
            }
            
            if sim_data is not None:
                result_dict["trajectory"] = sim_data["I"].values.copy()
            else:
                result_dict["trajectory"] = None
            
            results.append(result_dict)
        
        results_df = pd.DataFrame(results)
        
        if not results_df.empty and 'distance' in results_df.columns:
            n_accept = max(1, int(len(results_df) * accept_ratio))
            accepted = results_df.nsmallest(n_accept, "distance")
        else:
            accepted = pd.DataFrame()
        
        print(f"SEIR ABC rejection accepted {len(accepted)} parameter sets")
        return accepted
    
    def annealing_abc(self, n_samples=50, cooling_steps=3, accept_ratio=0.1):
        """
        ABC with simulated annealing
        """
        if self.hm_results is None or self.hm_results.empty:
            print("Warning: No SEIR history matching results available. Cannot run annealing ABC.")
            return pd.DataFrame()
        
        print(f"Running SEIR ABC annealing with {cooling_steps} cooling steps...")
        
        initial_epsilon = self.hm_results['distance'].quantile(0.5)
        final_epsilon = self.hm_results['distance'].quantile(0.1)
        epsilons = np.geomspace(initial_epsilon, final_epsilon, cooling_steps)
        
        param_bounds = {
            'tau': (self.hm_results['tau'].min(), self.hm_results['tau'].max()),
            'rho': (self.hm_results['rho'].min(), self.hm_results['rho'].max())
        }
        
        current_samples = []
        for _ in range(n_samples):
            hm_idx = np.random.randint(0, len(self.hm_results))
            hm_sample = self.hm_results.iloc[hm_idx]
            sample = {param: hm_sample[param] for param in ['tau', 'rho']}
            current_samples.append(sample)
        
        for step, epsilon in enumerate(epsilons):
            print(f"SEIR Annealing step {step+1}/{cooling_steps}, epsilon = {epsilon:.2f}")
            
            results = []
            for sample in tqdm(current_samples, desc=f"SEIR Annealing Step {step+1}"):
                sim_data = self.run_simulation(sample['tau'], sample['rho'])
                distance = self.calculate_distance(sim_data)
                
                result_dict = {
                    "tau": sample["tau"],
                    "rho": sample["rho"],
                    "distance": distance
                }
                
                if sim_data is not None:
                    result_dict["trajectory"] = sim_data["I"].values.copy()
                else:
                    result_dict["trajectory"] = None
                
                results.append(result_dict)
            
            results_df = pd.DataFrame(results)
            
            if not results_df.empty and 'distance' in results_df.columns:
                n_accept = max(1, int(len(results_df) * accept_ratio))
                accepted = results_df.nsmallest(n_accept, "distance")
            else:
                accepted = pd.DataFrame()
            
            if step < cooling_steps - 1 and not accepted.empty:
                new_samples = []
                for _ in range(n_samples):
                    sample = {}
                    for param in ['tau', 'rho']:
                        param_mean = accepted[param].mean()
                        param_std = max(accepted[param].std(), 0.01)
                        
                        new_value = norm.rvs(loc=param_mean, scale=param_std)
                        param_min, param_max = param_bounds[param]
                        sample[param] = np.clip(new_value, param_min, param_max)
                    
                    new_samples.append(sample)
                current_samples = new_samples
        
        print(f"SEIR ABC annealing accepted {len(accepted)} parameter sets")
        return accepted
    
    def smc_abc(self, n_particles=50, n_populations=3, accept_ratio=0.1):
        """
        ABC Sequential Monte Carlo
        """
        if self.hm_results is None or self.hm_results.empty:
            print("Warning: No SEIR history matching results available. Cannot run SMC ABC.")
            return pd.DataFrame()
        
        print(f"Running SEIR ABC-SMC with {n_populations} populations...")
        
        initial_epsilon = self.hm_results['distance'].quantile(0.7)
        final_epsilon = self.hm_results['distance'].quantile(0.05)
        epsilons = np.geomspace(initial_epsilon, final_epsilon, n_populations)
        
        param_bounds = {
            'tau': (self.hm_results['tau'].min(), self.hm_results['tau'].max()),
            'rho': (self.hm_results['rho'].min(), self.hm_results['rho'].max())
        }
        
        particles = []
        for _ in range(n_particles):
            hm_idx = np.random.randint(0, len(self.hm_results))
            hm_sample = self.hm_results.iloc[hm_idx]
            particle = {param: hm_sample[param] for param in ['tau', 'rho']}
            particles.append(particle)
        
        weights = np.ones(n_particles) / n_particles
        
        for t in range(n_populations):
            epsilon = epsilons[t]
            print(f"SEIR SMC Population {t+1}/{n_populations}, epsilon = {epsilon:.2f}")
            
            distances = []
            trajectories = []
            
            for particle in tqdm(particles, desc=f"SEIR SMC Population {t+1}"):
                sim_data = self.run_simulation(particle['tau'], particle['rho'])
                distance = self.calculate_distance(sim_data)
                distances.append(distance)
                
                if sim_data is not None:
                    trajectories.append(sim_data['I'].values.copy())
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
            print(f"SEIR Effective sample size: {ESS:.1f}")
            
            if ESS < n_particles / 2:
                indices = np.random.choice(n_particles, size=n_particles, p=new_weights)
                particles = [particles[i] for i in indices]
                trajectories = [trajectories[i] for i in indices]
                weights = np.ones(n_particles) / n_particles
            else:
                weights = new_weights
            
            if t < n_populations - 1:
                param_values = np.array([[p['tau'], p['rho']] for p in particles])
                cov = np.cov(param_values.T) + np.eye(2) * 1e-6
                
                new_particles = []
                for particle in particles:
                    attempts = 0
                    while attempts < 50:
                        perturbation = multivariate_normal.rvs(mean=[0, 0], cov=cov)
                        new_particle = {
                            'tau': particle['tau'] + perturbation[0],
                            'rho': particle['rho'] + perturbation[1]
                        }
                        
                        tau_valid = param_bounds['tau'][0] <= new_particle['tau'] <= param_bounds['tau'][1]
                        rho_valid = param_bounds['rho'][0] <= new_particle['rho'] <= param_bounds['rho'][1]
                        
                        if tau_valid and rho_valid:
                            new_particles.append(new_particle)
                            break
                        attempts += 1
                    
                    if attempts >= 50:
                        new_particles.append(particle)
                
                particles = new_particles
        
        final_results = []
        for i, particle in enumerate(particles):
            final_results.append({
                "tau": particle["tau"],
                "rho": particle["rho"],
                "weight": weights[i],
                "distance": distances[i],
                "trajectory": trajectories[i]
            })
        
        results_df = pd.DataFrame(final_results)
        print(f"SEIR ABC-SMC completed with {len(results_df)} particles")
        return results_df

def get_best_params_safely(results):
    """
    Safely get best parameters from results DataFrame
    """
    if results is None or results.empty:
        return None
    if 'distance' not in results.columns:
        return None
    
    try:
        best_idx = results['distance'].idxmin()
        if pd.isna(best_idx) or best_idx not in results.index:
            return None
        return results.loc[best_idx]
    except Exception as e:
        print(f"Error getting best parameters: {e}")
        return None

class ABM_to_NetworkSEIR_Framework:
    """
    Complete framework for switching from ABM to Network SEIR model
    """
    
    def __init__(self, total_population=10000, switch_fraction=0.01, switch_day=None, data_path="./chelyabinsk_10/"):
        self.total_population = total_population
        self.switch_fraction = switch_fraction
        self.switch_day = switch_day  # support for user-defined switch day
        self.data_path = data_path
        self.abm_data = None
        self.switch_point = None
        self.calibration_results = {}
        self.combined_trajectories = {}
        
    def generate_abm_data(self, alpha=0.78, lmbd=0.4, days=100):
        """
        Generate ABM data using actual ABM
        """
        print(f"Generating ABM data with alpha={alpha}, lmbd={lmbd}, days={days}")
        
        self.abm_data = generate_synthetic_data(
            alpha=alpha, 
            lmbd=lmbd, 
            days=range(1, days+1),
            data_path=self.data_path
        )
        
        self.abm_data = ensure_day_column(self.abm_data)
        
        print_data_info(self.abm_data, "ABM Data")
        
        return self.abm_data
    
    def find_switch_point(self):
        """
        Determine when to switch from ABM to Network SEIR model
        Support for both fraction-based and day-based switching
        """
        if self.abm_data is None:
            raise ValueError("Must generate ABM data first")
        
        # calculate infection fraction
        infected_fraction = self.abm_data['H1N1'] / self.total_population
        
        if self.switch_day is not None:
            # user-defined switch day takes priority
            self.switch_point = min(self.switch_day, len(self.abm_data) - 1)
            print(f"Switching at user-specified day: {self.switch_point}")
        else:
            # switch based on infection fraction
            switch_candidates = infected_fraction >= self.switch_fraction
            if switch_candidates.any():
                self.switch_point = switch_candidates.idxmax()
                print(f"Switching when infection fraction reaches {self.switch_fraction*100:.1f}% at day: {self.switch_point}")
            else:
                # !!! IF THE FRACTION WAS NOT REACHED
                # fallback: switch at 2/3 of simulation period
                self.switch_point = len(self.abm_data) * 2 // 3
                print(f"Threshold not reached, switching at day: {self.switch_point}")
        
        return self.switch_point

def run_complete_framework(alpha=0.78, lmbd=0.4, days=100, switch_fraction=0.01, switch_day=None, data_path="./chelyabinsk_10/"):
    """
    Run the complete ABM to Network SEIR framework with proper plotting
    """
    framework = ABM_to_NetworkSEIR_Framework(
        total_population=10000,
        switch_fraction=switch_fraction,
        switch_day=switch_day,  # pass user-defined switch day
        data_path=data_path
    )
    
    # Step 1: Generate ABM data
    print("\nStep 1: Generating ABM data...")
    abm_data = framework.generate_abm_data(alpha=alpha, lmbd=lmbd, days=days)
    
    # Step 2: Find switch point
    print("\nStep 2: Finding switch point...")
    switch_point = framework.find_switch_point()
    
    # Step 3: Prepare data for SEIR calibration
    print(f"\nStep 3: Preparing data from day {switch_point} onwards for SEIR calibration...")
    seir_observed_data = abm_data.iloc[switch_point:].reset_index(drop=True)
    seir_observed_data['I'] = seir_observed_data['H1N1']
    
    # Debug: print SEIR data info
    #print_data_info(seir_observed_data, "SEIR Observed Data")
    
    # Get initial infected count at switch point
    initial_infected_count = int(abm_data.iloc[switch_point]['H1N1'])
    print(f"Initial infected count at switch: {initial_infected_count}")
    
    # Step 4: Run SEIR calibration
    print("\nStep 4: Running Network-SEIR calibration...")
    seir_calibrator = NetworkSEIR_Calibrator(seir_observed_data)
    
    prior_ranges = {
        "tau": (0.001, 0.5),
        "rho": (0.0001, 0.1)
    }
    
    calibration_results = {}
    
    try:
        print("Running SEIR History Matching...")
        calibration_results['history_matching'] = seir_calibrator.history_matching(
            prior_ranges, n_samples=50  # reduced for faster execution, can be changed
        )
        
        print("Running SEIR Rejection ABC...")
        calibration_results['rejection'] = seir_calibrator.rejection_abc(
            n_samples=30 # reduced for faster execution, can be changed
        )
        
        print("Running SEIR Annealing ABC...")
        calibration_results['annealing'] = seir_calibrator.annealing_abc(
            n_samples=20, cooling_steps=3 # reduced for faster execution, can be changed
        )
        
        print("Running SEIR SMC ABC...")
        calibration_results['smc'] = seir_calibrator.smc_abc(
            n_particles=20, n_populations=3 # reduced for faster execution, can be changed
        )
        
    except Exception as e:
        print(f"Error in SEIR calibration: {e}")
    
    # Step 5: Generate complete trajectories with best parameters
    print("\nStep 5: Generating complete trajectories with calibrated parameters...")
    
    combined_trajectories = {}
    best_parameters = {}
    
    for method, results in calibration_results.items():
        if not results.empty:
            best_params = get_best_params_safely(results)
            
            if best_params is not None:
                best_parameters[method] = {
                    'tau': best_params['tau'],
                    'rho': best_params['rho'],
                    'distance': best_params['distance']
                }
                
                seir_sim = seir_calibrator.run_simulation(
                    tau=best_params['tau'], 
                    rho=best_params['rho'],
                    initial_infected_count=initial_infected_count
                )
                
                if seir_sim is not None:
                    combined_trajectory = []
                    for i in range(switch_point):
                        combined_trajectory.append(abm_data.iloc[i]['H1N1'])
                    for i in range(len(seir_sim)):
                        if switch_point + i < len(abm_data): 
                            combined_trajectory.append(seir_sim.iloc[i]['I'])
                    combined_trajectories[method] = combined_trajectory
                else:
                    print(f"Warning: SEIR simulation failed for {method}")
            else:
                print(f"Warning: No valid parameters found for {method}")
    
    # Step 6: Save results and create plots
    print("\nStep 6: Saving results and creating plots...")
    
    observed_filename = f"observed_real_data_alpha_{alpha}_lmbd_{lmbd}.csv"
    abm_data.to_csv(observed_filename, index=False)
    print(f"Saved observed data to: {observed_filename}")
    
    for method, results in calibration_results.items():
        if not results.empty:
            params_filename = f"{method}_parameters_SEIR.csv"
            # TODO: HERE WE REMOVE THE TRAJECTORY DATA BUT IT CAN BE SAVED
            results_to_save = results.drop('trajectory', axis=1, errors='ignore')
            results_to_save.to_csv(params_filename, index=False)
            print(f"Saved {method} parameters to: {params_filename}")
    try:
        fig = plt.figure(figsize=(24, 12))
        spec = gridspec.GridSpec(ncols=6, nrows=2, figure=fig)
        ax1 = fig.add_subplot(spec[0, 0:2]) # complete trajectories
        ax2 = fig.add_subplot(spec[0, 2:4]) # switch region zoom  
        ax3 = fig.add_subplot(spec[0, 4:]) # Tau distribution
        ax4 = fig.add_subplot(spec[1, 1:3]) # Rho distribution
        ax5 = fig.add_subplot(spec[1, 3:5]) # distance comparison

        abm_data = ensure_day_column(abm_data)

        # Plot 1: Complete trajectories (ABM + All SEIR methods) with success indicators
        ax1.plot(abm_data['day'], abm_data['H1N1'], 'k-', linewidth=3, label='ABM Data', alpha=0.8)
        ax1.axvline(switch_point, color='red', linestyle='--', linewidth=2, label=f'Switch at day {switch_point}')
        colors = ['blue', 'green', 'orange', 'purple']
        line_styles = ['-', '--', '-.', ':']
        all_methods = ['history_matching', 'rejection', 'annealing', 'smc']
        for i, method in enumerate(all_methods):
            if method in combined_trajectories and combined_trajectories[method] and len(combined_trajectories[method]) > 0:
                # Successful method - tick
                trajectory = combined_trajectories[method]
                days_full = list(range(len(trajectory)))
                ax1.plot(days_full, trajectory, color=colors[i], linewidth=2.5, 
                        linestyle=line_styles[i], alpha=0.8, label=f'✓ SEIR {method.title()}')
            else:
                # Failed method - cross (we create empty plot for legend)
                ax1.plot([], [], color=colors[i], linewidth=2, linestyle='--', 
                        alpha=0.4, label=f'✗ SEIR {method.title()}')

        ax1.set_title('Hybrid: ABM → Network SEIR', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Infected')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Zoomed view of switch region for better understanding
        zoom_start = max(0, switch_point - 10)
        zoom_end = min(len(abm_data), switch_point + 20)

        if zoom_end > zoom_start:
            ax2.plot(abm_data['day'][zoom_start:zoom_end], abm_data['H1N1'][zoom_start:zoom_end], 
                     'k-', linewidth=3, label='ABM Data')
            ax2.axvline(switch_point, color='red', linestyle='--', linewidth=2, label=f'Switch point')

            for i, method in enumerate(all_methods):
                if method in combined_trajectories and combined_trajectories[method]:
                    trajectory = combined_trajectories[method]
                    if len(trajectory) > zoom_start:
                        days_zoom = list(range(zoom_start, min(zoom_end, len(trajectory))))
                        traj_zoom = trajectory[zoom_start:min(zoom_end, len(trajectory))]
                        if len(days_zoom) == len(traj_zoom) and len(days_zoom) > 0:
                            ax2.plot(days_zoom, traj_zoom, color=colors[i], linewidth=2.5, 
                                    linestyle=line_styles[i], alpha=0.8, label=f'SEIR {method.title()}')

        ax2.set_title('Switch region zoomed view', fontsize=12)
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Infected')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Parameter distributions (tau) with success indicators
        method_order = list(calibration_results.keys())
        for i, method in enumerate(method_order):
            results = calibration_results[method]
            if not results.empty and 'tau' in results.columns:
                tau_values = results['tau'].dropna()
                if len(tau_values) > 0:
                    success_indicator = '✓' if method in combined_trajectories else '✗'
                    ax3.hist(tau_values, alpha=0.6, bins=min(15, len(tau_values)), 
                            label=f'{success_indicator} {method.title()}', color=colors[i], density=True)

        ax3.set_title('Tau parameter distribution', fontsize=12)
        ax3.set_xlabel('Tau (transmission rate)')
        ax3.set_ylabel('Density')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Parameter distributions (rho) with success indicators
        for i, method in enumerate(method_order):
            results = calibration_results[method]
            if not results.empty and 'rho' in results.columns:
                rho_values = results['rho'].dropna()
                if len(rho_values) > 0:
                    success_indicator = '✓' if method in combined_trajectories else '✗'
                    ax4.hist(rho_values, alpha=0.6, bins=min(15, len(rho_values)), 
                            label=f'{success_indicator} {method.title()}', color=colors[i], density=True)

        ax4.set_title('Rho parameter Ddistribution', fontsize=12)
        ax4.set_xlabel('Rho (initial infection fraction)')
        ax4.set_ylabel('Density')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        # Plot 5: Best distances comparison with success indicators
        methods = []
        distances = []
        method_colors = []

        for i, method in enumerate(method_order):
            results = calibration_results[method]
            if not results.empty and 'distance' in results.columns:
                success_indicator = '✓' if method in combined_trajectories else '✗'
                methods.append(f'{success_indicator} {method.title()}')
                distances.append(results['distance'].min())
                method_colors.append(colors[i])

        if methods and distances:
            bars = ax5.bar(methods, distances, color=method_colors, alpha=0.7)
            ax5.set_title('Best distance by method', fontsize=12)
            ax5.set_ylabel('Distance (MSE)')
            ax5.tick_params(axis='x', rotation=45)
            for bar, dist in zip(bars, distances):
                if not np.isnan(dist) and not np.isinf(dist):
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(distances)*0.01,
                            f'{dist:.0f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout(pad=3.0)
        plot_filename = f"ABM_to_Network_SEIR_alpha_{alpha}_lmbd_{lmbd}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {plot_filename}")
        plt.show()
    
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()

    print(f"ABM Parameters: alpha={alpha}, lmbd={lmbd}")
    print(f"Switch Point: Day {switch_point}")
    if framework.switch_day is not None:
        print(f"Switch mode: user-defined day ({framework.switch_day})")
    else:
        print(f"Switch mode: infection fraction threshold ({framework.switch_fraction*100:.1f}%)")
    print(f"Post-switch data points: {len(seir_observed_data)}")
    print(f"Initial infected count at switch: {initial_infected_count}")
    
    for method, params in best_parameters.items():
        print(f"\n{method.upper()} best parameters:")
        print(f"  Tau: {params['tau']:.6f}")
        print(f"  Rho: {params['rho']:.6f}")
        print(f"  Distance: {params['distance']:.2f}")
    
    return abm_data, calibration_results, switch_point, combined_trajectories, best_parameters

if __name__ == "__main__":
    # 1: Switch at infection fraction threshold
    fraction = 0.10
    print(f"Running with infection fraction threshold ({fraction*100}%)...")
    try:
        results_1 = run_complete_framework(
            alpha=0.78, 
            lmbd=0.4, 
            days=100, 
            switch_fraction=fraction,
            switch_day=None,  # we use fraction-based switching
            data_path="./chelyabinsk_10/"
        )
        
        if results_1:
            abm_data, calibration_results, switch_point, combined_trajectories, best_parameters = results_1
            print(f"\nCompleted! Switch occurred at day {switch_point}")
            print(f"Generated {len(combined_trajectories)} model trajectories")
    
    except Exception as e:
        print(f"Error in first run: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-"*80 + "\n")
    
    # switch at the specific day
    day = 12
    print(f"Running with user-defined switch day (day {day})...")
    try:
        results_2 = run_complete_framework(
            alpha=0.78, 
            lmbd=0.4, 
            days=100, 
            switch_fraction=0.01,  # this will be ignored
            switch_day=day,  # switch at day X regardless of infection level
            data_path="./chelyabinsk_10/"
        )
        
        if results_2:
            abm_data_2, calibration_results_2, switch_point_2, combined_trajectories_2, best_parameters_2 = results_2
            print(f"\nCompleted! Real switch occurred at day {switch_point_2}")
            print(f"Generated {len(combined_trajectories_2)} model trajectories")
    
    except Exception as e:
        print(f"Error in second run: {e}")
        import traceback
        traceback.print_exc()
