import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import os
from tqdm import tqdm
from scipy.stats import uniform, norm, multivariate_normal
from scipy.integrate import odeint
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
        """
        Run ABM simulation with given parameters
        """
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
            pool.runs_params(
                num_runs=num_runs,
                days=[1, len(self.days)],
                data_folder=self.data_path
            )
            pool.age_groups_params(
                age_groups=['0-10', '11-17', '18-59', '60-150'],
                vaccined_fraction=[0, 0, 0, 0]
            )
            pool.start(with_seirb=True)
            all_results = []
            for run_number in range(num_runs):
                results_path = os.path.join(pool.results_dir, f"prevalence_seed_{run_number}.csv")
                if os.path.exists(results_path):
                    sim_results = pd.read_csv(results_path, sep='\t')
                    sim_results['run'] = run_number
                    all_results.append(sim_results)
            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)
                avg_results = combined_results.groupby('day').mean().reset_index()
                return avg_results
            else:
                return None
        except Exception as e:
            print(f"ABM Simulation error: {e}")
            return None
    
    def calculate_distance(self, sim_data):
        """Calculate MSE distance between simulated and observed data"""
        if sim_data is None:
            return np.inf
        try:
            min_len = min(len(self.observed_data), len(sim_data))
            obs = self.observed_data['H1N1'].values[:min_len]
            sim = sim_data['H1N1'].values[:min_len]
            distance = np.mean((obs - sim)**2)
            return distance
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return np.inf
    
    def history_matching(self, prior_ranges, n_samples=100, accept_ratio=0.2):
        """Perform history matching to find plausible parameter regions"""
        print(f"Running ABM history matching with {n_samples} samples...")
        samples = []
        for _ in range(n_samples):
            sample = {}
            for param, (min_val, max_val) in prior_ranges.items():
                sample[param] = uniform.rvs(loc=min_val, scale=max_val-min_val)
            samples.append(sample)
        results = []
        for sample in tqdm(samples, desc="ABM History Matching"):
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
            print(f"Distance stats: min={results_df['distance'].min()}, max={results_df['distance'].max()}, mean={results_df['distance'].mean()}")
            n_accept = max(1, int(len(results_df) * accept_ratio))
            accepted = results_df.nsmallest(n_accept, "distance")
        else:
            print("No valid results from ABM history matching")
            accepted = pd.DataFrame()
        print(f"ABM History matching accepted {len(accepted)} parameter sets")
        self.hm_results = accepted
        return accepted
    
    def rejection_abc(self, n_samples=100, accept_ratio=0.1):
        """ABC rejection sampling"""
        if self.hm_results is None or self.hm_results.empty:
            print("Warning: No ABM history matching results available. Cannot run rejection ABC.")
            return pd.DataFrame()
        print(f"Running ABM ABC rejection with {n_samples} samples...")
        param_bounds = {
            'alpha': (self.hm_results['alpha'].min(), self.hm_results['alpha'].max()),
            'lmbd': (self.hm_results['lmbd'].min(), self.hm_results['lmbd'].max())
        }
        samples = []
        for _ in range(n_samples):
            hm_idx = np.random.randint(0, len(self.hm_results))
            hm_sample = self.hm_results.iloc[hm_idx]
            sample = {}
            for param in ['alpha', 'lmbd']:
                perturb = uniform.rvs(loc=-0.02, scale=0.04)
                param_min, param_max = param_bounds[param]
                sample[param] = np.clip(hm_sample[param] + perturb, param_min, param_max)
            samples.append(sample)
        results = []
        for sample in tqdm(samples, desc="ABM ABC Rejection"):
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
            n_accept = max(1, int(len(results_df) * accept_ratio))
            accepted = results_df.nsmallest(n_accept, "distance")
        else:
            accepted = pd.DataFrame()
        print(f"ABM ABC rejection accepted {len(accepted)} parameter sets")
        return accepted
    def annealing_abc(self, n_samples=50, cooling_steps=3, accept_ratio=0.1):
        """ABC with simulated annealing"""
        if self.hm_results is None or self.hm_results.empty:
            print("Warning: No ABM history matching results available. Cannot run annealing ABC.")
            return pd.DataFrame()
        print(f"Running ABM ABC annealing with {cooling_steps} cooling steps...")
        initial_epsilon = self.hm_results['distance'].quantile(0.5)
        final_epsilon = self.hm_results['distance'].quantile(0.1)
        epsilons = np.geomspace(initial_epsilon, final_epsilon, cooling_steps)
        param_bounds = {
            'alpha': (self.hm_results['alpha'].min(), self.hm_results['alpha'].max()),
            'lmbd': (self.hm_results['lmbd'].min(), self.hm_results['lmbd'].max())
        }
        current_samples = []
        for _ in range(n_samples):
            hm_idx = np.random.randint(0, len(self.hm_results))
            hm_sample = self.hm_results.iloc[hm_idx]
            sample = {param: hm_sample[param] for param in ['alpha', 'lmbd']}
            current_samples.append(sample)
        for step, epsilon in enumerate(epsilons):
            print(f"ABM Annealing step {step+1}/{cooling_steps}, epsilon = {epsilon:.2f}")
            results = []
            for sample in tqdm(current_samples, desc=f"ABM Annealing Step {step+1}"):
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
                n_accept = max(1, int(len(results_df) * accept_ratio))
                accepted = results_df.nsmallest(n_accept, "distance")
            else:
                accepted = pd.DataFrame()
            if step < cooling_steps - 1 and not accepted.empty:
                new_samples = []
                for _ in range(n_samples):
                    sample = {}
                    for param in ['alpha', 'lmbd']:
                        param_mean = accepted[param].mean()
                        param_std = max(accepted[param].std(), 0.01)
                        new_value = norm.rvs(loc=param_mean, scale=param_std)
                        param_min, param_max = param_bounds[param]
                        sample[param] = np.clip(new_value, param_min, param_max)
                    new_samples.append(sample)
                current_samples = new_samples
        print(f"ABM ABC annealing accepted {len(accepted)} parameter sets")
        return accepted
    
    def smc_abc(self, n_particles=50, n_populations=3, accept_ratio=0.1):
        """ABC Sequential Monte Carlo"""
        if self.hm_results is None or self.hm_results.empty:
            print("Warning: No ABM history matching results available. Cannot run SMC ABC.")
            return pd.DataFrame()
        print(f"Running ABM ABC-SMC with {n_populations} populations...")
        initial_epsilon = self.hm_results['distance'].quantile(0.7)
        final_epsilon = self.hm_results['distance'].quantile(0.05)
        epsilons = np.geomspace(initial_epsilon, final_epsilon, n_populations)
        param_bounds = {
            'alpha': (self.hm_results['alpha'].min(), self.hm_results['alpha'].max()),
            'lmbd': (self.hm_results['lmbd'].min(), self.hm_results['lmbd'].max())
        }
        particles = []
        for _ in range(n_particles):
            hm_idx = np.random.randint(0, len(self.hm_results))
            hm_sample = self.hm_results.iloc[hm_idx]
            particle = {param: hm_sample[param] for param in ['alpha', 'lmbd']}
            particles.append(particle)
        weights = np.ones(n_particles) / n_particles
        for t in range(n_populations):
            epsilon = epsilons[t]
            print(f"ABM SMC Population {t+1}/{n_populations}, epsilon = {epsilon:.2f}")
            distances = []
            trajectories = []
            for particle in tqdm(particles, desc=f"ABM SMC Population {t+1}"):
                sim_data = self.run_simulation(particle)
                distance = self.calculate_distance(sim_data)
                distances.append(distance)
                if sim_data is not None:
                    trajectories.append(sim_data['H1N1'].copy())
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
            print(f"ABM Effective sample size: {ESS:.1f}")
            if ESS < n_particles / 2:
                indices = np.random.choice(n_particles, size=n_particles, p=new_weights)
                particles = [particles[i] for i in indices]
                trajectories = [trajectories[i] for i in indices]
                weights = np.ones(n_particles) / n_particles
            else:
                weights = new_weights
            if t < n_populations - 1:
                param_values = np.array([[p['alpha'], p['lmbd']] for p in particles])
                cov = np.cov(param_values.T) + np.eye(2) * 1e-6
                new_particles = []
                for particle in particles:
                    attempts = 0
                    while attempts < 50:
                        perturbation = multivariate_normal.rvs(mean=[0, 0], cov=cov)
                        new_particle = {
                            'alpha': particle['alpha'] + perturbation[0],
                            'lmbd': particle['lmbd'] + perturbation[1]
                        }
                        alpha_valid = param_bounds['alpha'][0] <= new_particle['alpha'] <= param_bounds['alpha'][1]
                        lmbd_valid = param_bounds['lmbd'][0] <= new_particle['lmbd'] <= param_bounds['lmbd'][1]
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
                "alpha": particle["alpha"],
                "lmbd": particle["lmbd"],
                "weight": weights[i],
                "distance": distances[i],
                "trajectory": trajectories[i]
            })
        results_df = pd.DataFrame(final_results)
        print(f"ABM ABC-SMC completed with {len(results_df)} particles")
        return results_df

class SEIR_Calibrator:
    def __init__(self, observed_data, population=10000, fixed_alpha=1/3, fixed_gamma=1/5):
        self.observed_data = observed_data
        self.population = population
        self.fixed_alpha = fixed_alpha  # 1/incubation_period
        self.fixed_gamma = fixed_gamma  # 1/infectious_period
        self.hm_results = None
    
    def seir_model(self, y, t, beta, alpha, gamma):
        """
        SEIR differential equation model
        """
        S, E, I, R = y
        dSdt = -beta * S * I 
        dEdt = beta * S * I  - alpha * E
        dIdt = alpha * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    def run_simulation(self, beta, initial_infected, alpha, t_max=None):
        """
        Run SEIR simulation
        """
        try:
            if t_max is None:
                t_max = len(self.observed_data) - 1
            I0 = initial_infected * self.population
            E0 = 0
            R0 = alpha * self.population
            S0 = self.population - I0 - E0 - R0
            y0 = [S0, E0, I0, R0]
            t = np.linspace(0, t_max, t_max + 1)
            solution = odeint(self.seir_model, y0, t, args=(beta, self.fixed_alpha, self.fixed_gamma))
            sim_results = pd.DataFrame({
                'day': range(t_max + 1),
                'S': solution[:, 0],
                'E': solution[:, 1],
                'I': solution[:, 2],
                'R': solution[:, 3]
            })
            return sim_results
        except Exception as e:
            print(f"SEIR Simulation error: {e}")
            return None
    
    def calculate_distance(self, sim_data):
        """Calculate MSE distance between simulated and observed data"""
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
        """History matching calibration"""
        print(f"Running SEIR history matching with {n_samples} samples...")
        samples = []
        for _ in range(n_samples):
            sample = {}
            for param, (min_val, max_val) in prior_ranges.items():
                sample[param] = uniform.rvs(loc=min_val, scale=max_val-min_val)
            samples.append(sample)
        results = []
        for sample in tqdm(samples, desc="SEIR History Matching"):
            sim_data = self.run_simulation(sample['beta'], sample['initial_infected'], alpha = alpha)
            distance = self.calculate_distance(sim_data)
            result_dict = {
                "beta": sample["beta"],
                "initial_infected": sample["initial_infected"],
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
        """ABC rejection sampling"""
        if self.hm_results is None or self.hm_results.empty:
            print("Warning: No SEIR history matching results available. Cannot run rejection ABC.")
            return pd.DataFrame()
        print(f"Running SEIR ABC rejection with {n_samples} samples...")
        param_bounds = {
            'beta': (self.hm_results['beta'].min(), self.hm_results['beta'].max()),
            'initial_infected': (self.hm_results['initial_infected'].min(), self.hm_results['initial_infected'].max())
        }
        samples = []
        for _ in range(n_samples):
            hm_idx = np.random.randint(0, len(self.hm_results))
            hm_sample = self.hm_results.iloc[hm_idx]
            sample = {}
            for param in ['beta', 'initial_infected']:
                if param == 'beta':
                    perturb = uniform.rvs(loc=-0.02, scale=0.04)
                else:
                    perturb = uniform.rvs(loc=-2, scale=4) 
                param_min, param_max = param_bounds[param]
                new_value = hm_sample[param] + perturb
                if param == 'initial_infected':
                    new_value = int(np.clip(new_value, param_min, param_max))
                else:
                    new_value = np.clip(new_value, param_min, param_max)
                sample[param] = new_value
            samples.append(sample)
        results = []
        for sample in tqdm(samples, desc="SEIR ABC Rejection"):
            sim_data = self.run_simulation(sample['beta'], sample['initial_infected'])
            distance = self.calculate_distance(sim_data)
            result_dict = {
                "beta": sample["beta"],
                "initial_infected": sample["initial_infected"],
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
        """ABC with simulated annealing"""
        if self.hm_results is None or self.hm_results.empty:
            print("Warning: No SEIR history matching results available. Cannot run annealing ABC.")
            return pd.DataFrame()
        print(f"Running SEIR ABC annealing with {cooling_steps} cooling steps...")
        initial_epsilon = self.hm_results['distance'].quantile(0.5)
        final_epsilon = self.hm_results['distance'].quantile(0.1)
        epsilons = np.geomspace(initial_epsilon, final_epsilon, cooling_steps)
        param_bounds = {
            'beta': (self.hm_results['beta'].min(), self.hm_results['beta'].max()),
            'initial_infected': (self.hm_results['initial_infected'].min(), self.hm_results['initial_infected'].max())
        }
        current_samples = []
        for _ in range(n_samples):
            hm_idx = np.random.randint(0, len(self.hm_results))
            hm_sample = self.hm_results.iloc[hm_idx]
            sample = {param: hm_sample[param] for param in ['beta', 'initial_infected']}
            current_samples.append(sample)
        for step, epsilon in enumerate(epsilons):
            print(f"SEIR Annealing step {step+1}/{cooling_steps}, epsilon = {epsilon:.2f}")
            results = []
            for sample in tqdm(current_samples, desc=f"SEIR Annealing Step {step+1}"):
                sim_data = self.run_simulation(sample['beta'], sample['initial_infected'])
                distance = self.calculate_distance(sim_data)
                result_dict = {
                    "beta": sample["beta"],
                    "initial_infected": sample["initial_infected"],
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
                    for param in ['beta', 'initial_infected']:
                        param_mean = accepted[param].mean()
                        param_std = max(accepted[param].std(), 0.01)
                        new_value = norm.rvs(loc=param_mean, scale=param_std)
                        param_min, param_max = param_bounds[param]
                        if param == 'initial_infected':
                            new_value = int(np.clip(new_value, param_min, param_max))
                        else:
                            new_value = np.clip(new_value, param_min, param_max)
                        sample[param] = new_value
                    new_samples.append(sample)
                current_samples = new_samples
        print(f"SEIR ABC annealing accepted {len(accepted)} parameter sets")
        return accepted
    
    def smc_abc(self, n_particles=50, n_populations=3, accept_ratio=0.1):
        """ABC Sequential Monte Carlo"""
        if self.hm_results is None or self.hm_results.empty:
            print("Warning: No SEIR history matching results available. Cannot run SMC ABC.")
            return pd.DataFrame()
        print(f"Running SEIR ABC-SMC with {n_populations} populations...")
        initial_epsilon = self.hm_results['distance'].quantile(0.7)
        final_epsilon = self.hm_results['distance'].quantile(0.05)
        epsilons = np.geomspace(initial_epsilon, final_epsilon, n_populations)
        param_bounds = {
            'beta': (self.hm_results['beta'].min(), self.hm_results['beta'].max()),
            'initial_infected': (self.hm_results['initial_infected'].min(), self.hm_results['initial_infected'].max())
        }
        particles = []
        for _ in range(n_particles):
            hm_idx = np.random.randint(0, len(self.hm_results))
            hm_sample = self.hm_results.iloc[hm_idx]
            particle = {param: hm_sample[param] for param in ['beta', 'initial_infected']}
            particles.append(particle)
        weights = np.ones(n_particles) / n_particles
        for t in range(n_populations):
            epsilon = epsilons[t]
            print(f"SEIR SMC Population {t+1}/{n_populations}, epsilon = {epsilon:.2f}")
            distances = []
            trajectories = []
            for particle in tqdm(particles, desc=f"SEIR SMC Population {t+1}"):
                sim_data = self.run_simulation(particle['beta'], particle['initial_infected'])
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
                param_values = np.array([[p['beta'], p['initial_infected']] for p in particles])
                cov = np.cov(param_values.T) + np.eye(2) * 1e-6
                new_particles = []
                for particle in particles:
                    attempts = 0
                    while attempts < 50:
                        perturbation = multivariate_normal.rvs(mean=[0, 0], cov=cov)
                        new_particle = {
                            'beta': particle['beta'] + perturbation[0],
                            'initial_infected': particle['initial_infected'] + perturbation[1]
                        }
                        # ensure initial_infected is integer and within bounds
                        new_particle['initial_infected'] = int(new_particle['initial_infected'])
                        beta_valid = param_bounds['beta'][0] <= new_particle['beta'] <= param_bounds['beta'][1]
                        infected_valid = param_bounds['initial_infected'][0] <= new_particle['initial_infected'] <= param_bounds['initial_infected'][1]
                        if beta_valid and infected_valid:
                            new_particles.append(new_particle)
                            break
                        attempts += 1
                    if attempts >= 50:
                        new_particles.append(particle)
                particles = new_particles
        final_results = []
        for i, particle in enumerate(particles):
            final_results.append({
                "beta": particle["beta"],
                "initial_infected": particle["initial_infected"],
                "weight": weights[i],
                "distance": distances[i],
                "trajectory": trajectories[i]
            })
        results_df = pd.DataFrame(final_results)
        print(f"SEIR ABC-SMC completed with {len(results_df)} particles")
        return results_df

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

def get_best_params_safely(results):
    """
    Get best parameters from results DataFrame
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

class ABM_to_SEIR_Framework:
    """
    Complete framework for switching from ABM to SEIR model
    """
    def __init__(self, total_population=10000, switch_fraction=0.01, switch_day=None, data_path="./chelyabinsk_10/"):
        self.total_population = total_population
        self.switch_fraction = switch_fraction
        self.switch_day = switch_day  # supports user-defined switch day
        self.data_path = data_path
        self.abm_data = None
        self.switch_point = None
        self.calibration_results = {}
        self.combined_trajectories = {}
        
    def generate_abm_data(self, alpha=0.78, lmbd=0.4, days=100):
        """
        Generate ABM data using your actual ABM system
        """
        print(f"Generating ABM data with alpha={alpha}, lmbd={lmbd}, days={days}")
        self.abm_data = generate_synthetic_data(
            alpha=alpha, 
            lmbd=lmbd, 
            days=range(1, days+1),
            data_path=self.data_path
        )
        self.abm_data = ensure_day_column(self.abm_data)
        # Debug: print data info
        #print_data_info(self.abm_data, "ABM Data")
        
        return self.abm_data
    
    def find_switch_point(self):
        """
        Determine when to switch from ABM to SEIR model
        Supports both fraction-based and day-based switching
        """
        if self.abm_data is None:
            raise ValueError("One must generate ABM data first")
        cumulative_infected = self.abm_data['H1N1'].cumsum() 
        cumulative_infected_fraction = cumulative_infected / self.total_population
        if self.switch_day is not None:
            # user-defined switch day takes priority
            self.switch_point = min(self.switch_day, len(self.abm_data) - 1)
            print(f"Switching at user-specified day: {self.switch_point}")
        else:
            # switch based on CUMULATIVE infection fraction
            switch_candidates = cumulative_infected_fraction >= self.switch_fraction
            if switch_candidates.any():
                self.switch_point = switch_candidates.idxmax()
                cumulative_at_switch = cumulative_infected_fraction.iloc[self.switch_point]
                print(f"Switching when infection fraction reaches {self.switch_fraction*100:.1f}% at day: {self.switch_point}")
                print(f"Cumulative infected fraction at switch: {cumulative_at_switch*100:.2f}%")
            else:
                # fallback: switch at 2/3 of simulation period
                self.switch_point = len(self.abm_data) * 2 // 3
                print(f"Threshold not reached, switching at day: {self.switch_point}")
        return self.switch_point


def run_complete_framework(alpha=0.78, lmbd=0.4, days=100, switch_fraction=0.01, switch_day=None, data_path="./chelyabinsk_10/"):
    """
    Run the ABM to SEIR framework with plotting
    """
    framework = ABM_to_SEIR_Framework(
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
    seir_observed_data['I'] = seir_observed_data['H1N1']  # Map H1N1 to I compartment
    # Debug: Print SEIR data info
    #print_data_info(seir_observed_data, "SEIR Observed Data")
    # get initial infected count at switch point
    initial_infected_count = int(abm_data.iloc[switch_point]['H1N1'])
    print(f"Initial infected count at switch: {initial_infected_count}")
    
    # Step 4: Run SEIR calibration
    print("\nStep 4: Running SEIR calibration...")
    seir_calibrator = SEIR_Calibrator(seir_observed_data, population=framework.total_population)
    prior_ranges = {
        "beta": (0.001, 0.9),
        "initial_infected": (0.0001, 0.5)
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
        # Continue with available results
    
    # Step 5: Generate trajectories with best parameters
    print("\nStep 5: Generating trajectories with calibrated parameters...")
    combined_trajectories = {}
    best_parameters = {}
    for method, results in calibration_results.items():
        if not results.empty:
            best_params = get_best_params_safely(results)
            if best_params is not None:
                best_parameters[method] = {
                    'beta': best_params['beta'],
                    'initial_infected': best_params['initial_infected'],
                    'distance': best_params['distance']
                }
                seir_sim = seir_calibrator.run_simulation(
                    beta=best_params['beta'], 
                    initial_infected=best_params['initial_infected'],
                    alpha = alpha
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
        ax3 = fig.add_subplot(spec[0, 4:])  # Tau/Beta distribution
        ax4 = fig.add_subplot(spec[1, 1:3]) # Rho/Initial infected distribution
        ax5 = fig.add_subplot(spec[1, 3:5]) # distance comparison
        abm_data = ensure_day_column(abm_data)

        # Plot 1: Complete trajectories (ABM + All SEIR methods) with success indicators
        ax1.plot(abm_data['day'], abm_data['H1N1'], 'k-', linewidth=3, label='ABM Data', alpha=0.8)
        ax1.axvline(switch_point, color='red', linestyle='--', linewidth=2, label=f'Switch at Day {switch_point}')
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
        ax1.set_title('Hybrid: ABM → SEIR', fontsize=14, fontweight='bold')
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
            if not results.empty and 'beta' in results.columns:
                beta_values = results['beta'].dropna()
                if len(beta_values) > 0:
                    success_indicator = '✓' if method in combined_trajectories else '✗'
                    ax3.hist(beta_values, alpha=0.6, bins=min(15, len(beta_values)), 
                            label=f'{success_indicator} {method.title()}', color=colors[i], density=True)
        ax3.set_title('Tau parameter distribution', fontsize=12)
        ax3.set_xlabel('Tau (transmission rate)')
        ax3.set_ylabel('Density')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Parameter distributions (rho) with success indicators
        for i, method in enumerate(method_order):
            results = calibration_results[method]
            if not results.empty and 'initial_infected' in results.columns:
                infected_values = results['initial_infected'].dropna()
                if len(infected_values) > 0:
                    success_indicator = '✓' if method in combined_trajectories else '✗'
                    ax4.hist(infected_values, alpha=0.6, bins=min(15, len(infected_values)), 
                            label=f'{success_indicator} {method.title()}', color=colors[i], density=True)
        ax4.set_title('Rho parameter distribution', fontsize=12)
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
            ax5.set_title('Best Distance by Method', fontsize=12)
            ax5.set_ylabel('Distance (MSE)')
            ax5.tick_params(axis='x', rotation=45)
            for bar, dist in zip(bars, distances):
                if not np.isnan(dist) and not np.isinf(dist):
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(distances)*0.01,
                            f'{dist:.0f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout(pad=3.0)
        plot_filename = f"ABM_to_SEIR_alpha_{alpha}_lmbd_{lmbd}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {plot_filename}")
        plt.show()
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()
    print(f"ABM Parameters: alpha={alpha}, lmbd={lmbd}")
    print(f"Switch point: Day {switch_point}")
    if framework.switch_day is not None:
        print(f"Switch mode: user-defined day ({framework.switch_day})")
    else:
        print(f"Switch mode: infection fraction threshold ({framework.switch_fraction*100:.1f}%)")
    print(f"Post-switch data points: {len(seir_observed_data)}")
    print(f"Initial infected count at switch: {initial_infected_count}")
    for method, params in best_parameters.items():
        print(f"\n{method.upper()} Best Parameters:")
        print(f"  Beta: {params['beta']:.6f}")
        print(f"  Initial infected: {params['initial_infected']}")
        print(f"  Distance: {params['distance']:.2f}")
    return abm_data, calibration_results, switch_point, combined_trajectories, best_parameters

if __name__ == "__main__":
    global alpha 
    alpha = 0.78
    lmbd = 0.4
    prediction_days = 100
    fraction = 0.2
    day = 12
    data_path = "./chelyabinsk_10/"

    # switch at infection fraction threshold
    print(f"Running with infection fraction threshold ({fraction*100:.1f}%)...")
    try:
        results_1 = run_complete_framework(
            alpha=alpha, 
            lmbd=lmbd, 
            days=prediction_days, 
            switch_fraction=fraction,
            switch_day=None,  # use fraction-based switching
            data_path=data_path
        )
        if results_1:
            abm_data, calibration_results, switch_point, combined_trajectories, best_parameters = results_1
            print(f"\nCompleted! Switch occurred at day {switch_point}")
            print(f"Generated {len(combined_trajectories)} complete model trajectories")
    except Exception as e:
        print(f"Error in first run: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "."*80 + "\n")
    
    # switch at specific day
    print(f"Running model with user-defined switch day (day {day})...")
    try:
        results_2 = run_complete_framework(
            alpha=alpha, 
            lmbd=lmbd, 
            days=prediction_days, 
            switch_fraction=fraction,  # this will be ignored
            switch_day=day,  # switch at day X regardless of infection level
            data_path=data_path
        )
        if results_2:
            abm_data_2, calibration_results_2, switch_point_2, combined_trajectories_2, best_parameters_2 = results_2
            print(f"\nCompleted! Switch occurred at day {switch_point_2}")
            print(f"Generated {len(combined_trajectories_2)} complete model trajectories")
    except Exception as e:
        print(f"Error in second run: {e}")
        import traceback
        traceback.print_exc()
