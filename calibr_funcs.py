import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.stats import uniform, norm, multivariate_normal
from main_pool import Main
import multiprocessing as mp
from functools import partial
import seaborn as sns 


def load_data(data_path, frac=1):
    data = pd.read_csv(f'data/{data_path}/people.txt', sep='\t')
    data = data[['sp_id', 'sp_hh_id', 'age', 'sex', 'work_id']]
    data=data.sample(frac=frac)
    
    households = pd.read_csv(f'data/{data_path}/households.txt', sep='\t')
    households = households[['sp_id', 'latitude', 'longitude']]
    dict_school_id = {str(i[0]): list(i[1].index) for i in data[(
        data.age < 18) & (data.work_id != 'X')].groupby('work_id')}
    dict_school_len = [len(dict_school_id[i]) for i in dict_school_id.keys()]
    return data, households, dict_school_id


def preprocess_data(data, households, dict_school_id):
    data[['sp_id', 'sp_hh_id', 'age']] = data[[
        'sp_id', 'sp_hh_id', 'age']].astype(int)
    data[['work_id']] = data[['work_id']].astype(str)
    data = data.sample(frac=1)
    households[['sp_id']] = households[['sp_id']].astype(int)
    households[['latitude', 'longitude']] = households[[
        'latitude', 'longitude']].astype(float)
    households.index = households.sp_id
    return data, households, dict_school_id


# ABC class for parameter estimation
class ABC_Agent:
    """
    Approximate Bayesian Computation for agent-based model parameter estimation
    """
    def __init__(self, observed_data, data_path="chelyabinsk_10/", 
                 days=range(1, 100)):
        """
        Initialize ABC class
        
        Parameters:
        -----------
        observed_data : pandas.DataFrame
            Observed epidemic data
        data_path : str
            Path to the population data files
        days : range
            Range of days to simulate
        """
        self.observed_data = observed_data
        self.data_path = data_path
        self.days = days
        
        # define strain keys
        self.strains_keys = ['H1N1', 'H3N2', 'B']
        
        # prepare data only once
        print("Loading and preprocessing data...")
        self.data, self.households, self.dict_school_id = load_data(data_path)
        self.data, self.households, self.dict_school_id = preprocess_data(self.data, 
                                                                          self.households,
                                                                          self.dict_school_id)
        self.dict_school_len = [len(self.dict_school_id[i]
                                   ) for i in self.dict_school_id.keys()]
        
        # store history matching results
        self.hm_results = None
    
    
    def run_simulation(self, params):
        """
        Run a simulation with given parameters using Main class
        
        Parameters:
        -----------
        params : dict
            Dictionary with model parameters (alpha, lmbd)
        
        Returns:
        --------
        simulation_output : pandas.DataFrame
            Time series of simulated data
        """
        alpha = params['alpha']
        lmbd = params['lmbd']
        
        # create a temporary directory for results
        sim_dir = f"temp_sim_{np.random.randint(0, 100000)}/"
        if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)
        
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
                results_path = os.path.join(pool.results_dir, 
                                            f"prevalence_seed_{run_number}.csv")
                if os.path.exists(results_path):
                    sim_results = pd.read_csv(results_path, sep='\t')
                    sim_results['run'] = run_number
                    all_results.append(sim_results)
    
            combined_results = pd.concat(all_results, ignore_index=True)
            return combined_results
            
        except Exception as e:
            print(f"Simulation error: {e}")
            return None
        finally:
            # clean up temporary directory
            import shutil
            if os.path.exists(sim_dir):
                shutil.rmtree(sim_dir)
    
    
    def calculate_distance(self, sim_data):
        """
        Calculate distance between simulated and observed data
        
        Parameters:
        -----------
        sim_data : pandas.DataFrame
            Simulated data
            
        Returns:
        --------
        distance : float
            Distance metric (MSE)
        """
        if sim_data is None:
            return np.inf
            
        try:
            # use H1N1 strain for comparison
            min_len = min(len(self.observed_data), len(sim_data))
            obs = self.observed_data['H1N1'].values[:min_len]
            sim = sim_data['H1N1'].values[:min_len]
            
            # mean squared error
            distance = np.mean((obs - sim)**2)
            return distance
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return np.inf
    
    
    def history_matching(self, prior_ranges, n_samples=100, epsilon=0.1, 
                         adaptive=True, accept_ratio=0.2):
        """
        Perform history matching to find plausible parameter regions
        """
        print(f"Running history matching with {n_samples} samples...")
    
        # generate samples from prior ranges (which we know from prior knowledge)
        samples = []
        for _ in range(n_samples):
            sample = {}
            for param, (min_val, max_val) in prior_ranges.items():
                sample[param] = uniform.rvs(loc=min_val, scale=max_val-min_val)
            samples.append(sample)
    
        # run simulations and calculate distances (between generated and observed data)
        results = []
        for sample in tqdm(samples):
            sim_data = self.run_simulation(sample)
            distance = self.calculate_distance(sim_data)

            # add trajectory to results dictionary
            trajectory = []
            if sim_data is not None:
                trajectory = sim_data["H1N1"].values.tolist()
                
            # store trajectory data
            result = [sample["alpha"], sample["lmbd"], distance, trajectory]
            results.append(result)

        results_df = pd.DataFrame(results)
        results_df.columns = ['alpha','lmbd','distance','trajectory']
        # print distance statistics for debugging
        print(f"Distance stats: min={results_df['distance'].min()},"+
              f" max={results_df['distance'].max()}, mean={results_df['distance'].mean()}")
    
        # filter results (accept)
        if adaptive:
            n_accept = max(1, int(len(results_df) * accept_ratio))
            accepted = results_df.nsmallest(n_accept, "distance")
        else:
            accepted = results_df[results_df["distance"] < epsilon]
    
        print(f"Accepted {len(accepted)} parameter sets")
        
        accepted.to_csv(f'calibr_output/hm_{n_samples}_samples_{epsilon}_eps.csv',
                        index=False)
        # store results for other methods to use
        self.hm_results = accepted
        
        return accepted

    
    def rejection_abc(self, n_samples=100, epsilon=1e-5):
        """
        Perform ABC rejection sampling based on history matching results
        
        Parameters:
        -----------
        n_samples : int
            Number of parameter samples to evaluate
        epsilon : float
            Threshold for accepting parameter sets
            
        Returns:
        --------
        accepted_params : pandas.DataFrame
            Accepted parameter sets
        """
        if self.hm_results is None:
            raise ValueError("Must run history_matching before rejection_abc")
            
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
            alpha_perturb = uniform.rvs(loc=-0.05, scale=0.1)  # +- 0.05
            lmbd_perturb = uniform.rvs(loc=-0.05, scale=0.1)   # +- 0.05
            
            alpha = np.clip(hm_sample['alpha'] + alpha_perturb, alpha_min, alpha_max)
            lmbd = np.clip(hm_sample['lmbd'] + lmbd_perturb, lmbd_min, lmbd_max)
            
            samples.append({"alpha": alpha, "lmbd": lmbd})
            
        # run simulations and calculate distances
        results = []
        for sample in tqdm(samples):
            sim_data = self.run_simulation(sample)
            distance = self.calculate_distance(sim_data)
            
            # add trajectory to results dictionary
            trajectory = []
            if sim_data is not None:
                trajectory = sim_data["H1N1"].values.tolist()
                
            # store trajectory data
            result = [sample["alpha"], sample["lmbd"], distance, trajectory]
            results.append(result)

            
        results_df = pd.DataFrame(results)
        results_df.columns = ['alpha','lmbd','distance','trajectory']
        
        # select only parameters with distance < epsilon
        accepted = results_df[results_df["distance"] < epsilon]
        if len(accepted) == 0:
            print(f"No samples accepted at epsilon = {epsilon}. Taking best samples.")
            accepted = results_df.nsmallest(max(1, int(n_samples * 0.1)), "distance")
            
        accepted.to_csv(f'calibr_output/rejection_{n_samples}_samples_{epsilon}_eps.csv',
                        index=False)
        return accepted
    
    
    def annealing_abc(self, n_samples=100, initial_epsilon=1e-3, 
                      final_epsilon=1e-5, cooling_steps=3):
        """
        Perform ABC with simulated annealing using history matching results
    
        Parameters:
        -----------
        n_samples : int
            Number of parameter samples per step
        initial_epsilon : float
            Initial acceptance threshold
        final_epsilon : float
            Final acceptance threshold
        cooling_steps : int
            Number of annealing steps
            
        Returns:
        --------
        accepted_params : pandas.DataFrame
            Accepted parameter sets
        """
        if self.hm_results is None:
            raise ValueError("Must run history_matching before annealing_abc")
            
        print(f"Running ABC annealing with {cooling_steps} cooling steps...")
        
        # calculate epsilon values for each step
        epsilons = np.geomspace(initial_epsilon, final_epsilon, cooling_steps)
        
        # get parameter bounds from history matching
        alpha_min = self.hm_results['alpha'].min()
        alpha_max = self.hm_results['alpha'].max()
        lmbd_min = self.hm_results['lmbd'].min()
        lmbd_max = self.hm_results['lmbd'].max()
        
        # initial samples from history matching results
        current_samples = []
        for _ in range(n_samples):
            # randomly select a parameter set from history matching results
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
            for sample in tqdm(current_samples):
                sim_data = self.run_simulation(sample)
                distance = self.calculate_distance(sim_data)
                
                 # add trajectory to results dictionary
                trajectory = []
                if sim_data is not None:
                    trajectory = sim_data["H1N1"].values.tolist()

                # store trajectory data
                result = [sample["alpha"], sample["lmbd"], distance, trajectory]
                results.append(result)

            results_df = pd.DataFrame(results)
            results_df.columns = ['alpha','lmbd','distance','trajectory']
        
            accepted = results_df[results_df["distance"] < epsilon]
            
            if len(accepted) == 0:
                print(f"No samples accepted at epsilon = {epsilon}. Taking best samples.")
                accepted = results_df.nsmallest(max(1, int(n_samples * 0.1)), "distance")
            
            # generate new samples for next iteration
            if step < cooling_steps - 1:
                # calculate mean and variance of accepted parameters
                alpha_mean = accepted["alpha"].mean()
                lmbd_mean = accepted["lmbd"].mean()
                
                # mke absolutely sure the variances are strictly positive
                alpha_var = accepted["alpha"].var()
                if np.isnan(alpha_var) or alpha_var <= 1e-6:
                    alpha_var = 1e-4  # set a safe minimum value
                
                lmbd_var = accepted["lmbd"].var()
                if np.isnan(lmbd_var) or lmbd_var <= 1e-6:
                    lmbd_var = 1e-4  # set a safe minimum value
                
                # generate new samples from normal distribution around accepted values
                current_samples = []
                for _ in range(n_samples):
                    try:
                        alpha = norm.rvs(loc=alpha_mean, scale=np.sqrt(alpha_var))
                        lmbd = norm.rvs(loc=lmbd_mean, scale=np.sqrt(lmbd_var))
                    except ValueError:
                        # fallback if there's still an error
                        perturb_scale = 0.05  # 5% perturbation
                        alpha = alpha_mean + np.random.uniform(-perturb_scale, perturb_scale
                                                              ) * (alpha_max - alpha_min)
                        lmbd = lmbd_mean + np.random.uniform(-perturb_scale, perturb_scale
                                                            ) * (lmbd_max - lmbd_min)
                    
                    # ensure parameters are within bounds from history matching
                    alpha = max(alpha_min, min(alpha, alpha_max))
                    lmbd = max(lmbd_min, min(lmbd, lmbd_max))
                    
                    current_samples.append({
                        "alpha": alpha,
                        "lmbd": lmbd
                    })
                    
        accepted.to_csv(f'calibr_output/annealing_{n_samples}_samples_{epsilon}_eps.csv',
                        index=False)
        # return final accepted samples
        return accepted

    
    def smc_abc(self, n_particles=100, n_populations=3,
                initial_epsilon=1e-3, final_epsilon=1e-5):
        """
        Perform ABC Sequential Monte Carlo using history matching results
        
        Parameters:
        -----------
        n_particles : int
            Number of particles
        n_populations : int
            Number of SMC iterations
        initial_epsilon : float
            Initial acceptance threshold
        final_epsilon : float
            Final acceptance threshold
            
        Returns:
        --------
        posterior : pandas.DataFrame
            Posterior distribution with weights
        """
        if self.hm_results is None:
            raise ValueError("Must run history_matching before smc_abc")
            
        print(f"Running ABC-SMC with {n_populations} populations...")
        
        # calculate epsilon sequence
        epsilons = np.geomspace(initial_epsilon, final_epsilon, n_populations)
        
        # get parameter bounds from history matching
        alpha_min = self.hm_results['alpha'].min()
        alpha_max = self.hm_results['alpha'].max()
        lmbd_min = self.hm_results['lmbd'].min()
        lmbd_max = self.hm_results['lmbd'].max()
        
        # initialize first population from history matching results
        particles = []
        for _ in range(n_particles):
            # randomly select a parameter set from history matching
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
            for particle in tqdm(particles):
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
                # take best 10% particles
                sorted_indices = np.argsort(distances)
                for i in range(max(1, int(n_particles * 0.1))):
                    new_weights[sorted_indices[i]] = 1.0
                new_weights = new_weights / np.sum(new_weights)
            
            # calculate effective sample size
            ESS = 1.0 / np.sum(new_weights**2)
            print(f"Effective Sample Size: {ESS:.2f}")
            
            # resample if needed
            if ESS < n_particles / 2 or t == n_populations - 1:
                # resample based on weights
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
                cov = np.cov(params.T) + np.eye(2) * 1e-6  # add small diagonal for stability
                
                # perturb particles
                new_particles = []
                for i, particle in enumerate(particles):
                    accepted = False
                    attempts = 0
                    while not accepted and attempts < 100:
                        attempts += 1
                        # multivariate normal perturbation
                        perturbation = multivariate_normal.rvs(mean=[0, 0], cov=cov)
                        new_alpha = particle["alpha"] + perturbation[0]
                        new_lmbd = particle["lmbd"] + perturbation[1]
                        
                        # check if within history matching bounds
                        alpha_in_bounds = alpha_min <= new_alpha <= alpha_max
                        lmbd_in_bounds = lmbd_min <= new_lmbd <= lmbd_max
                        
                        if alpha_in_bounds and lmbd_in_bounds:
                            accepted = True
                            new_particles.append({"alpha": new_alpha, "lmbd": new_lmbd})
                    
                    # if couldn't generate valid particle after max attempts, keep original
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
            
        fin = pd.DataFrame(final_results)
        fin.to_csv(f'calibr_output/smc_{n_particles}_particles_{final_epsilon}_eps.csv',
                        index=False)
        return fin

    
# function for generating synthetic data and plotting
def generate_synthetic_data(alpha=0.78, lmbd=0.4, days=range(1, 100), 
                            data_folder='chelyabinsk_0.3_sampled_data/',
                            with_seirb=True, with_switch=True):
    """
    Generate synthetic data with the actual agent-based model.
    The synthetic data generated should be created in one sample.
    """
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
        data_folder=data_folder
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


def plot_results(observed_data, abc_results, 
                 method_name='', n_trajectories=5,
                 true_alpha=0, true_lmbd=0, hm_results=[]):
    """
    Plot parameter posterior and time series comparison
    """
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    if len(abc_results) == 0:
        axes[0].text(0.5, 0.5, f"No accepted parameter sets for {method_name}", 
                    horizontalalignment='center', verticalalignment='center')
        axes[1].text(0.5, 0.5, f"No accepted parameter sets for {method_name}",
                    horizontalalignment='center', verticalalignment='center')
    else:
        # ____ Accepted parameters ____
        if 'weight' in abc_results.columns:
            s=abc_results["weight"]*100
        else:
            s=40
        scatter = axes[0].scatter(abc_results["alpha"], abc_results["lmbd"], 
                                  alpha=0.6, s=s, label='Accepted')    
        # 'true' parameters    
        axes[0].scatter(true_alpha, true_lmbd, alpha=0.6, 
                        color='black', label='Observed', s=50)

        axes[0].set_title(f"Accepted parameters - {method_name}")
        axes[0].set_xlabel("Alpha")
        axes[0].set_ylabel("Lambda")
        
        if len(hm_results):
            axes[0].set_xlim(hm_results['alpha'].quantile(.2),
                            hm_results['alpha'].quantile(.8))
            axes[0].set_ylim(hm_results['lmbd'].quantile(.2),
                            hm_results['lmbd'].quantile(.8))
        axes[0].legend()
        axes[0].grid()

        # ____ Trajectories from accepted parameters ____
        n_plot = min(n_trajectories, len(abc_results))
        for i in range(n_plot):
            traj = abc_results.iloc[i]["trajectory"]
            if i==0:
                label='Accepted trajectory'
            else:
                label=''
            axes[1].plot(traj, alpha=0.4, label=label)

        # plot time series (of observed)
        axes[1].plot(#np.tile(observed_data["H1N1"], 5), 
                    observed_data["H1N1"],
                     label="Observed", color="black", linestyle="--")
        axes[1].set_title("Time series comparison")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Infected")
        axes[1].legend()
        axes[1].grid()

        stop = observed_data[observed_data.H1N1==0].index
        if stop.shape[0]:
            axes[1].set_xlim(-5, stop[0]+10)
        else:
            axes[1].set_xlim(-5, observed_data.shape[0]+10)

        # ____ Posterior destribution of parameter 1 ____
        plt.style.use("default")
        axes[2].hist(abc_results['alpha'], alpha=0.4, color='gray')
        sns.kdeplot(abc_results['alpha'], color="tab:blue", 
                    shade=True, ax=axes[2])
        axes[2].axvline(true_alpha, ls='--', color='black',
                        label='Observed')
        axes[2].set_title("Alpha, posterior destribution")
        axes[2].set_xlabel('Alpha')
        axes[2].legend()
        axes[2].grid()

        # ____ Posterior destribution of parameter 2 ____
        plt.style.use("default")
        axes[3].hist(abc_results['lmbd'], alpha=0.4, color='gray')
        sns.kdeplot(abc_results['lmbd'], color="tab:blue", 
                    shade=True, ax=axes[3])
        axes[3].axvline(true_lmbd, ls='--', color='black',
                        label='Observed')
        axes[3].set_title("Lambda, posterior destribution")
        axes[3].set_xlabel('Lambda')
        axes[3].legend()
        axes[3].grid()

    plt.tight_layout()
