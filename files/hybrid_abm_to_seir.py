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

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

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


class SEIR_Calibrator:
    """
    SEIR model calibrator with ABC methods
    """
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
    
    def run_simulation(self, beta, initial_infected_fraction, alpha=None, t_max=None):
        """
        Run SEIR simulation
        """
        try:
            if alpha is None:
                alpha = self.fixed_alpha
            if t_max is None:
                t_max = len(self.observed_data) - 1
            
            I0 = initial_infected_fraction * self.population
            E0 = 0
            R0 = 0
            S0 = self.population - I0 - E0 - R0
            y0 = [S0, E0, I0, R0]
            
            t = np.linspace(0, t_max, t_max + 1)
            solution = odeint(self.seir_model, y0, t, args=(beta, alpha, self.fixed_gamma))
            
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
                    perturb = uniform.rvs(loc=-0.002, scale=0.004)
                param_min, param_max = param_bounds[param]
                new_value = hm_sample[param] + perturb
                sample[param] = np.clip(new_value, param_min, param_max)
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


class BetaEstimationSEIR:
    """
    SEIR model with time-varying beta estimation using expanding window
    """
    
    def __init__(self, observed_data, population=10000, fixed_alpha=1/3, fixed_gamma=1/5):
        self.observed_data = observed_data
        self.population = population
        self.fixed_alpha = fixed_alpha
        self.fixed_gamma = fixed_gamma
        self.beta_estimates = None
        self.expanding_mean_estimates = None
        
        print(f"Beta estimation SEIR initialized with population {population}")
    
    def estimate_beta_instantaneous(self, window_start=7):
        """
        Estimate instantaneous beta values from observed data
        """
        print("Estimating instantaneous beta values...")
        
        beta_estimates = []
        times = []
        
        for t in range(window_start, len(self.observed_data)):
            if t > 0:
                # Get current state
                I_curr = self.observed_data['I'].iloc[t]
                I_prev = self.observed_data['I'].iloc[t-1]
                
                # Estimate dI/dt
                dI_dt = I_curr - I_prev
                
                # Simple beta estimation: dI/dt = beta*S*I/N - gamma*I
                # Assume S ≈ N for early stages, so beta ≈ (dI/dt + gamma*I) / I
                if I_curr > 0:
                    # Simple estimation
                    beta_est = (dI_dt + self.fixed_gamma * I_curr) / I_curr
                    beta_est = max(0, beta_est)  # ensure non-negative
                else:
                    beta_est = 0.1  # default value
                
                beta_estimates.append(beta_est)
                times.append(t)
        
        self.beta_estimates = pd.DataFrame({
            'time': times,
            'beta_estimate': beta_estimates
        })
        
        return self.beta_estimates
    
    def estimate_beta_expanding_mean(self, window_start=7):
        """
        Estimate beta using expanding window means with actual .expanding(1).mean()
        """
        if self.beta_estimates is None:
            self.estimate_beta_instantaneous(window_start)
        
        print("Applying .expanding(1).mean() to beta estimates...")
        
        # Use actual pandas .expanding(1).mean() method
        beta_series = pd.Series(self.beta_estimates['beta_estimate'].values)
        expanding_means = beta_series.expanding(1).mean()
        
        self.expanding_mean_estimates = pd.DataFrame({
            'time': self.beta_estimates['time'].values,
            'beta_instantaneous': self.beta_estimates['beta_estimate'].values,
            'beta_expanding_mean': expanding_means.values
        })
        
        print(f"Applied .expanding(1).mean() to {len(expanding_means)} beta estimates")
        return self.expanding_mean_estimates
    
    def seir_model_time_varying(self, y, t, beta_func, alpha, gamma):
        """SEIR model with time-varying beta"""
        S, E, I, R = y
        beta_t = beta_func(t)
        
        dSdt = -beta_t * S * I / self.population
        dEdt = beta_t * S * I / self.population - alpha * E
        dIdt = alpha * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    def create_beta_function(self, method='expanding_mean'):
        """
        Create interpolation function for time-varying beta
        """
        if method == 'expanding_mean' and self.expanding_mean_estimates is not None:
            times = self.expanding_mean_estimates['time'].values
            betas = self.expanding_mean_estimates['beta_expanding_mean'].values
        elif self.beta_estimates is not None:
            times = self.beta_estimates['time'].values
            betas = self.beta_estimates['beta_estimate'].values
        else:
            # Fallback to constant beta
            return lambda t: 0.3
        
        # Create interpolation function
        def beta_func(t):
            if t <= times[0]:
                return betas[0]
            elif t >= times[-1]:
                return betas[-1]
            else:
                # Linear interpolation
                idx = np.searchsorted(times, t)
                if idx >= len(times):
                    return betas[-1]
                if idx == 0:
                    return betas[0]
                
                t1, t2 = times[idx-1], times[idx]
                b1, b2 = betas[idx-1], betas[idx]
                return b1 + (b2 - b1) * (t - t1) / (t2 - t1)
        
        return beta_func
    
    def run_time_varying_simulation(self, initial_infected_fraction, method='expanding_mean', t_max=None):
        """
        Run SEIR simulation with time-varying beta
        """
        try:
            if t_max is None:
                t_max = len(self.observed_data) - 1
            
            I0 = initial_infected_fraction * self.population
            E0 = 0
            R0 = 0
            S0 = self.population - I0 - E0 - R0
            y0 = [S0, E0, I0, R0]
            
            t = np.linspace(0, t_max, t_max + 1)
            beta_func = self.create_beta_function(method)
            
            solution = odeint(self.seir_model_time_varying, y0, t, 
                            args=(beta_func, self.fixed_alpha, self.fixed_gamma))
            
            sim_results = pd.DataFrame({
                'day': range(t_max + 1),
                'S': solution[:, 0],
                'E': solution[:, 1],
                'I': solution[:, 2],
                'R': solution[:, 3]
            })
            
            return sim_results
            
        except Exception as e:
            print(f"Time-varying SEIR simulation error: {e}")
            return None
    
    def plot_beta_estimates(self):
        """Plot beta estimates and expanding means"""
        if self.expanding_mean_estimates is None:
            print("No beta estimates available. Run estimate_beta_expanding_mean first.")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Beta estimates comparison
        ax1.plot(self.expanding_mean_estimates['time'], 
                self.expanding_mean_estimates['beta_instantaneous'], 
                'b-', alpha=0.6, linewidth=1, label='Instantaneous Beta')
        ax1.plot(self.expanding_mean_estimates['time'], 
                self.expanding_mean_estimates['beta_expanding_mean'], 
                'r-', linewidth=3, label='Beta expanding mean')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Beta (transmission rate)')
        ax1.set_title('Beta Estimation: Instantaneous vs .expanding(1).mean()')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Infected curve
        ax2.plot(self.observed_data.index, self.observed_data['I'], 
                'k-', linewidth=2, label='Observed Infected')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Number of Infected')
        ax2.set_title('Observed Infected Cases')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Beta vs Infected overlay
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(self.expanding_mean_estimates['time'], 
                        self.expanding_mean_estimates['beta_expanding_mean'], 
                        'r-', linewidth=2, label='Beta expanding mean')
        line2 = ax3_twin.plot(self.observed_data.index, self.observed_data['I'], 
                             'k-', alpha=0.7, label='Infected')
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Beta (transmission rate)', color='r')
        ax3_twin.set_ylabel('Infected Count', color='k')
        ax3.set_title('Beta expanding mean vs Infected Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Plot 4: Model comparison
        if len(self.observed_data) > 0:
            # Run simulation with expanding mean beta
            sim_data = self.run_time_varying_simulation(
                initial_infected_fraction=self.observed_data['I'].iloc[0] / self.population,
                method='expanding_mean'
            )
            
            if sim_data is not None:
                ax4.plot(self.observed_data.index, self.observed_data['I'], 
                        'k-', linewidth=3, label='Observed')
                ax4.plot(sim_data['day'], sim_data['I'], 
                        'r--', linewidth=2, label='SEIR with .expanding(1).mean() Beta')
                ax4.set_xlabel('Time')
                ax4.set_ylabel('Infected')
                ax4.set_title('Model Fit with .expanding(1).mean() Beta')
                ax4.grid(True, alpha=0.3)
                ax4.legend()
        
        plt.tight_layout()
        return fig


class ABM_to_SEIR_Framework:
    """
    Complete framework for switching from ABM to SEIR model with .expanding(1).mean() beta estimation
    """
    def __init__(self, total_population=10000, switch_fraction=0.01, switch_day=None, data_path="./chelyabinsk_10/"):
        self.total_population = total_population
        self.switch_fraction = switch_fraction
        self.switch_day = switch_day
        self.data_path = data_path
        self.abm_data = None
        self.switch_point = None
        self.calibration_results = {}
        self.combined_trajectories = {}
        self.beta_estimation_results = {}
        
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
        
        return self.abm_data
    
    def find_switch_point(self):
        """
        Determine when to switch from ABM to SEIR model
        Supports both fraction-based and day-based switching
        """
        if self.abm_data is None:
            raise ValueError("Must generate ABM data first")
        
        cumulative_infected = self.abm_data['H1N1'].cumsum()
        cumulative_infected_fraction = cumulative_infected / self.total_population
        
        if self.switch_day is not None:
            self.switch_point = min(self.switch_day, len(self.abm_data) - 1)
            print(f"Switching at user-specified day: {self.switch_point}")
        else:
            switch_candidates = cumulative_infected_fraction >= self.switch_fraction
            if switch_candidates.any():
                self.switch_point = switch_candidates.idxmax()
                cumulative_at_switch = cumulative_infected_fraction.iloc[self.switch_point]
                print(f"Switching when infection fraction reaches {self.switch_fraction*100:.1f}% at day: {self.switch_point}")
                print(f"Cumulative infected fraction at switch: {cumulative_at_switch*100:.2f}%")
            else:
                self.switch_point = len(self.abm_data) * 2 // 3
                print(f"Threshold not reached, switching at day: {self.switch_point}")
        
        return self.switch_point
    
    def run_beta_estimation(self, seir_observed_data):
        """
        Run beta estimation on SEIR data using .expanding(1).mean()
        """
        print("Running beta estimation with .expanding(1).mean()...")
        
        beta_estimator = BetaEstimationSEIR(
            seir_observed_data, 
            population=self.total_population
        )
        
        # Estimate instantaneous beta
        beta_estimates = beta_estimator.estimate_beta_instantaneous(window_start=3)
        
        expanding_estimates = beta_estimator.estimate_beta_expanding_mean()
        
        # Run simulation with time-varying beta
        initial_infected_fraction = seir_observed_data['I'].iloc[0] / self.total_population
        time_varying_sim = beta_estimator.run_time_varying_simulation(
            initial_infected_fraction=initial_infected_fraction,
            method='expanding_mean'
        )
        
        self.beta_estimation_results = {
            'estimator': beta_estimator,
            'beta_estimates': beta_estimates,
            'expanding_estimates': expanding_estimates,
            'simulation': time_varying_sim
        }
        
        return self.beta_estimation_results


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
    """Ensure data has a day column"""
    if 'day' not in data.columns:
        data['day'] = range(1, len(data) + 1)
        print("Added 'day' column to data")
    return data


def get_best_params_safely(results):
    """Get best parameters from results DataFrame"""
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


def run_complete_framework(alpha=0.78, lmbd=0.4, days=100, switch_fraction=0.01, switch_day=None, data_path="./chelyabinsk_10/"):
    """
    Run the complete ABM to SEIR framework with .expanding(1).mean() beta estimation
    """
    framework = ABM_to_SEIR_Framework(
        total_population=10000,
        switch_fraction=switch_fraction,
        switch_day=switch_day,
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
            prior_ranges, n_samples=50
        )
        
        print("Running SEIR Rejection ABC...")
        calibration_results['rejection'] = seir_calibrator.rejection_abc(
            n_samples=30
        )
    except Exception as e:
        print(f"Error in SEIR calibration: {e}")
    
    # Step 5: Run beta estimation with .expanding(1).mean()
    print("\nStep 5: Running beta estimation with .expanding(1).mean()...")
    beta_estimation_results = framework.run_beta_estimation(seir_observed_data)
    
    # Step 6: Generate trajectories with best parameters
    print("\nStep 6: Generating trajectories...")
    combined_trajectories = {}
    best_parameters = {}
    
    # Standard SEIR trajectories
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
                    initial_infected_fraction=best_params['initial_infected']
                )
                
                if seir_sim is not None:
                    combined_trajectory = []
                    # ABM part
                    for i in range(switch_point):
                        combined_trajectory.append(abm_data.iloc[i]['H1N1'])
                    # SEIR part
                    for i in range(len(seir_sim)):
                        if switch_point + i < len(abm_data):
                            combined_trajectory.append(seir_sim.iloc[i]['I'])
                    combined_trajectories[method] = combined_trajectory
    
    # Beta estimation trajectory with .expanding(1).mean()
    if beta_estimation_results['simulation'] is not None:
        beta_trajectory = []
        # ABM part
        for i in range(switch_point):
            beta_trajectory.append(abm_data.iloc[i]['H1N1'])
        # Beta estimation SEIR part
        sim_data = beta_estimation_results['simulation']
        for i in range(len(sim_data)):
            if switch_point + i < len(abm_data):
                beta_trajectory.append(sim_data.iloc[i]['I'])
        combined_trajectories['beta_expanding_mean'] = beta_trajectory
    
    # Step 7: Create comprehensive plots
    print("\nStep 7: Creating comprehensive plots...")
    try:
        fig = plt.figure(figsize=(24, 16))
        spec = gridspec.GridSpec(ncols=4, nrows=4, figure=fig)
        
        # Plot 1: Complete trajectories with beta estimation
        ax1 = fig.add_subplot(spec[0, :2])
        abm_data = ensure_day_column(abm_data)
        ax1.plot(abm_data['day'], abm_data['H1N1'], 'k-', linewidth=3, label='ABM Data', alpha=0.8)
        ax1.axvline(switch_point, color='red', linestyle='--', linewidth=2, label=f'Switch at Day {switch_point}')
        
        colors = ['blue', 'green', 'purple', 'orange']
        line_styles = ['-', '--', '-.', ':']
        
        for i, (method, trajectory) in enumerate(combined_trajectories.items()):
            if trajectory and len(trajectory) > 0:
                days_full = list(range(len(trajectory)))
                if method == 'beta_expanding_mean':
                    ax1.plot(days_full, trajectory, color='red', linewidth=4, 
                            linestyle='-', alpha=0.9, label='✓ Beta expanding mean')
                else:
                    ax1.plot(days_full, trajectory, color=colors[i % len(colors)], linewidth=2.5, 
                            linestyle=line_styles[i % len(line_styles)], alpha=0.8, 
                            label=f'✓ SEIR {method.title()}')
        
        ax1.set_title('Hybrid: ABM → SEIR with Beta expanding mean', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Infected')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Beta estimation details
        ax2 = fig.add_subplot(spec[0, 2:])
        if beta_estimation_results['expanding_estimates'] is not None:
            est_data = beta_estimation_results['expanding_estimates']
            ax2.plot(est_data['time'], est_data['beta_instantaneous'], 
                    'b-', alpha=0.6, linewidth=1, label='Instantaneous Beta', marker='o', markersize=3)
            ax2.plot(est_data['time'], est_data['beta_expanding_mean'], 
                    'r-', linewidth=4, label='Beta expanding mean', marker='s', markersize=4)
            ax2.set_xlabel('Time (days after switch)')
            ax2.set_ylabel('Beta (transmission rate)')
            ax2.set_title('Beta Estimation: .expanding(1).mean() Method', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Parameter distributions
        ax3 = fig.add_subplot(spec[1, :2])
        for i, (method, results) in enumerate(calibration_results.items()):
            if not results.empty and 'beta' in results.columns:
                beta_values = results['beta'].dropna()
                if len(beta_values) > 0:
                    ax3.hist(beta_values, alpha=0.6, bins=min(15, len(beta_values)), 
                            label=f'{method.title()}', color=colors[i], density=True)
        ax3.set_title('Beta Parameter Distributions (Standard SEIR)')
        ax3.set_xlabel('Beta (transmission rate)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Switch region zoom
        ax4 = fig.add_subplot(spec[1, 2:])
        zoom_start = max(0, switch_point - 10)
        zoom_end = min(len(abm_data), switch_point + 20)
        if zoom_end > zoom_start:
            ax4.plot(abm_data['day'][zoom_start:zoom_end], abm_data['H1N1'][zoom_start:zoom_end], 
                     'k-', linewidth=3, label='ABM Data')
            ax4.axvline(switch_point, color='red', linestyle='--', linewidth=2, label='Switch point')
            
            # Add beta estimation trajectory in zoom
            if 'beta_expanding_mean' in combined_trajectories:
                trajectory = combined_trajectories['beta_expanding_mean']
                if len(trajectory) > zoom_start:
                    days_zoom = list(range(zoom_start, min(zoom_end, len(trajectory))))
                    traj_zoom = trajectory[zoom_start:min(zoom_end, len(trajectory))]
                    if len(days_zoom) == len(traj_zoom) and len(days_zoom) > 0:
                        ax4.plot(days_zoom, traj_zoom, color='red', linewidth=4, 
                                alpha=0.9, label='Beta expanding mean SEIR')
        
        ax4.set_title('Switch Region Detail')
        ax4.set_xlabel('Day')
        ax4.set_ylabel('Infected')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Beta vs Infected overlay
        ax5 = fig.add_subplot(spec[2, :2])
        ax5_twin = ax5.twinx()
        
        if beta_estimation_results['expanding_estimates'] is not None:
            est_data = beta_estimation_results['expanding_estimates']
            # Adjust time to align with original data
            adjusted_time = est_data['time'] + switch_point
            line1 = ax5.plot(adjusted_time, est_data['beta_expanding_mean'], 
                            'r-', linewidth=3, label='Beta expanding mean', marker='s', markersize=4)
        
        line2 = ax5_twin.plot(abm_data['day'], abm_data['H1N1'], 'k-', alpha=0.7, label='ABM Infected')
        
        ax5.set_xlabel('Day')
        ax5.set_ylabel('Beta (transmission rate)', color='r')
        ax5_twin.set_ylabel('Infected Count', color='k')
        ax5.set_title('Beta expanding mean Evolution vs Epidemic Curve', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Plot 6: Model comparison
        ax6 = fig.add_subplot(spec[2, 2:])
        if beta_estimation_results['simulation'] is not None:
            ax6.plot(seir_observed_data.index, seir_observed_data['I'], 
                    'k-', linewidth=3, label='Observed (post-switch)')
            
            sim_data = beta_estimation_results['simulation']
            ax6.plot(sim_data['day'], sim_data['I'], 
                    'r--', linewidth=3, label='Beta expanding mean Model')
            
            # Add best standard SEIR for comparison
            for method, results in calibration_results.items():
                if not results.empty:
                    best_params = get_best_params_safely(results)
                    if best_params is not None:
                        seir_sim = seir_calibrator.run_simulation(
                            beta=best_params['beta'], 
                            initial_infected_fraction=best_params['initial_infected']
                        )
                        if seir_sim is not None:
                            ax6.plot(seir_sim['day'], seir_sim['I'], 
                                    '--', alpha=0.7, label=f'Best {method.title()}')
                        break
            
            ax6.set_title('Post-Switch Model Comparison')
            ax6.set_xlabel('Time (days after switch)')
            ax6.set_ylabel('Infected')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # Plot 7: Beta statistics with .expanding(1).mean() details
        ax7 = fig.add_subplot(spec[3, :2])
        if beta_estimation_results['expanding_estimates'] is not None:
            est_data = beta_estimation_results['expanding_estimates']
            
            # Show the actual .expanding(1).mean() computation
            ax7.plot(est_data['time'], est_data['beta_expanding_mean'], 'r-', linewidth=3, 
                    label='Beta expanding mean', marker='s', markersize=4)
            
            # Add confidence bands
            rolling_std = pd.Series(est_data['beta_instantaneous']).expanding(1).std()
            ax7.fill_between(est_data['time'], 
                           est_data['beta_expanding_mean'] - rolling_std,
                           est_data['beta_expanding_mean'] + rolling_std,
                           alpha=0.3, color='red', label='±1 Expanding Std')
            
            ax7.set_xlabel('Time (days after switch)')
            ax7.set_ylabel('Beta')
            ax7.set_title('Beta expanding mean with Expanding Standard Deviation', fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # Plot 8: Summary statistics
        ax8 = fig.add_subplot(spec[3, 2:])
        ax8.axis('off')
        
        summary_text = "SUMMARY STATISTICS\n\n"
        summary_text += f"ABM Parameters: alpha={alpha}, lmbd={lmbd}\n"
        summary_text += f"Switch point: Day {switch_point}\n"
        summary_text += f"Post-switch data points: {len(seir_observed_data)}\n"
        summary_text += f"Initial infected at switch: {initial_infected_count}\n\n"
        
        # Beta estimation summary with .expanding(1).mean()
        if beta_estimation_results['expanding_estimates'] is not None:
            est_data = beta_estimation_results['expanding_estimates']
            summary_text += f"Beta Estimation .expanding(1).mean():\n"
            summary_text += f"  Initial β: {est_data['beta_expanding_mean'].iloc[0]:.4f}\n"
            summary_text += f"  Final β: {est_data['beta_expanding_mean'].iloc[-1]:.4f}\n"
            summary_text += f"  Mean β: {est_data['beta_expanding_mean'].mean():.4f}\n"
            summary_text += f"  Std β: {est_data['beta_expanding_mean'].std():.4f}\n"
            summary_text += f"  Min β: {est_data['beta_expanding_mean'].min():.4f}\n"
            summary_text += f"  Max β: {est_data['beta_expanding_mean'].max():.4f}\n\n"
        
        # Standard SEIR summary
        for method, params in best_parameters.items():
            summary_text += f"{method.upper()} Best Parameters:\n"
            summary_text += f"  Beta: {params['beta']:.6f}\n"
            summary_text += f"  Initial infected: {params['initial_infected']:.6f}\n"
            summary_text += f"  Distance: {params['distance']:.2f}\n\n"
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10, 
                 verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout(pad=3.0)
        plot_filename = f"ABM_to_SEIR_Beta_expanding_mean_alpha_{alpha}_lmbd_{lmbd}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {plot_filename}")
        plt.show()
        
        # Create separate beta estimation plot
        if beta_estimation_results['estimator'] is not None:
            beta_fig = beta_estimation_results['estimator'].plot_beta_estimates()
            if beta_fig is not None:
                beta_plot_filename = f"Beta_expanding_mean_Details_alpha_{alpha}_lmbd_{lmbd}.png"
                beta_fig.savefig(beta_plot_filename, dpi=300, bbox_inches='tight')
                print(f"Saved beta estimation plot to: {beta_plot_filename}")
                plt.show()
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()
    
    return abm_data, calibration_results, switch_point, combined_trajectories, best_parameters, beta_estimation_results


if __name__ == "__main__":
    alpha = 0.78
    lmbd = 0.4
    prediction_days = 100
    fraction = 0.02
    day = 25
    data_path = "./chelyabinsk_10/"
    
    # Run with infection fraction threshold
    print(f"\nRunning with infection fraction threshold ({fraction*100:.1f}%)...")
    try:
        results_1 = run_complete_framework(
            alpha=alpha, 
            lmbd=lmbd, 
            days=prediction_days, 
            switch_fraction=fraction,
            switch_day=None,
            data_path=data_path
        )
        
        if results_1:
            abm_data, calibration_results, switch_point, combined_trajectories, best_parameters, beta_results = results_1
            print(f"\n Switch occurred at day {switch_point}")
            print(f"Generated {len(combined_trajectories)} complete model trajectories")
            print(f"Beta expanding mean produced {len(beta_results['expanding_estimates']) if beta_results['expanding_estimates'] is not None else 0} beta estimates")
            
    except Exception as e:
        print(f"Error in first run: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80 + "\n")
    
    # Run with specific switch day
    print(f"Running with user-defined switch day (day {day})...")
    try:
        results_2 = run_complete_framework(
            alpha=alpha, 
            lmbd=lmbd, 
            days=prediction_days, 
            switch_fraction=fraction,
            switch_day=day,
            data_path=data_path
        )
        
        if results_2:
            abm_data_2, calibration_results_2, switch_point_2, combined_trajectories_2, best_parameters_2, beta_results_2 = results_2
            print(f"\n Switch occurred at day {switch_point_2}")
            print(f"Generated {len(combined_trajectories_2)} complete model trajectories")
            print(f"Beta expanding mean produced {len(beta_results_2['expanding_estimates']) if beta_results_2['expanding_estimates'] is not None else 0} beta estimates")
            
    except Exception as e:
        print(f"Error in second run: {e}")
        import traceback
        traceback.print_exc()
