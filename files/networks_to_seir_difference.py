import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random
from tqdm import tqdm
from scipy.stats import uniform, norm, multivariate_normal
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)

class StandardSEIR:
    """
    Difference SEIR model
    """
    
    def __init__(self, observed_data, fixed_alpha=0.2, fixed_gamma=0.1):
        self.observed_data = observed_data
        self.fixed_alpha = fixed_alpha
        self.fixed_gamma = fixed_gamma
        self.hm_results = None
        
        print(f"Fixed parameters: alpha = {self.fixed_alpha}, gamma = {self.fixed_gamma}")
        print("Calibrating parameters: beta (transmission rate), rho (initial infection fraction)")
    
    def seir_ode(self, y, t, beta, alpha, gamma):
        """SEIR differential equations"""
        S, E, I, R = y
        
        dS = -beta * S * I
        dE = beta * S * I - alpha * E
        dI = alpha * E - gamma * I
        dR = gamma * I
        
        return [dS, dE, dI, dR]
    
    def simulator_function(self, beta, rho):
        """Run Differential SEIR simulation"""
        try:
            N = 1000  # total population TODO: change to the actual population
            tmax = len(self.observed_data) - 1
            t = np.linspace(0, tmax, tmax + 1)
            
            # initial conditions
            I0 = int(rho * N)
            E0 = 0
            R0 = 0
            S0 = N - I0 - E0 - R0
            
            y0 = [S0, E0, I0, R0]
            
            # solve ODE
            sol = odeint(self.seir_ode, y0, t, args=(beta, self.fixed_alpha, self.fixed_gamma))
            
            sim_results = pd.DataFrame({
                'day': range(len(sol)),
                'S': sol[:, 0],
                'E': sol[:, 1],
                'I': sol[:, 2],
                'R': sol[:, 3]
            })
            
            return sim_results
            
        except Exception as e:
            print(f"Simulation error: {e}")
            return None
    
    def calculate_distance(self, sim_data):
        """Calculate distance between simulated and observed data"""
        if sim_data is None:
            return np.inf
        
        try:
            min_len = min(len(self.observed_data), len(sim_data))
            obs = self.observed_data['I'].values[:min_len]
            sim = sim_data['I'].values[:min_len]
            
            # mean squared error
            distance = np.mean((obs - sim)**2)
            return distance
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return np.inf
    
    def history_matching(self, prior_ranges, n_samples=100, epsilon=1000, adaptive=False, accept_ratio=0.2):
        """History matching for beta and rho"""
        print(f"Running DIFFERENTIAL SEIR history matching with {n_samples} samples...")
        
        samples = []
        for _ in range(n_samples):
            sample = {}
            for param, (min_val, max_val) in prior_ranges.items():
                if param in ['beta', 'rho']:
                    sample[param] = uniform.rvs(loc=min_val, scale=max_val-min_val)
            samples.append(sample)
        
        results = []
        for sample in tqdm(samples):
            sim_data = self.simulator_function(sample["beta"], sample["rho"])
            distance = self.calculate_distance(sim_data)
            
            result_dict = {
                "beta": sample["beta"],
                "rho": sample["rho"],
                "alpha": self.fixed_alpha,
                "gamma": self.fixed_gamma,
                "distance": distance
            }
            
            if sim_data is not None:
                result_dict["trajectory"] = sim_data["I"].copy()
            
            results.append(result_dict)
        
        results_df = pd.DataFrame(results)
        
        if adaptive:
            n_accept = max(1, int(len(results_df) * accept_ratio))
            accepted = results_df.nsmallest(n_accept, "distance")
        else:
            accepted = results_df[results_df["distance"] < epsilon]
        
        print(f"Accepted {len(accepted)} parameter sets")
        self.hm_results = accepted
        return accepted
    
    def plot_results(self, results_df, method_name="Differential SEIR", n_trajectories=5):
        """Plot results for SEIR"""
        if len(results_df) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"No accepted parameter sets for {method_name}", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        fig = plt.figure(figsize=(15, 10))
        
        # 2D Parameter posterior plot (beta vs rho)
        ax1 = fig.add_subplot(2, 2, 1)
        scatter = ax1.scatter(results_df["beta"], results_df["rho"],
                            c=results_df["distance"], cmap='viridis_r', alpha=0.7, s=80,
                            edgecolors='black', linewidth=0.5)
        cbar = fig.colorbar(scatter, ax=ax1)
        cbar.set_label('Distance')
        ax1.set_xlabel('Beta (transmission rate)')
        ax1.set_ylabel('Rho (initial infection fraction)')
        ax1.set_title(f'{method_name} Parameter Posterior\n({len(results_df)} points)')
        ax1.grid(True, alpha=0.3)
        
        # Beta distribution
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.hist(results_df["beta"], bins=15, alpha=0.7, density=True, 
                edgecolor='black', color='blue')
        ax2.set_title("Beta distribution")
        ax2.set_xlabel("Beta (transmission rate)")
        ax2.set_ylabel("Density")
        ax2.grid(True, alpha=0.3)
        
        # Rho distribution
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.hist(results_df["rho"], bins=15, alpha=0.7, density=True, 
                edgecolor='black', color='red')
        ax3.set_title("Rho distribution")
        ax3.set_xlabel("Rho (initial infection fraction)")
        ax3.set_ylabel("Density")
        ax3.grid(True, alpha=0.3)
        
        # Time series comparison
        ax4 = fig.add_subplot(2, 2, 4)
        n_plot = min(n_trajectories, len(results_df))
        if n_plot > 0:
            sorted_indices = results_df['distance'].nsmallest(n_plot).index
            for i, idx in enumerate(sorted_indices):
                traj = results_df.loc[idx].get("trajectory")
                if traj is not None:
                    alpha_val = 0.8 if i < 3 else 0.4
                    ax4.plot(traj, alpha=alpha_val, label=f"Sim {i+1}", linewidth=1.5)
        
        ax4.plot(self.observed_data["I"], color="black", linestyle="--", 
                linewidth=3, label="Observed")
        ax4.set_title("Time series comparison")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Infected")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class NetworkSEIR_tuned:
    """
    Network-based SEIR model parameter estimation class
    """
    
    def __init__(self, observed_data, network_params=None, fixed_alpha=0.2, fixed_gamma=0.1):
        """Initialization with fixed alpha and gamma"""
        self.observed_data = observed_data
        
        # fixed parameters
        self.fixed_alpha = fixed_alpha 
        self.fixed_gamma = fixed_gamma 
        
        # default network parameters
        if network_params is None:
            network_params = {
                'n_nodes': 1000,
                'network_type': 'barabasi_albert',
                'network_params': {'m': 3}
            }
        self.network_params = network_params
        
        # store history matching results
        self.hm_results = None
        
        print(f"Network SEIR - Fixed parameters: alpha = {self.fixed_alpha}, gamma = {self.fixed_gamma}")
        print("Calibrating parameters: tau (transmission rate), rho (initial infection fraction)")
    
    def generate_network(self):
        """Generate network based on specified parameters"""
        n_nodes = self.network_params['n_nodes']
        network_type = self.network_params['network_type']
        net_params = self.network_params['network_params']
        
        if network_type == 'barabasi_albert':
            return nx.barabasi_albert_graph(n_nodes, net_params['m'])
        elif network_type == 'erdos_renyi':
            return nx.erdos_renyi_graph(n_nodes, net_params['p'])
        elif network_type == 'watts_strogatz':
            return nx.watts_strogatz_graph(n_nodes, net_params['k'], net_params['p'])
        elif network_type == 'complete':
            return nx.complete_graph(n_nodes)
        elif network_type == 'regular':
            return nx.random_regular_graph(net_params['d'], n_nodes)
        else:
            raise ValueError(f"Unknown network type: {network_type}")
    
    def SEIR_network(self, G, tau, alpha, gamma, rho, tmax):
        """SEIR network simulation"""
        # initialize states: S=0, E=1, I=2, R=3
        for node in G.nodes():
            G.nodes[node]['state'] = 0
        
        initial_infected = int(rho * len(G.nodes()))
        initial_infected_nodes = random.sample(list(G.nodes()), initial_infected)
        for node in initial_infected_nodes:
            G.nodes[node]['state'] = 2
        
        susceptible_count = []
        exposed_count = []
        infected_count = []
        recovered_count = []
        
        for day in range(tmax + 1):
            new_states = {}
            
            # Count current states
            susceptible = sum(1 for n in G.nodes if G.nodes[n]['state'] == 0)
            exposed = sum(1 for n in G.nodes if G.nodes[n]['state'] == 1)
            infected = sum(1 for n in G.nodes if G.nodes[n]['state'] == 2)
            recovered = sum(1 for n in G.nodes if G.nodes[n]['state'] == 3)
            
            susceptible_count.append(susceptible)
            exposed_count.append(exposed)
            infected_count.append(infected)
            recovered_count.append(recovered)
            
            for node in G.nodes():
                if G.nodes[node]['state'] == 2:  # infected node
                    for neighbor in G.neighbors(node):
                        if G.nodes[neighbor]['state'] == 0:  # susceptible neighbor
                            if random.random() < tau:
                                new_states[neighbor] = 1  # expose neighbor
                    if random.random() < gamma:
                        new_states[node] = 3  # recover
                
                elif G.nodes[node]['state'] == 1:  # exposed node
                    if random.random() < alpha:
                        new_states[node] = 2  # become infected
            
            # apply state transitions
            for node, new_state in new_states.items():
                G.nodes[node]['state'] = new_state
        
        return [index for index in range(tmax + 1)], susceptible_count, exposed_count, infected_count, recovered_count
    
    def simulator_function(self, tau, rho):
        """Run SEIR network simulation with given tau, rho and fixed alpha, gamma"""
        try:
            G = self.generate_network()
            tmax = len(self.observed_data) - 1
            
            days, S, E, I, R = self.SEIR_network(G, tau, self.fixed_alpha, self.fixed_gamma, rho, tmax)
            
            sim_results = pd.DataFrame({
                'day': days,
                'S': S,
                'E': E,
                'I': I,
                'R': R
            })
            
            return sim_results
            
        except Exception as e:
            print(f"Simulation error: {e}")
            return None
    
    def calculate_distance(self, sim_data):
        """Calculate distance between simulated and observed data"""
        if sim_data is None:
            return np.inf
        
        try:
            min_len = min(len(self.observed_data), len(sim_data))
            obs = self.observed_data['I'].values[:min_len]
            sim = sim_data['I'].values[:min_len]
            
            distance = np.mean((obs - sim)**2)
            return distance
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return np.inf
    
    def history_matching(self, prior_ranges, n_samples=100, epsilon=1000, adaptive=False, accept_ratio=0.2):
        """History matching to find plausible parameter regions for tau and rho only"""
        print(f"Running Network SEIR history matching with {n_samples} samples...")
        
        samples = []
        for _ in range(n_samples):
            sample = {}
            for param, (min_val, max_val) in prior_ranges.items():
                if param in ['tau', 'rho']:
                    sample[param] = uniform.rvs(loc=min_val, scale=max_val-min_val)
            samples.append(sample)
        
        results = []
        for sample in tqdm(samples):
            sim_data = self.simulator_function(sample["tau"], sample["rho"])
            distance = self.calculate_distance(sim_data)
            
            result_dict = {
                "tau": sample["tau"],
                "rho": sample["rho"],
                "alpha": self.fixed_alpha,
                "gamma": self.fixed_gamma,
                "distance": distance
            }
            
            if sim_data is not None:
                result_dict["trajectory"] = sim_data["I"].copy()
            
            results.append(result_dict)
        
        results_df = pd.DataFrame(results)
        
        if adaptive:
            n_accept = max(1, int(len(results_df) * accept_ratio))
            accepted = results_df.nsmallest(n_accept, "distance")
        else:
            accepted = results_df[results_df["distance"] < epsilon]
        
        print(f"Accepted {len(accepted)} parameter sets")
        self.hm_results = accepted
        return accepted
    
    def plot_results(self, results_df, method_name="Network SEIR", n_trajectories=5):
        """Plot parameter posterior for tau and rho with time series comparison"""
        if len(results_df) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"No accepted parameter sets for {method_name}", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        fig = plt.figure(figsize=(15, 10))
        
        # 2D Parameter posterior plot (tau vs rho)
        ax1 = fig.add_subplot(2, 2, 1)
        scatter = ax1.scatter(results_df["tau"], results_df["rho"],
                            c=results_df["distance"], cmap='viridis_r', alpha=0.7, s=80,
                            edgecolors='black', linewidth=0.5)
        cbar = fig.colorbar(scatter, ax=ax1)
        cbar.set_label('Distance')
        ax1.set_xlabel('Tau (transmission rate)')
        ax1.set_ylabel('Rho (initial infection fraction)')
        ax1.set_title(f'{method_name} Parameter Posterior\n({len(results_df)} points)')
        ax1.grid(True, alpha=0.3)
        
        # Tau distribution
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.hist(results_df["tau"], bins=15, alpha=0.7, density=True, 
                edgecolor='black', color='blue')
        ax2.set_title("Tau distribution")
        ax2.set_xlabel("Tau (transmission rate)")
        ax2.set_ylabel("Density")
        ax2.grid(True, alpha=0.3)
        
        # Rho distribution
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.hist(results_df["rho"], bins=15, alpha=0.7, density=True, 
                edgecolor='black', color='red')
        ax3.set_title("Rho distribution")
        ax3.set_xlabel("Rho (initial infection fraction)")
        ax3.set_ylabel("Density")
        ax3.grid(True, alpha=0.3)
        
        # Time series comparison
        ax4 = fig.add_subplot(2, 2, 4)
        n_plot = min(n_trajectories, len(results_df))
        if n_plot > 0:
            sorted_indices = results_df['distance'].nsmallest(n_plot).index
            for i, idx in enumerate(sorted_indices):
                traj = results_df.loc[idx].get("trajectory")
                if traj is not None:
                    alpha_val = 0.8 if i < 3 else 0.4
                    ax4.plot(traj, alpha=alpha_val, label=f"Sim {i+1}", linewidth=1.5)
        
        ax4.plot(self.observed_data["I"], color="black", linestyle="--", 
                linewidth=3, label="Observed")
        ax4.set_title("Time series comparison")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Infected")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class BetaEstimationSEIR:
    """
    SEIR model with time-varying beta estimation using expanding window
    """
    
    def __init__(self, observed_data, model_type='standard', network_params=None):
        self.observed_data = observed_data
        self.model_type = model_type
        self.network_params = network_params
        self.beta_estimates = None
        
        print(f"Beta estimation SEIR ({model_type}) initialized")
    
    def estimate_beta_expanding(self, window_start=7):
        """
        Estimate time-varying beta using expanding window approach
        """
        print("Estimating time-varying beta with expanding windows...")
        
        beta_estimates = []
        times = []
        
        for t in range(window_start, len(self.observed_data)):
            # Use expanding window from start to current time
            window_data = self.observed_data.iloc[:t+1].copy()
            
            if t > 0 and window_data['I'].iloc[-1] > 0:
                dI = window_data['I'].iloc[-1] - window_data['I'].iloc[-2]
                I_curr = window_data['I'].iloc[-1]
                
                if I_curr > 0:
                    growth_rate = dI / I_curr
                    if len(beta_estimates) == 0:
                        beta_est = max(0, growth_rate + 0.1)  # add gamma approximation
                    else:
                        # Use expanding mean
                        recent_estimates = beta_estimates.copy()
                        recent_estimates.append(max(0, growth_rate + 0.1))
                        beta_est = np.mean(recent_estimates)
                else:
                    beta_est = 0.1 if len(beta_estimates) == 0 else beta_estimates[-1]
            else:
                beta_est = 0.1 if len(beta_estimates) == 0 else beta_estimates[-1]
            
            beta_estimates.append(beta_est)
            times.append(t)
        
        self.beta_estimates = pd.DataFrame({
            'time': times,
            'beta_estimate': beta_estimates
        })
        
        return self.beta_estimates
    
    def plot_beta_estimates(self):
        """Plot time-varying beta estimates"""
        if self.beta_estimates is None:
            print("No beta estimates available. Run estimate_beta_expanding first.")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot beta estimates
        ax1.plot(self.beta_estimates['time'], self.beta_estimates['beta_estimate'], 
                'b-', linewidth=2, label='Beta estimate')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Beta (transmission rate)')
        ax1.set_title('Time-varying Beta Estimation (Expanding Window)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot infected curve
        ax2.plot(self.observed_data.index, self.observed_data['I'], 
                'r-', linewidth=2, label='Infected')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Number of Infected')
        ax2.set_title('Observed Infected')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig


def generate_synthetic_seir_data(tau=0.3, alpha=0.2, gamma=0.1, rho=0.01, tmax=99, network_params=None, model_type='network'):
    """
    Generate synthetic SEIR epidemic data with known parameters
    """
    if model_type == 'network':
        if network_params is None:
            network_params = {
                'n_nodes': 1000,
                'network_type': 'barabasi_albert',
                'network_params': {'m': 3}
            }
        
        dummy_seir = NetworkSEIR_tuned(pd.DataFrame({'I': [0]}), network_params, 
                                       fixed_alpha=alpha, fixed_gamma=gamma)
        G = dummy_seir.generate_network()
        days, S, E, I, R = dummy_seir.SEIR_network(G, tau, alpha, gamma, rho, tmax)
        
    else:  # standard ODE model
        dummy_seir = StandardSEIR(pd.DataFrame({'I': [0]}), fixed_alpha=alpha, fixed_gamma=gamma)
        
        N = 1000
        t = np.linspace(0, tmax, tmax + 1)
        I0 = int(rho * N)
        E0 = 0
        R0 = 0
        S0 = N - I0 - E0 - R0
        y0 = [S0, E0, I0, R0]
        
        sol = odeint(dummy_seir.seir_ode, y0, t, args=(tau, alpha, gamma))
        days = list(range(len(sol)))
        S, E, I, R = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]
    
    data = pd.DataFrame({
        'day': days,
        'S': S,
        'E': E,
        'I': I,
        'R': R
    })
    
    return data


def generate_real_data_from_seed(seed=0, tau=0.3, alpha=0.2, gamma=0.1, rho=0.01, tmax=99):
    print(f"Generating real data with seed {seed}")
    
    # Set specific seed for this data generation
    np.random.seed(seed)
    random.seed(seed)
    
    network_params = {
        'n_nodes': 1000,
        'network_type': 'barabasi_albert',
        'network_params': {'m': 3}
    }
    
    real_data = generate_synthetic_seir_data(
        tau=tau, alpha=alpha, gamma=gamma, rho=rho, 
        tmax=tmax, network_params=network_params, model_type='network'
    )
    
    # Reset seeds
    np.random.seed(42)
    random.seed(42)
    
    return real_data


def compare_models(real_data, n_samples=200):
    print("=== COMPARING SEIR MODELS ===")
    
    # parameter ranges
    network_prior_ranges = {
        "tau": (0.1, 0.9),
        "rho": (0.001, 0.05)
    }
    
    standard_prior_ranges = {
        "beta": (0.1, 0.9),
        "rho": (0.001, 0.05)
    }
    
    network_params = {
        'n_nodes': 1000,
        'network_type': 'barabasi_albert',
        'network_params': {'m': 3}
    }
    
    # 1. Network SEIR
    print("\n1. Running Network SEIR...")
    network_seir = NetworkSEIR_tuned(real_data, network_params, fixed_alpha=0.2, fixed_gamma=0.1)
    network_results = network_seir.history_matching(network_prior_ranges, n_samples=n_samples, adaptive=True, accept_ratio=0.1)
    
    # 2. Differential SEIR
    print("\n2. Running Differential SEIR...")
    standard_seir = StandardSEIR(real_data, fixed_alpha=0.2, fixed_gamma=0.1)
    standard_results = standard_seir.history_matching(standard_prior_ranges, n_samples=n_samples, adaptive=True, accept_ratio=0.1)
    
    # 3. Beta estimation
    print("\n3. Running Beta Estimation...")
    beta_estimator = BetaEstimationSEIR(real_data, model_type='standard')
    beta_estimates = beta_estimator.estimate_beta_expanding(window_start=7)
    
    return {
        'network_seir': network_seir,
        'network_results': network_results,
        'standard_seir': standard_seir,
        'standard_results': standard_results,
        'beta_estimator': beta_estimator,
        'beta_estimates': beta_estimates
    }


def plot_model_comparison(comparison_results, real_data):
    fig = plt.figure(figsize=(20, 15))
    
    # Extract results
    network_results = comparison_results['network_results']
    standard_results = comparison_results['standard_results']
    beta_estimates = comparison_results['beta_estimates']
    
    # 1. Parameter posteriors comparison
    ax1 = fig.add_subplot(3, 4, 1)
    if len(network_results) > 0:
        ax1.scatter(network_results["tau"], network_results["rho"], 
                   alpha=0.6, s=50, c='blue', label='Network SEIR')
    ax1.set_xlabel('Transmission Rate')
    ax1.set_ylabel('Initial Infection Fraction')
    ax1.set_title('Network SEIR Parameters')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = fig.add_subplot(3, 4, 2)
    if len(standard_results) > 0:
        ax2.scatter(standard_results["beta"], standard_results["rho"], 
                   alpha=0.6, s=50, c='red', label='Differential SEIR')
    ax2.set_xlabel('Transmission Rate')
    ax2.set_ylabel('Initial Infection Fraction')
    ax2.set_title('Differential SEIR Parameters')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 2. Time series comparison
    ax3 = fig.add_subplot(3, 4, 3)
    ax3.plot(real_data['I'], 'k-', linewidth=3, label='Real Data')
    
    # Plot best trajectories from each method
    if len(network_results) > 0:
        best_network_idx = network_results['distance'].idxmin()
        best_network_traj = network_results.loc[best_network_idx, 'trajectory']
        if best_network_traj is not None:
            ax3.plot(best_network_traj, 'b--', linewidth=2, alpha=0.7, label='Best Network SEIR')
    
    if len(standard_results) > 0:
        best_standard_idx = standard_results['distance'].idxmin()
        best_standard_traj = standard_results.loc[best_standard_idx, 'trajectory']
        if best_standard_traj is not None:
            ax3.plot(best_standard_traj, 'r:', linewidth=2, alpha=0.7, label='Best Differential SEIR')
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Infected')
    ax3.set_title('Best Fit Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3. Beta estimates over time
    ax4 = fig.add_subplot(3, 4, 4)
    if beta_estimates is not None and len(beta_estimates) > 0:
        ax4.plot(beta_estimates['time'], beta_estimates['beta_estimate'], 
                'g-', linewidth=2, label='Beta Estimates')
        ax4.axhline(y=0.3, color='orange', linestyle='--', linewidth=2, label='True Beta (0.3)')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Beta Estimate')
    ax4.set_title('Time-varying Beta Estimation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 4. Distance distributions
    ax5 = fig.add_subplot(3, 4, 5)
    if len(network_results) > 0:
        ax5.hist(network_results['distance'], bins=15, alpha=0.6, label='Network SEIR', color='blue')
    if len(standard_results) > 0:
        ax5.hist(standard_results['distance'], bins=15, alpha=0.6, label='Differential SEIR', color='red')
    ax5.set_xlabel('Distance')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distance Distributions')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 5. Parameter distributions - Transmission rates
    ax6 = fig.add_subplot(3, 4, 6)
    if len(network_results) > 0:
        ax6.hist(network_results['tau'], bins=15, alpha=0.6, label='Network τ', color='blue')
    if len(standard_results) > 0:
        ax6.hist(standard_results['beta'], bins=15, alpha=0.6, label='Standard β', color='red')
    ax6.axvline(x=0.3, color='orange', linestyle='--', linewidth=2, label='True Value (0.3)')
    ax6.set_xlabel('Transmission Rate')
    ax6.set_ylabel('Density')
    ax6.set_title('Transmission Rate Distributions')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 6. Parameter distributions - Initial infection
    ax7 = fig.add_subplot(3, 4, 7)
    if len(network_results) > 0:
        ax7.hist(network_results['rho'], bins=15, alpha=0.6, label='Network ρ', color='blue')
    if len(standard_results) > 0:
        ax7.hist(standard_results['rho'], bins=15, alpha=0.6, label='Standard ρ', color='red')
    ax7.axvline(x=0.01, color='orange', linestyle='--', linewidth=2, label='True Value (0.01)')
    ax7.set_xlabel('Initial Infection Fraction')
    ax7.set_ylabel('Density')
    ax7.set_title('Initial Infection Distributions')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 7. Beta estimation with infected curve
    ax8 = fig.add_subplot(3, 4, 8)
    ax8_twin = ax8.twinx()
    
    if beta_estimates is not None and len(beta_estimates) > 0:
        line1 = ax8.plot(beta_estimates['time'], beta_estimates['beta_estimate'], 
                        'g-', linewidth=2, label='Beta Estimates')
    line2 = ax8_twin.plot(real_data.index, real_data['I'], 'k-', alpha=0.6, label='Infected')
    
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Beta Estimate', color='g')
    ax8_twin.set_ylabel('Infected Count', color='k')
    ax8.set_title('Beta vs Infected Over Time')
    ax8.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax8.get_legend_handles_labels()
    lines2, labels2 = ax8_twin.get_legend_handles_labels()
    ax8.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 8. Summary statistics
    ax9 = fig.add_subplot(3, 4, 9)
    ax9.axis('off')
    
    summary_text = "SUMMARY STATISTICS\n\n"
    
    if len(network_results) > 0:
        summary_text += f"Network SEIR:\n"
        summary_text += f"  Best τ: {network_results.loc[network_results['distance'].idxmin(), 'tau']:.4f}\n"
        summary_text += f"  Best ρ: {network_results.loc[network_results['distance'].idxmin(), 'rho']:.4f}\n"
        summary_text += f"  Min distance: {network_results['distance'].min():.2f}\n\n"
    
    if len(standard_results) > 0:
        summary_text += f"Differential SEIR:\n"
        summary_text += f"  Best β: {standard_results.loc[standard_results['distance'].idxmin(), 'beta']:.4f}\n"
        summary_text += f"  Best ρ: {standard_results.loc[standard_results['distance'].idxmin(), 'rho']:.4f}\n"
        summary_text += f"  Min distance: {standard_results['distance'].min():.2f}\n\n"
    
    if beta_estimates is not None and len(beta_estimates) > 0:
        summary_text += f"Beta Estimation:\n"
        summary_text += f"  Mean β: {beta_estimates['beta_estimate'].mean():.4f}\n"
        summary_text += f"  Final β: {beta_estimates['beta_estimate'].iloc[-1]:.4f}\n"
        summary_text += f"  Std β: {beta_estimates['beta_estimate'].std():.4f}\n\n"
    
    summary_text += f"True Parameters:\n"
    summary_text += f"  τ/β: 0.3\n"
    summary_text += f"  ρ: 0.01\n"
    summary_text += f"  α: 0.2\n"
    summary_text += f"  γ: 0.1"
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("=== MODEL COMPARISON ===")
    
    # Generate "real data" from specific seed
    real_data = generate_real_data_from_seed(seed=0, tau=0.3, alpha=0.2, gamma=0.1, rho=0.01, tmax=99)
    
    print(f"Generated real data with {len(real_data)} time points")
    print(f"Peak infected: {real_data['I'].max()}")
    print(f"Total infected at end: {real_data['I'].iloc[-1]}")
    
    # Run model comparison
    comparison_results = compare_models(real_data, n_samples=200)
    
    # Plot comparison
    comparison_fig = plot_model_comparison(comparison_results, real_data)
    plt.show()
    
    # Plot individual model results
    print("\n=== INDIVIDUAL MODEL PLOTS ===")
    
    # Network SEIR detailed plots
    network_fig = comparison_results['network_seir'].plot_results(
        comparison_results['network_results'], "Network SEIR Detailed", n_trajectories=10)
    plt.show()
    
    # Differential SEIR detailed plots
    standard_fig = comparison_results['standard_seir'].plot_results(
        comparison_results['standard_results'], "Differential SEIR Detailed", n_trajectories=10)
    plt.show()
    
    # Beta estimation detailed plots
    beta_fig = comparison_results['beta_estimator'].plot_beta_estimates()
    plt.show()
    
    # Print final summary
    print("\nFINAL SUMMARY:")
    print(f"    Network SEIR accepted parameters: {len(comparison_results['network_results'])}")
    print(f"    Differential SEIR accepted parameters: {len(comparison_results['standard_results'])}")
    print(f"    Beta estimation points: {len(comparison_results['beta_estimates'])}")
