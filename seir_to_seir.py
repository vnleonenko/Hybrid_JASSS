import pandas as pd
import numpy as np
import os
import glob
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt

def load_seir_data(base_dirs=None, pattern="seirb_seed_*.csv"):
    """
    Load SEIR data from multiple directories
    """
    if base_dirs is None:
        base_dirs = ['./results/chelyabinsk_10']
    
    all_data = []
    files_found = 0
    
    for base_dir in base_dirs:
        search_pattern = os.path.join(base_dir, pattern)
        files = glob.glob(search_pattern)
        for file in files:
            print(file)
            try:
                data = pd.read_csv(file, delimiter="\t").fillna(0)
                data['source_file'] = os.path.basename(file)
                all_data.append(data)
                files_found += 1
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    print(f"Successfully loaded {files_found} files")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        print("No valid data found!")
        return pd.DataFrame()

def prepare_for_history_matching(data, beta_col='beta_H1N1', target_cols=None):
    """
    Prepare data for history matching, focusing on beta parameter
    """
    if target_cols is None:
        target_cols = ['S_H1N1', 'E_H1N1', 'I_H1N1', 'R_H1N1']
    
    if data.empty:
        raise ValueError("No data to process")
    
    if beta_col not in data.columns:
        raise ValueError(f"Beta column '{beta_col}' not found in data")
    
    missing_cols = [col for col in target_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing target columns: {missing_cols}")
    
    processed_data = data[[beta_col] + target_cols].copy()
    
    # data quality check:
    print(f"Beta parameter range: {processed_data[beta_col].min()} to {processed_data[beta_col].max()}")
    print(f"Total samples: {len(processed_data)}")
    
    return processed_data

def history_matching_beta(data, target_col='I_H1N1', beta_col='beta_H1N1',
                         observed_value=None, obs_error_var=100):
    """
    Perform history matching to identify plausible beta values
    """
    if observed_value is None:
        # if no observed value provided, we use mean of the data as example
        observed_value = data[target_col].mean()
        print(f"Using mean value {observed_value} as observed {target_col}")
    
    # prepare data for emulator
    X = data[[beta_col]].values
    y = data[target_col].values
    
    # scale the input data to improve numerical stability
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    
    # scale the target data if values are very small or very large
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    
    # adjust kernel parameters based on data characteristics
    length_scale = 1.0  # Default value for scaled data
    kernel = C(1.0, (1e-8, 1e2)) * RBF(length_scale, (1e-8, 1e2)) 
    # C (ConstantKernel) controls the overall magnitude of predictions
    # RBF (Radial Basis Function) determines how smooth patterns are
    
    # custom optimizer with increased max_iter
    def custom_optimizer(obj_func, initial_theta, bounds): # fine-tunes kernel parameters to best match the data
        from scipy.optimize import minimize
        opt_res = minimize(
            obj_func,
            initial_theta,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 1000}
        )
        return opt_res.x, opt_res.fun
    
    # create and fit GP with custom optimizer
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=15,
        alpha=1e-8,
        normalize_y=False,
        optimizer=custom_optimizer,
        random_state=42
    )
    
    # fit the model with scaled data
    gp.fit(X_scaled, y_scaled)
    print("Gaussian Process emulator fitted")
    
    # define implausibility function that handles scaling properly
    def implausibility(beta_value):
        # scale the beta value the same way as training data
        beta_scaled = X_scaler.transform(np.array([[beta_value]]))
        
        # get prediction and scale back
        prediction_scaled, std_scaled = gp.predict(beta_scaled, return_std=True)
        prediction = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
        
        # scale standard deviation back to original scale
        std = std_scaled[0] * y_scaler.scale_[0]
        emulator_var = std**2
        
        # calculate implausibility
        imp = np.abs(prediction - observed_value) / np.sqrt(emulator_var + obs_error_var)
        return imp
    
    # scan beta range to find plausible values
    unique_betas = np.sort(data[beta_col].unique())
    if len(unique_betas) < 20:  # if few unique values, create more test points
        beta_range = np.linspace(min(unique_betas), max(unique_betas), 100)
    else:
        beta_range = unique_betas
    
    # calculate implausibility for each beta
    implausibilities = [implausibility(beta) for beta in beta_range]
    
    # identify plausible betas (implausibility < 3)
    plausible_mask = np.array(implausibilities) <= 3
    plausible_betas = beta_range[plausible_mask]
    
    # store original GP model and scalers for later use
    class ScaledEmulator:
        def __init__(self, gp, X_scaler, y_scaler):
            self.gp = gp
            self.X_scaler = X_scaler
            self.y_scaler = y_scaler
        
        def predict(self, X, return_std=False):
            X_scaled = self.X_scaler.transform(X)
            if return_std:
                y_pred_scaled, std_scaled = self.gp.predict(X_scaled, return_std=True)
                y_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                # scale standard deviation back
                std = std_scaled * self.y_scaler.scale_[0]
                return y_pred, std
            else:
                y_pred_scaled = self.gp.predict(X_scaled)
                return self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # create a wrapped emulator that handles scaling
    scaled_emulator = ScaledEmulator(gp, X_scaler, y_scaler)
    
    results = {
        'emulator': scaled_emulator,
        'beta_range': beta_range,
        'implausibilities': implausibilities,
        'plausible_betas': plausible_betas,
        'observed_value': observed_value
    }
    
    print(f"Found {len(plausible_betas)} plausible beta values out of {len(beta_range)} tested")
    if len(plausible_betas) > 0:
        print(f"Plausible beta range: {min(plausible_betas)} to {max(plausible_betas)}")
    
    return results

def visualize_history_matching(results):
    """
    Visualize history matching results
    """
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Implausibility
    plt.subplot(1, 2, 1)
    plt.scatter(results['beta_range'], results['implausibilities'],
                c=['green' if i < 3 else 'red' for i in results['implausibilities']],
                alpha=0.6)
    plt.axhline(y=3, color='black', linestyle='--', label='Threshold = 3')
    plt.xlabel('Beta')
    plt.ylabel('Implausibility')
    plt.title('Implausibility vs Beta')
    plt.legend()
    
    # Plot 2: Emulator prediction with uncertainty
    beta_fine = np.linspace(min(results['beta_range']), max(results['beta_range']), 200)
    beta_fine_reshaped = beta_fine.reshape(-1, 1)
    y_pred, sigma = results['emulator'].predict(beta_fine_reshaped, return_std=True)
    
    plt.subplot(1, 2, 2)
    plt.plot(beta_fine, y_pred, label='Emulator prediction')
    plt.fill_between(beta_fine, y_pred - 2*sigma, y_pred + 2*sigma, alpha=0.2, label='95% CI')
    plt.axhline(y=results['observed_value'], color='red', linestyle='-', label='Observed')
    plt.xlabel('Beta')
    plt.ylabel('Predicted infected')
    plt.title('Emulator prediction with uncertainty')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return plt

def estimate_rejection_epsilon(plausible_betas, emulator, observed_value, target_acceptance=0.1):
    """
    Estimate appropriate epsilon for rejection sampling to achieve target acceptance rate
    """
    print("Estimating appropriate epsilon for rejection sampling...")
    
    # sample distances from prior
    distances = []
    n_samples = 2000
    
    for _ in range(n_samples):
        beta = np.random.choice(plausible_betas)
        prediction, std = emulator.predict(np.array([[beta]]), return_std=True)
        simulated_value = np.random.normal(prediction[0], max(std[0], 1e-6))
        distance = np.abs(simulated_value - observed_value)
        distances.append(distance)
    
    distances = np.array(distances)
    
    # find epsilon that gives target acceptance rate
    epsilon_candidates = np.percentile(distances, [5, 10, 20, 30, 40, 50])
    
    print(f"Distance statistics:")
    print(f"  Min: {np.min(distances):.4f}")
    print(f"  5th percentile: {np.percentile(distances, 5):.4f}")
    print(f"  10th percentile: {np.percentile(distances, 10):.4f}")
    print(f"  Median: {np.median(distances):.4f}")
    print(f"  90th percentile: {np.percentile(distances, 90):.4f}")
    print(f"  Max: {np.max(distances):.4f}")
    
    # choose epsilon that gives roughly target acceptance rate
    target_percentile = target_acceptance * 100
    estimated_epsilon = np.percentile(distances, target_percentile)
    
    print(f"Estimated epsilon for {target_acceptance*100}% acceptance: {estimated_epsilon:.4f}")
    
    return estimated_epsilon, distances

# Rejection sampling
def abc_with_simple_rejection_improved(plausible_betas, emulator, observed_value,
                                     abc_samples=1000, epsilon=None, target_acceptance=0.1):
    """
    Improved rejection sampling with adaptive epsilon
    """
    start_time = time.time()
    
    if len(plausible_betas) == 0:
        print("No plausible beta values found for ABC")
        return None
    
    # estimate epsilon if not provided
    if epsilon is None:
        epsilon, prior_distances = estimate_rejection_epsilon(
            plausible_betas, emulator, observed_value, target_acceptance
        )
    
    print(f"Using epsilon = {epsilon:.4f}")
    
    # setup for ABC
    accepted_samples = []
    distances_tried = []
    attempts = 0
    max_attempts = abc_samples * 1000  # increased max attempts
    
    while len(accepted_samples) < abc_samples and attempts < max_attempts:
        # sample from plausible beta values
        beta = np.random.choice(plausible_betas)
        
        # use emulator as fast surrogate for the model
        prediction, std = emulator.predict(np.array([[beta]]), return_std=True)
        
        # add noise to represent model stochasticity
        simulated_value = np.random.normal(prediction[0], max(std[0], 1e-6))
        
        # ABC acceptance criterion
        distance = np.abs(simulated_value - observed_value)
        distances_tried.append(distance)
        
        if distance < epsilon:
            accepted_samples.append(beta)
        
        attempts += 1
        
        # progress update
        if attempts % (max_attempts // 20) == 0:
            current_rate = len(accepted_samples) / attempts * 100
            print(f"  Progress: {attempts}/{max_attempts}, accepted: {len(accepted_samples)} (rate: {current_rate:.2f}%)")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    acceptance_rate = len(accepted_samples) / attempts * 100
    
    print(f"ABC completed: {len(accepted_samples)} samples accepted")
    print(f"Acceptance rate: {acceptance_rate:.2f}%")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    if len(accepted_samples) == 0:
        print("\nTroubleshooting:")
        min_distance = np.min(distances_tried)
        print(f"Minimum distance achieved: {min_distance:.4f}")
        print(f"Current epsilon: {epsilon:.4f}")
        print(f"Try epsilon >= {min_distance * 1.1:.4f}")
    
    return {
        'accepted_samples': np.array(accepted_samples) if accepted_samples else np.array([]),
        'acceptance_rate': acceptance_rate,
        'attempts': attempts,
        'elapsed_time': elapsed_time,
        'epsilon_used': epsilon,
        'distances_tried': distances_tried
    }

# Simulated annealing
def abc_with_simulated_annealing(plausible_betas, emulator, observed_value,
                                abc_samples=1000, initial_temp=10.0, cooling_rate=0.95):
    """
    Perform ABC using simulated annealing within the history-matched parameter space
    """
    start_time = time.time()
    
    if len(plausible_betas) == 0:
        print("No plausible beta values found for ABC")
        return None
    
    # setup for SABC
    accepted_samples = []
    current_beta = np.random.choice(plausible_betas)  # initial state
    current_temp = initial_temp
    attempts = 0
    max_attempts = abc_samples * 100  # avoid infinite loops
    
    # function to calculate model fit quality
    def calculate_distance(beta):
        prediction, std = emulator.predict(np.array([[beta]]), return_std=True)
        # add noise to represent model stochasticity
        simulated_value = np.random.normal(prediction[0], max(std[0], 1e-6))
        return np.abs(simulated_value - observed_value)
    
    # get initial distance
    current_distance = calculate_distance(current_beta)
    
    # simulated annealing process
    while len(accepted_samples) < abc_samples and attempts < max_attempts:
        # propose new beta value
        candidate_beta = np.random.choice(plausible_betas)
        candidate_distance = calculate_distance(candidate_beta)
        
        # calculate acceptance probability
        if candidate_distance <= current_distance:
            # always accept better solutions
            acceptance_probability = 1.0
        else:
            # accept worse solutions with probability based on temperature
            acceptance_probability = np.exp(-(candidate_distance - current_distance) / current_temp)
        
        # accept or reject candidate
        if np.random.random() < acceptance_probability:
            current_beta = candidate_beta
            current_distance = candidate_distance
            accepted_samples.append(current_beta)
        
        # cool down temperature
        current_temp *= cooling_rate
        attempts += 1
        
        # restart if temperature gets too low
        if current_temp < 0.01:
            current_temp = initial_temp * 0.5  # restart with lower temperature
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    acceptance_rate = len(accepted_samples) / attempts * 100
    print(f"SABC completed: {len(accepted_samples)} samples accepted")
    print(f"Final temperature: {current_temp:.6f}")
    print(f"Acceptance rate: {acceptance_rate:.2f}%")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    return {
        'accepted_samples': np.array(accepted_samples),
        'acceptance_rate': acceptance_rate,
        'attempts': attempts,
        'elapsed_time': elapsed_time
    }

def estimate_initial_epsilon(plausible_betas, emulator, observed_value, percentile=90):
    """
    Estimate good initial epsilon based on prior predictive distances
    """

    print("Estimating appropriate initial epsilon...")
    distances = []
    
    for _ in range(1000):  # sample from prior
        beta = np.random.choice(plausible_betas)
        prediction, std = emulator.predict(np.array([[beta]]), return_std=True)
        simulated = np.random.normal(prediction[0], max(std[0], 1e-6))
        distance = np.abs(simulated - observed_value)
        distances.append(distance)
    
    distances = np.array(distances)
    initial_eps = np.percentile(distances, percentile)
    
    print(f"Distance statistics:")
    print(f"  Min: {np.min(distances):.4f}")
    print(f"  Median: {np.median(distances):.4f}")
    print(f"  {percentile}th percentile: {initial_eps:.4f}")
    print(f"  Max: {np.max(distances):.4f}")
    
    return initial_eps

# Sequential Monte Carlo
def abc_with_sequential_monte_carlo(plausible_betas, emulator, observed_value,
                                   n_populations=5, population_size=200,
                                   initial_epsilon=None, final_epsilon=0.05,
                                   kernel_width_factor=0.5):
    """
    Perform Sequential Monte Carlo ABC using the emulator and history-matched parameter space
    """
    start_time = time.time()
    
    if len(plausible_betas) == 0:
        print("No plausible beta values found for ABC")
        return None
    
    # estimate initial epsilon if not provided
    if initial_epsilon is None:
        initial_epsilon = estimate_initial_epsilon(plausible_betas, emulator, observed_value, percentile=95)
        initial_epsilon = max(initial_epsilon, 2.0)  # ensure minimum threshold
    
    print(f"Using initial epsilon: {initial_epsilon:.4f}")
    
    # calculate epsilon schedule with exponential decay
    epsilons = np.exp(np.linspace(np.log(initial_epsilon), np.log(final_epsilon), n_populations))
    
    # data structures to store results
    all_populations = []
    all_weights = []
    acceptance_rates = []
    
    # function to calculate distance between simulated and observed data
    def calculate_distance(beta):
        prediction, std = emulator.predict(np.array([[beta]]), return_std=True)
        # add noise to represent model stochasticity
        simulated_value = np.random.normal(prediction[0], max(std[0], 1e-6))  # ensure std > 0
        return np.abs(simulated_value - observed_value)
    
    print(f"SMC-ABC: starting with {n_populations} populations")
    print(f"Epsilon schedule: {epsilons}")
    
    # generate first population from plausible beta values
    population = []
    weights = []
    attempts = 0
    max_attempts = population_size * 1000  # increased max attempts
    
    print(f"Population 1/{n_populations}, epsilon={epsilons[0]:.4f}")
    
    # first generation: sample from history-matched prior
    while len(population) < population_size and attempts < max_attempts:
        beta = np.random.choice(plausible_betas)
        distance = calculate_distance(beta)
        
        if distance < epsilons[0]:
            population.append(beta)
            weights.append(1.0)  # uniform weights in first population
        
        attempts += 1
        
        # progress indicator for long runs
        if attempts % (max_attempts // 10) == 0:
            print(f"  Progress: {attempts}/{max_attempts}, accepted: {len(population)}")
    
    # check if we have enough samples
    if len(population) == 0:
        print(f"No samples accepted with epsilon={epsilons[0]:.4f}")
        print("Suggestions:")
        print(f"- Try initial_epsilon >= {epsilons[0] * 5:.2f}")
        print("- Check if plausible_betas range is appropriate")
        print("- Verify emulator predictions are reasonable")
        return None
    
    if len(population) < population_size * 0.1:  # less than 10% of target
        print(f"Warning: Only {len(population)} samples accepted (target: {population_size})")
        print("Consider increasing initial_epsilon for better sampling efficiency")
    
    # normalize weights
    weights = np.array(weights) / sum(weights)
    population = np.array(population)
    
    acceptance_rate = len(population) / attempts * 100
    acceptance_rates.append(acceptance_rate)
    all_populations.append(population)
    all_weights.append(weights)
    
    print(f" Accepted: {len(population)}/{attempts} (rate: {acceptance_rate:.2f}%)")
    
    # generate subsequent populations
    for t in range(1, n_populations):
        print(f"Population {t+1}/{n_populations}, epsilon={epsilons[t]:.4f}")
        
        new_population = []
        new_weights = []
        attempts = 0
        
        # calculate kernel width based on weighted variance of previous population
        try:
            # check if weights sum to zero (fixed bug)
            if np.sum(weights) == 0 or not np.all(np.isfinite(weights)):
                print("Warning: Invalid weights detected. Using uniform weights.")
                weights = np.ones(len(population)) / len(population)
            
            weighted_mean = np.average(population, weights=weights)
            weighted_var = np.sum(weights * (population - weighted_mean)**2)
            
            # ensure kernel width is not zero or too small
            kernel_width = max(np.sqrt(weighted_var) * kernel_width_factor, 1e-6)
            
            # additional safety check
            if kernel_width == 0 or np.isnan(kernel_width) or np.isinf(kernel_width):
                kernel_width = (np.max(plausible_betas) - np.min(plausible_betas)) / 20
                
        except Exception as e:
            print(f"Error calculating kernel width: {e}")
            kernel_width = (np.max(plausible_betas) - np.min(plausible_betas)) / 20
        
        print(f" Kernel width: {kernel_width:.6f}")
        
        max_attempts_pop = population_size * 1000
        while len(new_population) < population_size and attempts < max_attempts_pop:
            # sample from previous population according to weights
            try:
                if np.sum(weights) > 0 and np.all(np.isfinite(weights)):
                    beta_idx = np.random.choice(len(population), p=weights)
                else:
                    beta_idx = np.random.choice(len(population))
                beta_old = population[beta_idx]
            except (ValueError, IndexError):
                # if weights are problematic, use uniform sampling
                beta_idx = np.random.choice(len(population))
                beta_old = population[beta_idx]
            
            # perturb the parameter (add noise)
            beta_proposed = np.random.normal(beta_old, kernel_width)
            
            # check if proposed beta is in plausible range
            if beta_proposed >= np.min(plausible_betas) and beta_proposed <= np.max(plausible_betas):
                distance = calculate_distance(beta_proposed)
                
                if distance < epsilons[t]:
                    new_population.append(beta_proposed)
                    
                    # calculate weight using importance sampling formula
                    prior_weight = 1.0  # uniform prior over plausible range
                    
                    # transition density (Gaussian kernel)
                    transition_densities = np.exp(-0.5 * ((beta_proposed - population) / kernel_width)**2) / (kernel_width * np.sqrt(2*np.pi))
                    transition_density = np.sum(weights * transition_densities)
                    
                    # avoid division by zero
                    if transition_density == 0 or not np.isfinite(transition_density):
                        new_weight = 1.0  # fallback weight
                    else:
                        new_weight = prior_weight / transition_density
                    
                    # ensure weight is finite and positive
                    if not np.isfinite(new_weight) or new_weight <= 0:
                        new_weight = 1.0
                    
                    new_weights.append(new_weight)
            
            attempts += 1
            
            # progress indicator
            if attempts % (max_attempts_pop // 10) == 0:
                print(f"  Progress: {attempts}/{max_attempts_pop}, accepted: {len(new_population)}")
        
        # check if we have enough samples
        if len(new_population) < population_size * 0.05:  # less than 5% of target
            print(f" Warning: Only {len(new_population)} samples accepted. Stopping early.")
            break
        
        # normalize weights
        new_population = np.array(new_population)
        new_weights = np.array(new_weights)
        
        # handle case where all weights are zero or invalid
        if len(new_weights) == 0 or np.sum(new_weights) == 0 or not np.all(np.isfinite(new_weights)):
            print(" Warning: Invalid weights detected. Using uniform weights.")
            new_weights = np.ones(len(new_population))
        
        new_weights = new_weights / np.sum(new_weights)
        
        acceptance_rate = len(new_population) / attempts * 100
        acceptance_rates.append(acceptance_rate)
        
        # update for next iteration
        population = new_population
        weights = new_weights
        all_populations.append(population)
        all_weights.append(weights)
        
        print(f" Accepted: {len(population)}/{attempts} (rate: {acceptance_rate:.2f}%)")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"SMC-ABC completed in {elapsed_time:.2f} seconds")
    
    # return final population and diagnostics
    return {
        'populations': all_populations,
        'weights': all_weights,
        'acceptance_rates': acceptance_rates,
        'epsilons': epsilons,
        'final_population': all_populations[-1],
        'final_weights': all_weights[-1],
        'elapsed_time': elapsed_time
    }

def run_abc_method(method, hm_results, **kwargs):
    """
    Run the specified ABC method with history matching results
    """
    if method == 'rejection':
        # default parameters for rejection sampling
        params = {
            'abc_samples': 1000,
            'epsilon': None,
            'target_acceptance': 0.15
        }
        params.update(kwargs)
        
        results = abc_with_simple_rejection_improved(
            hm_results['plausible_betas'],
            hm_results['emulator'],
            hm_results['observed_value'],
            **params
        )
        
        if results is not None and len(results['accepted_samples']) > 0:
            plt.figure(figsize=(10, 5))
            plt.hist(results['accepted_samples'], bins=20, density=True, alpha=0.7)
            plt.xlabel('Beta')
            plt.ylabel('Posterior Density')
            plt.title(f'ABC Posterior for Beta (Acceptance rate: {results["acceptance_rate"]:.2f}%, time: {results["elapsed_time"]:.2f}s)')
            plt.grid(alpha=0.3)
            plt.show()
        
    elif method == 'annealing':
        # default parameters for simulated annealing
        params = {
            'abc_samples': 1000,
            'initial_temp': 10.0,
            'cooling_rate': 0.95
        }
        params.update(kwargs)
        
        results = abc_with_simulated_annealing(
            hm_results['plausible_betas'],
            hm_results['emulator'],
            hm_results['observed_value'],
            **params
        )
        
        if results is not None and len(results['accepted_samples']) > 0:
            plt.figure(figsize=(10, 5))
            plt.hist(results['accepted_samples'], bins=20, density=True, alpha=0.7)
            plt.xlabel('Beta')
            plt.ylabel('Posterior density')
            plt.title(f'Posterior for Beta (Acceptance rate: {results["acceptance_rate"]:.2f}%, time: {results["elapsed_time"]:.2f}s)')
            plt.grid(alpha=0.3)
            plt.show()
        
    elif method == 'smc':
        # default parameters for sequential Monte Carlo
        params = {
            'n_populations': 5,
            'population_size': 200,
            'initial_epsilon': None,  # will be estimated automatically
            'final_epsilon': 0.05
        }
        params.update(kwargs)
        
        results = abc_with_sequential_monte_carlo(
            hm_results['plausible_betas'],
            hm_results['emulator'],
            hm_results['observed_value'],
            **params
        )
        
        if results is not None:
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Final weighted posterior
            plt.subplot(2, 2, 1)
            plt.hist(results['final_population'], bins=20, weights=results['final_weights'],
                    density=True, alpha=0.7)
            plt.xlabel('Beta')
            plt.ylabel('Posterior density')
            plt.title(f'SMC-ABC Posterior (Time: {results["elapsed_time"]:.2f}s)')
            
            # Plot 2: Epsilon and acceptance rate by population
            plt.subplot(2, 2, 2)
            plt.plot(range(1, len(results['epsilons'])+1), results['epsilons'], 'o-', label='Epsilon')
            plt.xlabel('Population')
            plt.ylabel('Epsilon')
            plt.title('Epsilon schedule')
            plt.yscale('log')
            ax2 = plt.twinx()
            ax2.plot(range(1, len(results['acceptance_rates'])+1), 
                    results['acceptance_rates'], 'r--', label='Acceptance %')
            ax2.set_ylabel('Acceptance rate (%)', color='r')
            
            # Plot 3: Population evolution
            plt.subplot(2, 1, 2)
            for i, (pop, w) in enumerate(zip(results['populations'], results['weights'])):
                plt.hist(pop, bins=20, weights=w, alpha=0.3, label=f'Pop {i+1}', density=True)
            plt.xlabel('Beta')
            plt.ylabel('Density')
            plt.title('Evolution of parameter distribution')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
    
    else:
        print(f"Unknown method: {method}. Choose from: rejection, annealing, smc")
        return None
    
    return results

def compare_abc_methods(hm_results, n_samples=1000):
    """
    Compare the performance of all three ABC methods with improved rejection sampling
    """
    results = {}
    
    # Run each method with error handling
    print("Running Improved Rejection Sampling...")
    try:
        results['rejection'] = abc_with_simple_rejection_improved(
            hm_results['plausible_betas'],
            hm_results['emulator'],
            hm_results['observed_value'],
            abc_samples=n_samples,
            epsilon=None, 
            target_acceptance=0.15
        )
    except Exception as e:
        print(f"Error in rejection sampling: {e}")
        results['rejection'] = None
    
    print("\nRunning Simulated Annealing...")
    try:
        results['annealing'] = abc_with_simulated_annealing(
            hm_results['plausible_betas'],
            hm_results['emulator'],
            hm_results['observed_value'],
            abc_samples=n_samples,
            initial_temp=100.0,
            cooling_rate=0.95
        )
    except Exception as e:
        print(f"Error in simulated annealing: {e}")
        results['annealing'] = None
    
    print("\nRunning Sequential Monte Carlo...")
    try:
        # estimate appropriate initial epsilon
        initial_eps = estimate_initial_epsilon(
            hm_results['plausible_betas'],
            hm_results['emulator'], 
            hm_results['observed_value'],
            percentile=95  # we use 95th percentile for very permissive start
        )
        
        results['smc'] = abc_with_sequential_monte_carlo(
            hm_results['plausible_betas'],
            hm_results['emulator'],
            hm_results['observed_value'],
            population_size=200,
            n_populations=5,
            initial_epsilon=max(initial_eps, 2.0),  # ensure minimum threshold
            final_epsilon=0.1
        )
    except Exception as e:
        print(f"Error in sequential monte carlo: {e}")
        results['smc'] = None
    
    # create comparison plots only for successful methods
    successful_methods = {k: v for k, v in results.items() if v is not None and 
                         (k != 'rejection' or len(v['accepted_samples']) > 0)}
    
    if successful_methods:
        # 1. Comparison of runtime
        methods = list(successful_methods.keys())
        runtimes = [successful_methods[m]['elapsed_time'] for m in methods]
        
        plt.figure(figsize=(14, 10))
        
        plt.subplot(2, 2, 1)
        plt.bar(methods, runtimes)
        plt.ylabel('Runtime (seconds)')
        plt.title('Performance comparison: Runtime')
        
        # 2. Comparison of posteriors
        plt.subplot(2, 2, 2)
        for method in methods:
            if method == 'smc':
                if len(successful_methods[method]['final_population']) > 0:
                    plt.hist(successful_methods[method]['final_population'], bins=20, 
                            weights=successful_methods[method]['final_weights'],
                            alpha=0.5, density=True, label=f'{method.upper()}')
            else:
                if len(successful_methods[method]['accepted_samples']) > 0:
                    plt.hist(successful_methods[method]['accepted_samples'], bins=20, 
                            alpha=0.5, density=True, label=f'{method.upper()}')
        plt.xlabel('Beta')
        plt.ylabel('Posterior density')
        plt.title('Posterior distributions')
        plt.legend()
        
        # 3. Acceptance rates
        plt.subplot(2, 2, 3)
        acceptance_rates = []
        for method in methods:
            if method == 'smc':
                # we use final population acceptance rate
                if len(successful_methods[method]['acceptance_rates']) > 0:
                    acceptance_rates.append(successful_methods[method]['acceptance_rates'][-1])
                else:
                    acceptance_rates.append(0)
            else:
                acceptance_rates.append(successful_methods[method]['acceptance_rate'])
        
        plt.bar(methods, acceptance_rates)
        plt.ylabel('Acceptance Rate (%)')
        plt.title('Performance comparison: Acceptance Rate')
        
        # 4. Distance distribution for rejection sampling (if available)
        if 'rejection' in results and results['rejection'] is not None:
            plt.subplot(2, 2, 4)
            distances = results['rejection']['distances_tried']
            epsilon_used = results['rejection']['epsilon_used']
            
            plt.hist(distances, bins=50, alpha=0.7, density=True)
            plt.axvline(epsilon_used, color='red', linestyle='--', 
                       label=f'Epsilon = {epsilon_used:.4f}')
            plt.xlabel('Distance')
            plt.ylabel('Density')
            plt.title('Distance distribution (Rejection sampling)')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("No methods completed successfully. Check your data and parameters.")
    
    return results

if __name__ == "__main__":
    data = load_seir_data()
    
    if not data.empty:
        processed_data = prepare_for_history_matching(data)
        hm_results = history_matching_beta(processed_data, observed_value=200)
        
        visualize_history_matching(hm_results)
        
        print("\n" + "="*50)
        print("TESTING EPSILON VALUES FOR REJECTION SAMPLING")
        print("="*50)
        
        for eps in [1.0, 2.0, 5.0, 10.0, 20.0]:
            print(f"\nTrying epsilon = {eps}")
            result = abc_with_simple_rejection_improved(
                hm_results['plausible_betas'],
                hm_results['emulator'],
                hm_results['observed_value'],
                abc_samples=100,
                epsilon=eps
            )
            if result and len(result['accepted_samples']) > 0:
                print(f"SUCCESS! Found working epsilon = {eps}")
                break
        
        print("\n" + "="*50)
        print("RUNNING FULL COMPARISON")
        print("="*50)
        
        comparison_results = compare_abc_methods(hm_results, n_samples=1000)
        
        print("\n" + "="*50)
        print("SUMMARY OF RESULTS")
        print("="*50)
        
        for method, result in comparison_results.items():
            if result is not None:
                if method == 'smc':
                    n_samples = len(result['final_population'])
                    time_taken = result['elapsed_time']
                    final_acc_rate = result['acceptance_rates'][-1] if result['acceptance_rates'] else 0
                    print(f"{method.upper()}: {n_samples} samples, {time_taken:.2f}s, {final_acc_rate:.2f}% final acceptance")
                else:
                    n_samples = len(result['accepted_samples'])
                    time_taken = result['elapsed_time']
                    acc_rate = result['acceptance_rate']
                    print(f"{method.upper()}: {n_samples} samples, {time_taken:.2f}s, {acc_rate:.2f}% acceptance")
            else:
                print(f"{method.upper()}: FAILED")
        
    else:
        print("No data loaded. Please check your data directory path.")
