import math
import scipy
import numpy as np
import networkx as nx
import EoN

from collections import defaultdict
from .model_output import SEIRModelOutput, SEIRParams


class SEIRNetworkModel():
    def __init__(self, population: int, ntype: str,
                chosen_seed):
        self.population = population

        # FOLLOWING PARAMETERS ARE EPIDEMICALLY DETERMINED
        # R_0 HERE LIES IN RANGE [1; 2.5]
        self.min_params = SEIRParams(
            beta=1/9, gamma=1/5, delta=1/9, 
            init_inf_frac=1e-6, init_rec_frac=1e-2)
        self.max_params = SEIRParams(
            beta=0.625, gamma=1, delta=1/4, 
            init_inf_frac=1e-3, init_rec_frac=2e-1)
        self.last_sim_params = None
        
        if ntype=='ba':
            self.G = nx.barabasi_albert_graph(population, 5, seed=chosen_seed)
            #print('ba')
        elif ntype=='sw':
            self.G = nx.watts_strogatz_graph(population, 5, 0.1, seed=chosen_seed)
            #print('sw')
        elif ntype=='r':
            # чтобы средняя степень была 8
            self.G = nx.fast_gnp_random_graph(population, 5/population, seed=chosen_seed)
            #print('r')
        
        
    @staticmethod
    def find_nearest_idx(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    
    def transform_event_times_to_days(self, model_output, tmax,
                                      I_frac_switch):
        indices = []
        for day in range(tmax):
            index = self.find_nearest_idx(model_output.t, day)
            indices.append(index)
            if model_output.I[index] > self.population*I_frac_switch:
                #print(day)
                break
            
        new_model_output = SEIRModelOutput(model_output.t[indices], 
                                           model_output.S[indices],
                                           model_output.E[indices], 
                                           model_output.I[indices],
                                           model_output.R[indices])
        return new_model_output

    
    def simulate(self, beta=1/7*1.5, gamma=1/2, delta=1/7, 
                 init_inf_frac=1e-4, init_rec_frac=0.15, 
                 tmax: int = 150, I_frac_switch=1):
        '''
        Parameters:

        beta: transmission rate
        gamma: rate of progression from exposed to infectious
        delta: recovery rate
        init_inf_frac: fraction of initially infected
        init_rec_frac: fraction of initially recovered
        '''
        H = nx.DiGraph()
        H.add_edge('E', 'I', rate=gamma)
        H.add_edge('I', 'R', rate=delta)
        J = nx.DiGraph()
        J.add_edge(('I', 'S'), ('I', 'E'), rate=beta)
        initial_infected = int(init_inf_frac*self.population)
        initial_status = defaultdict(lambda: 'S')
        for node in range(initial_infected):
            initial_status[node] = 'I'
        initial_recovered = int(init_rec_frac*self.population)
        assert initial_recovered + initial_infected < self.population, \
            "Incorrect initial conditions, immune + infected > population size!"
        for node in range(initial_recovered):
            initial_status[node+initial_infected] = 'R'
            
        seir_o = SEIRModelOutput(*EoN.Gillespie_simple_contagion(
                                        self.G, H, J, initial_status,
                                        return_statuses=('S', 'E', 
                                                         'I', 'R'),
                                        tmax=tmax,
                                        I_frac_switch=I_frac_switch)
                                 )
        self.result = self.transform_event_times_to_days(seir_o, tmax,
                                                         I_frac_switch)
        
        return self.result
