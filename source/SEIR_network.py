import math
import scipy
import numpy as np
import networkx as nx
import EoN

from collections import defaultdict
from .model_output import SEIRModelOutput, SEIRParams


class SEIRNetworkModel():
    def __init__(self, population: int):
        self.population = population

        # FOLLOWING PARAMETERS ARE EPIDEMICALLY DETERMINED
        # R_0 HERE LIES IN RANGE [1; 2.5]
        self.min_params = SEIRParams(
            alpha=0.99, beta=1/9, gamma=1/5, delta=1/9, init_inf_frac=1e-6)
        self.max_params = SEIRParams(
            alpha=0.8, beta=0.625, gamma=1, delta=1/4, init_inf_frac=1e-3)
        self.last_sim_params = None
        self.G = nx.barabasi_albert_graph(population, m=5, seed=42)

    @staticmethod
    def find_nearest_idx(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def transform_event_times_to_days(self, model_output, tmax):
        indices = []
        for day in range(tmax):
            index = self.find_nearest_idx(model_output.t, day)
            indices.append(index)
        new_model_output = SEIRModelOutput(model_output.t[indices], model_output.S[indices],
                                           model_output.E[indices], model_output.I[indices],
                                           model_output.R[indices])
        return new_model_output

    def simulate(self, alpha=0.85, beta=1/7*1.5, gamma=1/2, delta=1/7, init_inf_frac=1e-4, tmax: int = 150):
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
        initial_recovered = int((1-alpha)*self.population)
        assert initial_recovered + initial_infected < self.population, \
            "Incorrect initial conditions, immune + infected > population size!"
        for node in range(initial_recovered):
            initial_status[node+initial_infected] = 'R'
        self.result = self.transform_event_times_to_days(SEIRModelOutput
                                                         (*EoN.Gillespie_simple_contagion(self.G, H, J, initial_status,
                                                                                          return_statuses=('S', 'E', 'I', 'R'), tmax=tmax)), tmax)
        return self.result
