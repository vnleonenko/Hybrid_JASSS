import numpy as np


class SEIRParams:
    def __init__(self, beta, gamma, delta, init_inf_frac, init_rec_frac, tmax=None):
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.init_inf_frac = init_inf_frac
        self.init_rec_frac = init_rec_frac
        self.tmax = tmax

    def __call__(self):
        params = self.as_list()
        print('beta: {}, gamma: {}, delta: {}, init_inf_frac: {}, init_rec_frac: {}'.format(
            *np.round(params, 2)))
        return params

    def as_list(self):
        return [self.beta, self.gamma, self.delta, self.init_inf_frac, self.init_rec_frac]


class SEIRModelOutput:
    '''
    Class for storing output of SEIR model. 
    Takes time, susceptible, exposed, infectious and recovered arrays
    as input for initialization.
    '''

    def __init__(self, t, S, E, I, R):
        # compartments
        self.t = t
        self.S = S
        self.E = E
        self.I = I
        self.R = R

        # epidemic indicators
        self.daily_incidence = None
        self.weekly_incidence = None
        self.daily_rt = None
        self.weekly_rt = None

        # prepare epidemic indicators, calculate daily and weekly values for them
        self.calculate_incidence()
        self.calculate_rt()

    def pad_array_to_multiple_of_seven(self, arr):
        '''
        Auxiliary function used for padding array of daily data by zeroes for converting
        to weekly data
        '''
        current_size = len(arr)
        new_size = (current_size + 6) // 7 * 7
        padding_needed = new_size - current_size
        padded_array = np.pad(arr, (0, padding_needed),
                              mode='constant', constant_values=0)
        return padded_array

    def calculate_incidence(self):
        '''
        Calculates newly infected cases using SEIR compatrments arrays.
        '''
        self.daily_incidence = [0 if index == 0 else ((self.E[index-1] - self.E[index]) -
                                                      (self.S[index] - self.S[index-1])) for index in range(len(self.S))]
        daily_incidence_padded = self.pad_array_to_multiple_of_seven(
            self.daily_incidence)
        self.weekly_incidence = np.nansum(
            daily_incidence_padded.reshape(-1, 7), axis=1)

    def calculate_rt(self):
        '''
        Calculates effective reproduction number by the following equation: 
        Rt(i) = (newly infected(i))/(newly recovered(i)),
        where i is the number of day.
        '''
        new_recoveries = [0 if index == 0 else (
            self.R[index] - self.R[index-1]) for index in range(len(self.R))]
        self.rt_daily = [self.daily_incidence[index]/(new_recoveries[index]) if new_recoveries[index] != 0 else float('nan')
                         for index in range(len(self.daily_incidence))]
        daily_rt_padded = self.pad_array_to_multiple_of_seven(self.rt_daily)
        self.weekly_rt = np.nansum(daily_rt_padded.reshape(-1, 7), axis=1)
