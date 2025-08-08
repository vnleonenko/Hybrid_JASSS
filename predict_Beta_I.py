import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit


# our functions
import seir_discrete 

import warnings
warnings.filterwarnings(action='ignore')


def load_saved_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


def biexponential_decay_func(x,a,b,c): 
    return a*(np.exp(-b*x)- np.exp(-c*x))


def inc_learning(seed_df, start_day, model_path):
    model_il = load_saved_model(model_path)

    x2 = np.arange(start_day).reshape(-1, 1)
    y2 = seed_df.iloc[:start_day]['Beta'].replace(to_replace=0, 
                                                  method='bfill'
                                                 ).replace(to_replace=0,
                                                           method='ffill'
                                                          ).values
    y2 = np.log(y2)

    t = model_il.named_steps['standardscaler'].transform(x2)
    name_2nd = list(model_il.named_steps.keys())[1]
    t = model_il.named_steps[name_2nd].transform(t)

    if model_il.named_steps['sgdregressor'].warm_start:
        # for warm_start=True .use fit()
        model_il.named_steps['sgdregressor'].fit(t,y2)
    else:
        for i in range(3):
            # for warm_start=False use .partial_fit()
            model_il.named_steps['sgdregressor'].partial_fit(t,y2)

    return model_il


class LSTMPredictor:
    """
    Wraps the trained LSTM model to predict beta on a rolling window of
    [day, E] (2 features). 
    The model was trained to predict normalized log_beta, so this class
    denormalizes the prediction and returns beta.
    """
    def __init__(self, model, full_scaler, window_size):
        self.model = model
        self.n_feats = 2
        # Create a scaler for input features
        # Corrected feature_indices calculation:
        feature_indices = list(range(self.n_feats))
        self.input_scaler = StandardScaler()
        self.input_scaler.mean_ = full_scaler.mean_[feature_indices]
        self.input_scaler.scale_ = full_scaler.scale_[feature_indices]
        self.input_scaler.var_ = full_scaler.var_[feature_indices]
        self.input_scaler.n_features_in_ = self.n_feats
        self.window_size = window_size
        self.buffer = []
        # Store target parameters for log_beta (7th column)
        self.target_mean = full_scaler.mean_[-1]
        self.target_scale = full_scaler.scale_[-1]
        
    def update_buffer(self, new_data):
        # new_data should be a list with 3 elements: [day, E, prev_I]
        self.buffer.append(new_data)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
            
    def predict_next(self):
        # Ensure the buffer has window_size rows
        if len(self.buffer) < self.window_size:
            padded = np.zeros((self.window_size, self.n_feats))
            padded[-len(self.buffer):] = self.buffer
        else:
            padded = np.array(self.buffer[-self.window_size:])
            
        scaled = self.input_scaler.transform(padded)
        scaled_window = scaled.reshape(1, self.window_size, self.n_feats)
        normalized_pred = self.model.predict(scaled_window, verbose=0)[0][0]
        # Denormalize to obtain the raw log_beta
        raw_log_beta = normalized_pred * self.target_scale + self.target_mean
        # Compute beta by exponentiating the log_beta
        predicted_beta = np.exp(raw_log_beta)
        return predicted_beta

    
def predict_beta(I_prediction_method, seed_df, beta_prediction_method, predicted_days, 
                 stochastic, count_stoch_line, sigma, gamma, 
                 features_reg='', model_path='', window_size=14):
    
    '''
    Predict Beta values.

    Parameters:

    - I_prediction_method -- mathematical model for predicting Infected trajectories
        ['seir']
    - seed_df -- DataFrame of seed, created by a regular network
    - beta_prediction_method -- method for predicting Beta values
        ['last_value',
        'rolling mean last value',
        'expanding mean last value',
        'biexponential decay', 
        'median beta',
        'regression (day)'

        'median beta;\nshifted forecast',
        'regression (day);\nshifted forecast',
        'regression (day);\nincremental learning',
        'regression (day, SEIR, previous I)',       
        'lstm (day, E, previous I)']
    - predicted_days -- days for prediction
    - stochastic -- indicator of the presence of predicted trajectories by a stochastic mathematical model
    - count_stoch_line -- number of trajectories predicted by the stochastic mathematical model
    - sigma -- parameter of the SEIR-type mathematical model
    - gamma -- parameter of the SEIR-type mathematical model
    '''
    predicted_I = np.zeros((count_stoch_line+1, predicted_days.shape[0]))
    beggining_beta = []

    
    if beta_prediction_method == 'last value':
        predicted_beta = [seed_df.iloc[predicted_days[0]]['Beta'] 
                          for i in range(predicted_days.shape[0])]

    elif beta_prediction_method == 'rolling mean last value':
        window_size = 7
        beggining_beta = seed_df.iloc[:predicted_days[0]]['Beta'
                                                         ].rolling(window=window_size).mean()
        predicted_beta = [beggining_beta.iloc[-1] 
                          for i in range(predicted_days.shape[0])]
    
    elif beta_prediction_method == 'expanding mean last value':
        betas = seed_df.iloc[:predicted_days[0]]['Beta'].mean()
        beggining_beta = [seed_df.iloc[:i]['Beta'].mean() for i in range(predicted_days[0])]
        predicted_beta = [betas for i in range(predicted_days.shape[0])]

    elif beta_prediction_method == 'expanding mean':
        betas = seed_df.iloc[:]['Beta'].expanding(1).mean().values
        beggining_beta = betas[:predicted_days[0]]
        predicted_beta = betas[predicted_days[0]:]

    elif beta_prediction_method == 'biexponential decay':
        given_betas = seed_df.iloc[:predicted_days[0]]['Beta'].values
        given_days = np.arange(predicted_days[0])
        coeffs, _ = curve_fit(biexponential_decay_func, given_days, given_betas, maxfev=5000)
        beggining_beta = biexponential_decay_func(given_days, *coeffs)
        predicted_beta = biexponential_decay_func(predicted_days, *coeffs)
        predicted_beta[predicted_beta < 0] = 0

    elif beta_prediction_method == 'median beta':
        betas = pd.read_csv(model_path) #'train/median_beta.csv'
        beggining_beta = betas.iloc[:predicted_days[0]]['median_beta'].values
        predicted_beta = betas.iloc[predicted_days[0]:]['median_beta'].values

    elif beta_prediction_method == 'median beta;\nshifted forecast':
        betas = pd.read_csv(model_path)
        beggining_beta = betas.iloc[:predicted_days[0]]['median_beta'].values
        predicted_beta = betas.iloc[predicted_days[0]:]['median_beta'].values
        change = seed_df['Beta'].rolling(7).mean()[predicted_days[0]
                                                  ] - predicted_beta[0]
        beggining_beta = beggining_beta + change
        beggining_beta[beggining_beta<0] = 0
        predicted_beta = predicted_beta + change
        predicted_beta[predicted_beta<0] = 0
        

    elif beta_prediction_method == 'regression (day)':
        #model_path = 'regression_day_for_seir.joblib'
        model = load_saved_model(model_path)
        x_test = np.arange(0,predicted_days[0]).reshape(-1, 1)
        beggining_beta = np.exp(model.predict(x_test))
        x_test = np.arange(predicted_days[0], seed_df.shape[0]).reshape(-1, 1)
        predicted_beta = np.exp(model.predict(x_test))

    elif beta_prediction_method == 'regression (day);\nshifted forecast':
        #model_path = 'regression_day_for_seir.joblib'
        model = load_saved_model(model_path)
        x_test = np.arange(0,predicted_days[0]).reshape(-1, 1)
        beggining_beta = np.exp(model.predict(x_test))
        x_test = np.arange(predicted_days[0], seed_df.shape[0]).reshape(-1, 1)
        predicted_beta = np.exp(model.predict(x_test))
        
        change = seed_df['Beta'].rolling(7).mean()[predicted_days[0]
                                                  ] - predicted_beta[0]
        beggining_beta = beggining_beta + change
        beggining_beta[beggining_beta<0] = 0
        predicted_beta = predicted_beta + change
        predicted_beta[predicted_beta<0] = 0

    elif beta_prediction_method == 'regression (day);\nincremental learning':
        # model_path = 'regression_day_for_seir.joblib'
        model = inc_learning(seed_df, predicted_days[0], model_path)
        x_test = np.arange(0,predicted_days[0]).reshape(-1, 1)
        beggining_beta = np.exp(model.predict(x_test))
        x_test = np.arange(predicted_days[0], seed_df.shape[0]).reshape(-1, 1)
        predicted_beta = np.exp(model.predict(x_test))

    elif beta_prediction_method == 'regression (day, SEIR, previous I)':
        
        predicted_beta = np.empty((0,))
        S = np.zeros((count_stoch_line+1, 2))
        E = np.zeros((count_stoch_line+1, 2))
        R = np.zeros((count_stoch_line+1, 2))

        S[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['S']
        predicted_I[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['I']
        R[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['R']  
        E[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['E'] 
        
        features_reg = ['day',
                        #'prev_I',
                        'S','E','I',
                        #'R'
                       ]

        y = np.array([S[:,0], E[:,0], predicted_I[:,0], R[:,0]])
        y = y.T
        model = load_saved_model(model_path)
        
        prev_I = seed_df.iloc[predicted_days[0]-1:predicted_days[0]
                             ]['I'
                              ].to_numpy() if predicted_days[0
                                              ] > 1 else np.array([0.0, 0.0])
        pop = S[0, 0]+E[0, 0]+predicted_I[0, 0]+R[0, 0]

        var_dict = {
                    'day': predicted_days[0],
                    'prev_I': prev_I[0]/pop,
                    'S': S[0, 0]/pop,
                    'E': E[0, 0]/pop,
                    'I': predicted_I[0, 0]/pop,
                    'R': R[0, 0]/pop
        }
        X_input = [var_dict[feature] for feature in features_reg]

        log_beta = model.predict([X_input])
        
        beta = np.exp(log_beta)[0]
        
        predicted_beta = np.append(predicted_beta,max(beta, 0))

        full_pop = np.sum(y)
        
        for idx in range(predicted_days.shape[0]-1):
            # prediction of the Infected compartment trajectory
            S[0,:], E[0,:], predicted_I[0,idx:idx+2], R[0,:] = predict_I(
                                          I_prediction_method, y[0], 
                                          predicted_days[idx:idx+2], 
                                          predicted_beta[idx], sigma, gamma, 
                                          'det', beta_t=False)

            #print(S[0,0]+E[0,0]+predicted_I[0,idx]+R[0,0])

            #print(S[0,:], E[0,:], predicted_I[0,idx:idx+2], R[0,:] )
            
            if stochastic:
                for i in range(count_stoch_line):
                    S[i+1,:], E[i+1,:], predicted_I[i+1,idx:idx+2], R[i+1,:] = predict_I(
                                                  I_prediction_method, y[i+1], 
                                                  predicted_days[idx:idx+2], 
                                                  predicted_beta[idx], sigma, gamma, 
                                                  'stoch', beta_t=False) 
           
            y = np.array([S[:,1], E[:,1], predicted_I[:,idx+1], R[:,1]])
            y = y.T
            
            
            var_dict = {
            'day': predicted_days[idx+1],
            'S': S[0, 1]/pop,
            'E': E[0, 1]/pop,
            'I': predicted_I[0, idx+1]/pop,
            'R': R[0, 1]/pop ,
            'prev_I': predicted_I[0,idx]/pop
            }
            
            
            X_input = [var_dict[feature] for feature in features_reg]
            
            # если СЕИР предсказал где-то 0, тк у S убираем много людей,
            # то у нас будет потом None в следующих S.
            # и модель не сможет сработать. поэтому просто ставим 0
            
            if S[0,1] == 0:
                #print('breaking')
                #print( np.arange(idx, predicted_days.shape[0]-1))
                for j in range(idx, predicted_days.shape[0]-1):
                    predicted_beta = np.append(predicted_beta, 0)
                    #predicted_I
                break
            else:
                log_beta = model.predict([X_input])
                beta = np.exp(log_beta)[0]
                predicted_beta = np.append(predicted_beta, max(beta, 0))

    elif beta_prediction_method == 'lstm (day, E, previous I)':
        full_scaler = joblib.load(f'{model_path}.pkl')
        model = load_model(f'{model_path}.keras')
        predictor = LSTMPredictor(model, full_scaler, 
                                  window_size=window_size)
        '''
        prev_I = seed_df.iloc[predicted_days[0]-2:predicted_days[0]
                             ]['I'].to_numpy(
            ) if predicted_days[0] > 1 else np.array([0.0, 0.0])
        '''
        seed_df['day'] = range(len(seed_df))
        #seed_df['prev_I'] = seed_df['I'].shift(2).fillna(0)
        predicted_beta = np.empty((0,))
        S = np.zeros((count_stoch_line+1, 2))
        E = np.zeros((count_stoch_line+1, 2))
        R = np.zeros((count_stoch_line+1, 2))

        S[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['S']
        predicted_I[0:count_stoch_line+1,
                    0] = seed_df.iloc[predicted_days[0]]['I']
        R[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['R']  
        E[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['E']  
        
        pop = seed_df.iloc[0,:4].sum()
        # Initialize predictor buffer using the last 'window_size' days
        for i in range(predicted_days[0] - predictor.window_size + 1, predicted_days[0] + 1):
            row = seed_df.iloc[i]
            raw_features = [row['day'], row['E']/pop, #row['prev_I']
                           ]
            predictor.update_buffer(raw_features)
        y = np.array([S[:,0], E[:,0], predicted_I[:,0], R[:,0]])
        y = y.T
        
        for idx in range(predicted_days.shape[0]):
            predicted_beta = np.append(predicted_beta, predictor.predict_next())     
            if idx == predicted_days.shape[0]-1:
                break      
            # prediction of the Infected compartment trajectory
            S[0,:], E[0,:], predicted_I[0,idx:idx+2], R[0,:] = predict_I(I_prediction_method, y[0], 
                                    predicted_days[idx:idx+2], 
                                    predicted_beta[idx], sigma, gamma, 'det', beta_t=False)   
            if stochastic:
                for i in range(count_stoch_line):
                    S[i+1,:], E[i+1,:], predicted_I[i+1,idx:idx+2], \
                        R[i+1,:] = predict_I(I_prediction_method,
                                             y[i+1],
                                             predicted_days[idx:idx+2], 
                                             predicted_beta[idx], 
                                             sigma, gamma, 
                                             'stoch', beta_t=False) 
            y = np.array([S[:,1], E[:,1], predicted_I[:,idx+1], R[:,1]])
            y = y.T
            if idx == 0:
                predictor.update_buffer([predicted_days[idx+1], E[0,1]/pop,
                                         #prev_I[1]
                                        ])
            else:
                predictor.update_buffer([predicted_days[idx+1], E[0,1]/pop,
                                         #predicted_I[0,idx-1]
                                        ])
                
    return np.array(beggining_beta), np.array(predicted_beta), predicted_I 


def predict_I(I_prediction_method, y, 
              predicted_days, 
              predicted_beta, sigma, gamma, stype, beta_t=True):
    '''
    Predict Infected values.

    Parameters:

    - I_prediction_method -- mathematical model for predicting the Infected trajectory
        ['seir']
    - y -- compartment values on the day of switching to the mathematical model
    - predicted_days -- days for prediction
    - predicted_beta -- predicted Beta values
    - sigma -- parameter of the SEIR-type mathematical model
    - gamma -- parameter of the SEIR-type mathematical model
    - stype -- type of mathematical model
        ['stoch', 'det']
    '''
    
    
    S,E,I,R = seir_discrete.seir_model(y, predicted_days, 
                        predicted_beta, sigma, gamma, 
                        stype, beta_t).T

    return S,E,I,R