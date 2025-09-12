import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
#from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# our functions
import seir_discrete 

import warnings
warnings.filterwarnings(action='ignore')


def load_saved_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


class LSTMPredictor:
    """
    Wraps the trained LSTM model to predict beta on a rolling window of
    [day, E] (2 features). 
    The model was trained to predict normalized log_beta, so this class
    denormalizes the prediction and returns beta.
    """
    def __init__(self, model, full_scaler, window_size):
        self.model = model
        self.n_feats = 1
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
            
        scaled = self.input_scaler.transform(padded) # (4,1)
        # (1,4,1)
        scaled_window = scaled.reshape(1, self.window_size, self.n_feats)
        
        normalized_pred = self.model.predict_on_batch(scaled_window)[0][0]
        # Denormalize to obtain the raw log_beta
        raw_log_beta = normalized_pred * self.target_scale + self.target_mean
        # Compute beta by exponentiating the log_beta
        predicted_beta = np.exp(raw_log_beta)
        return predicted_beta

    
def predict_beta(I_prediction_method, seed_df, beta_prediction_method, predicted_days, 
                 stochastic=False, count_stoch_line=0, 
                 sigma=0, gamma=0, 
                 features_reg='', model_path='', window_size=14,modeling_duration=149):
    
    '''
    Predict Beta values.

    Parameters:

    - I_prediction_method -- mathematical model for predicting Infected trajectories
        ['seir']
    - seed_df -- DataFrame of seed, created by a regular network
    - beta_prediction_method -- method for predicting Beta values
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

        
    elif beta_prediction_method == 'median beta':
        betas = pd.read_csv(model_path) #'train/median_beta.csv'
        beggining_beta = betas.iloc[:predicted_days[0]+1,-1].values
        predicted_beta = betas.iloc[predicted_days[0]:,-1].values
    
    
    elif beta_prediction_method == 'lstm':
        full_scaler = joblib.load(f'{model_path}.pkl')
        model = load_model(f'{model_path}.keras')
        window_size=4
        predictor = LSTMPredictor(model, full_scaler, 
                                  window_size=window_size)
        inp = seed_df[['Beta']
                     ].shift(np.arange(window_size)
                            ).iloc[predicted_days[0]].values
        inp = np.log(inp+1e-7)
        inp = np.nan_to_num(inp, neginf=0, posinf=0)
        
        for i in inp[::-1]:
            predictor.update_buffer([i])
        #print(predictor.buffer)
        
        predicted_beta = []
        for i in range(predicted_days[0], 
                       modeling_duration):
            pred = predictor.predict_next()
            #print(pred)
            if (pred<0) or (pred == np.inf):
                pred = 0
            predicted_beta.append(pred)
            predictor.update_buffer([np.log(pred+1e-7)])
                
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