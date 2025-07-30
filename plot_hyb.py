import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import root_mean_squared_error as rmse
import time
import tkinter as tk
from tkinter import messagebox
import math
import matplotlib as mpl
from scipy.spatial import ConvexHull

# our functions
import predict_Beta_I
import choice_start_day

import warnings
warnings.filterwarnings(action='ignore')


def plot_one(ax, 
             predicted_days, seed_df, predicted_I, 
             beggining_beta, predicted_beta,
             seed_number, execution_time):
    '''
    Plotting the graph for a seed.
    
    Parameters:

    - ax -- area for the plot
    - predicted_days -- predicted days
    - seed_df -- DataFrame of seed, created by the regular network
    - predicted_I -- predicted trajectory of the Infected compartment
    - beggining_beta -- predicted initial values of Beta
    - predicted_beta -- predicted values of Beta
    - seed_number -- seed number        
    - execution_time -- time taken to predict Beta   
    - median_values -- sample mean of predicted_I on a specific day
    - lower_bound -- upper boundary of the interval (3 std of predicted_I on a specific day)
    - upper_bound -- lower boundary of the interval (3 std of predicted_I on a specific day)
    '''

    # when shifting forecasts, sometimes NaN values appear here
    predicted_I[np.isnan(predicted_I)] = 0.0  
    predicted_beta[np.isnan(predicted_beta)] = 0.0
    beggining_beta[np.isnan(beggining_beta)] = 0.0
 
    # find the maximum and its index
    predicted_peak_I = max(predicted_I[0])
    predicted_peak_day = predicted_days[0] + np.argmax(predicted_I[0])
    actual_I = seed_df.iloc[:]['I'].values 
    actual_peak_I = max(actual_I)
    actual_peak_day = np.argmax(actual_I)+1

    peak = [actual_peak_I, predicted_peak_I,actual_peak_day,predicted_peak_day] 

    # calculate RMSE for Infected and Beta values
    actual_I = seed_df.iloc[predicted_days[0]:]['I'].values 
    rmse_I = rmse(actual_I, predicted_I[0])
    
    actual_Beta = seed_df.iloc[predicted_days[0]:]['Beta'].values 
    predicted_beta = predicted_beta[:actual_Beta.shape[0]]
    rmse_Beta = rmse(actual_Beta, predicted_beta)   

    # display boundary of switch 
    ax.axvline(predicted_days[0], color='red',ls=':')

    if predicted_I.shape[0] > 1:
        # display trajectories of the stochastic mathematical model
        for i in range(predicted_I.shape[0]-1):
            ax.plot(predicted_days, predicted_I[i+1], color='tab:orange', ls='--', 
                    alpha=0.3, label='Predicted I (stoch.)' if i == 0 else '')

        # median calculation
        median_values = np.median(predicted_I, axis=0) 
        # standard error
        std_dev = np.std(predicted_I, axis=0)
        # boundaries: median ± 3σ (checked for negative values)
        lower_bound = median_values - 3 * std_dev
        upper_bound = median_values + 3 * std_dev
        lower_bound = np.maximum(lower_bound, 0)
        # add vertical lines with tick marks for confidence intervals
        for day in range(0, len(predicted_days), 5): 
            ax.errorbar(predicted_days[day], median_values[day],
                        yerr=[[median_values[day] - lower_bound[day]], 
                            [upper_bound[day] - median_values[day]]], 
                        fmt='o', color='black', capsize=2, markersize=2, elinewidth=1, 
                        alpha=0.6, label='$\mu \pm 3\sigma$' if day == 0 else '')

    # display actual and predicted Infected values
    ax.plot(seed_df.index, seed_df.iloc[:]['I'].values , color='tab:blue', 
            label='Actual I')
    ax.plot(predicted_days, predicted_I[0],color='red', ls='-', 
              alpha=0.9, label='Predicted I (det.)')
    
    # add axis labels
    ax.set_xlabel('Days')
    ax.set_ylabel('Infected', color='tab:blue')
    ax.grid(True, alpha=0.3)
        
    ax_b = ax.twinx()
    # display actual and predicted Beta values
    ax_b.plot(seed_df.index, seed_df['Beta'],  color='gray', ls='--', 
              alpha=0.4, label='Actual Beta')

    if len(beggining_beta) > 0:
        given_days = np.arange(predicted_days[0])
        ax_b.plot(given_days, beggining_beta,color='green', ls='--', 
                  alpha=0.7, label='Predicted Beta ')
    ax_b.plot(predicted_days, predicted_beta,color='green', ls='--', 
              alpha=0.7, label='Predicted Beta ')
    ax_b.set_ylabel("Beta", color='gray')

    # add legend and titles
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.set_title(f'Seed {seed_number}, Switch day {predicted_days[0]}\n'+
                 f'Peak I (act.):{actual_peak_I:.2f}, '+
                   f'Peak day (act.):{actual_peak_day:.2f}, \n' +
                 f'Peak I (pred.):{predicted_peak_I:.2f}, '+
                   f'Peak day (pred.):{predicted_peak_day:.2f}, \n' +
                 f'RMSE I:{rmse_I:.2f}, RMSE beta:{rmse_Beta:.2e}, \n'+
                 f'Predict time: {execution_time:.2e}' ,fontsize=10)
    return rmse_I, rmse_Beta, peak


def main_f(I_prediction_method, stochastic, count_stoch_line, 
           beta_prediction_method, type_start_day, seed_numbers,
           show_fig_flag, seed_dirs='test/', sigma=0.1, gamma=0.08,
           ax = None, model_path=''):
    '''
    Main function
    
    Parameters:
    
    - I_prediction_method -- model for constructing the trajectory of Infected
        ['seir']
    - stochastic -- presence of predicted stochastic trajectories of Infected 
    - count_stoch_line -- number of predicted stochastic trajectories
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
    - type_start_day -- type of choosing the switching day for the model 
        (changing or constant)
        ['roll_var', 'norm_var', 'roll_var_seq', 'roll_var_npeople', 
         40, 50, 60]
    - seed_numbers -- seed numbers for the experiments
    - show_fig_flag -- flag to show the plots
    - save_fig_flag -- flag to save the plots
    
    Output:
        Graph for seeds.
    '''
    features_reg = ''
    if ax is None:
        row_n = len(seed_numbers)//2+math.ceil(len(seed_numbers)%2)
        fig, axes = plt.subplots(row_n, 2, figsize=(13, 3.5*row_n))
        axes = axes.flatten()
    else:
        axes = ax

    print(beta_prediction_method, model_path)
    # list of RMSE Beta and I for each seed 
    all_rmse_I = []
    all_rmse_Beta = []
    all_peak = []
    start_days = []
    execution_time = []
    
    for idx, seed_number in enumerate(seed_numbers):
        #print(seed_number)
        # read the DataFrame of the seed: S,[E],I,R,Beta
        seed_df = pd.read_csv(seed_number)
        seed_df = seed_df.iloc[:,:5].copy()
        seed_df.columns = ['S','E','I','R','Beta']
        
        #seed_df = seed_df[pd.notna(seed_df['Beta'])]
        end_df  = seed_df[(seed_df.E==0)&(seed_df.I==0)]
        if end_df.shape[0]:
            seed_df = seed_df.iloc[:end_df.index[0]].copy()
        
        if seed_df['I'].max() < 10000:
            pass
        else:
            
            # switch moment
            
            start_day = choice_start_day.choose_method(seed_df, type_start_day)
            # ЗА сколько ДО пика
            if not isinstance(type_start_day, str):
                start_day = seed_df.I.argmax() - start_day

            start_days.append(start_day)
            # choosing the days for prediction
            predicted_days = np.arange(start_day, seed_df.shape[0])
            start_time = time.time()

            # prediction of Beta values and calculation of prediction time
            beggining_beta, predicted_beta, predicted_I = predict_Beta_I.predict_beta(
                                I_prediction_method, seed_df, beta_prediction_method, 
                                predicted_days, stochastic, count_stoch_line, sigma, gamma,
                                features_reg, model_path)

            if (beta_prediction_method != 'regression (day, SEIR, previous I)') & (
                beta_prediction_method != 'lstm (day, E, previous I)'):
                 # extract compartment values on the switch day
                y = seed_df.iloc[predicted_days[0],:4]

                # predict the Infected compartment trajectory
                _,_,predicted_I[0],_ = predict_Beta_I.predict_I(I_prediction_method, y, 
                                                                predicted_days-start_day,
                                                                predicted_beta,
                                                                sigma, gamma, 'det')
                if stochastic:
                    for i in range(count_stoch_line):
                        _,_,predicted_I[i+1],_ \
                            = predict_Beta_I.predict_I(I_prediction_method,
                                                       y, predicted_days-start_day, 
                                                       predicted_beta,sigma, gamma, 
                                                       'stoch')

            end_time = time.time()
            execution_time.append(end_time - start_time)

            if ax is None:
                # plot graph for seed_number
                rmse_I, rmse_Beta, peak = plot_one(axes[idx], 
                                                   predicted_days, seed_df, predicted_I, 
                                                   beggining_beta, predicted_beta, 
                                                   seed_number, end_time - start_time)        
                all_rmse_I.append(rmse_I)
                all_rmse_Beta.append(rmse_Beta)
                all_peak.append(peak)

        
    
    if ax is None:
        # add overall title
        '''
        fig.suptitle(f'I_prediction_method:{I_prediction_method}, \n'+
                    f'beta_prediction_method: {beta_prediction_method}, \n' +
                    f'Switching days type:{type_start_day}' ,fontsize=15)
        
        if len(seed_numbers)%2 == 1:
            fig.delaxes(axes[-1]) 
        '''
        plt.tight_layout()
        
        # show the plots
        if show_fig_flag:
            plt.show()
        else:
            plt.close(fig)

        return all_rmse_I, all_rmse_Beta, all_peak, execution_time, start_days
    else :
        return plot_one(axes, predicted_days, seed_df, 
               predicted_I, beggining_beta, predicted_beta, 
               seed_number, end_time - start_time)
        #return predicted_days, seed_df, predicted_I, beggining_beta, predicted_beta, seed_number, end_time - start_time

