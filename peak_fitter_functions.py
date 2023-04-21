# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:44:46 2023

@author: Elizabeth Allan-Cole
"""

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy import optimize
from scipy import integrate
from scipy.signal import find_peaks
from scipy.integrate import simpson
from scipy.integrate import quad
from pathlib import Path
from os import listdir, chdir
from os.path import isfile, join
import regex as re
from lmfit import Model
from lmfit.models import LinearModel, GaussianModel, ExponentialModel, ConstantModel, PowerLawModel, PolynomialModel, LorentzianModel, VoigtModel
from lmfit.model import save_modelresult, load_modelresult
import math
import time
import itertools as it

def make_dataframe(sample_name, data_path):

    file = open(os.path.join(data_path, sample_name))
    data = pd.read_csv(file, skiprows = 1, header = None, delim_whitespace=True)
    df = pd.DataFrame(data)
    df.columns = ['q','I']
        
    return df


def get_xy_motor(sample_name, data_path):

    try:
        # Find the x_motor position in the file title using Regex
        start_x = re.search('_x_', sample_name).end()
        end_x = re.search('mm_primary', sample_name).start() 
        x_motor = sample_name[start_x:end_x].replace(',', '.')
        x_motor = float(x_motor)

        # Find the y_motor position in the file title using Regex
        start_y = re.search('_y_', sample_name).end()
        end_y = re.search('mm_sample_x', sample_name).start()
        y_motor = sample_name[start_y:end_y].replace(',', '.')
        y_motor = float(y_motor)
    
    except AttributeError:
        print('oh shit bra, the name changed! (function could not find x and y position in file name)')
        x_motor = input('Whats the x value?')
        x_motor = float(x_motor)
        
        y_motor = input('Whats the y value?')
        y_motor = float(y_motor)
        print("Groovie.")
    
    return x_motor, y_motor


def normalize_data(df):
    
    #Pull the intensity of the copper (111) peaks
    q_min = 2.95
    q_max = 3.06
    
    df_Cu = df[(df['q'] >= q_min) & (df['q'] <= q_max)]
    max_intensity = df_Cu['I'].max()
    
    #The minumum intensity value is over the full data set 
    min_intensity = df_Cu['I'].min()
    
    #df_norm = pd.DataFrame(columns = ['q','I'])
    columns = ['q','I']
    values = []
    
    for i in range(len(df)):
        #calculate the normalized intensity 
        #norm_intensity = (df['I'][i])
        #print('Data is not being normalized, turn normalization back on')
        norm_intensity = ((df['I'][i] - min_intensity) / (max_intensity - min_intensity))*1000        
        
        #append the q and new normalized intensity to a list 
        values.append([df['q'][i], norm_intensity])

    # Put the list of q and normalized intensities into a dataframe
    df_norm = pd.DataFrame(values, columns=columns)
    
    return df_norm


def get_points(df,q_min,q_max):
    
    ''' This function creates a condensed dataframe that isolates the deired peak
    Inputs: data set in data frame (df), lower q bound for peak(q_min), upper q bound for peak(q_max)
    Outputs: shortened dataframe (df_cut)'''
    df_cut = df[(df['q'] >= q_min) & (df['q'] <= q_max)]
    sliced_q = df_cut['q'].to_numpy()
    sliced_I = df_cut['I'].to_numpy()
    return sliced_q, sliced_I


def make_model(q_max, q_min, model_centers, sig, amp):
    background = LinearModel(prefix=('b' + '_'))  
    pars = background.make_params()
    
    model = background
    
    # initial guesses     
    slope1 = 0 
    int1 = 50
    
    # For linear background
    pars = background.make_params()
    pars['b' + '_slope'].set(slope1)
    pars['b' + '_intercept'].set(int1)
    
    for peak, center in enumerate(model_centers):
        # create prefex for each peak
        pref = 'v'+str(peak)+'_'
        #peak = GaussianModel(prefix=pref)
        peak = VoigtModel(prefix=pref)
        # set the parimiters for each peak
        pars.update(peak.make_params())
        pars[pref+'center'].set(value=center, min=q_min, max=q_max)
        pars[pref+'sigma'].set(value=sig, max = 0.2)
        pars[pref+'amplitude'].set(amp, min = 0)
        pars[pref+'gamma'].set(value=sig, vary=True, expr='', min = 0)
        
        model = model + peak

    return (model, pars)


def get_model_list(q_max, q_min, num_of_centers, num_peaks, sig, amp, peak_name, Li_q_max, Li_q_min):
    # set some inital parimiters if its lithium we want to narrow the range it will guess for peaks
    if peak_name == 'Li':
        temp_max = q_max
        temp_min = q_min
        q_max = Li_q_max
        q_min = Li_q_min
    # generate a list of centers to try
    increment = (q_max - q_min) / num_of_centers
    n = 0
    center_list = []
    
    while n <= num_of_centers:
        center_list.append(n*increment + q_min)
        n += 1
    q_range = q_max - q_min
    
    if peak_name != 'Li':
        center_list[0] = center_list[0] + .1 * q_range
        # -1 refers to the last element in the list
        center_list[-1] = center_list[-1] - .1 * q_range
    
    # creat unique combination of peak positions returns a list of tuples. 
    # Tuples are samp length of num_peaks
    center_list = list(it.combinations(center_list, num_peaks))
    
    # if its lithium we now need to reset the q max/mmin so the model will look at the whole range
    if peak_name == 'Li':
        q_max = temp_max
        q_min = temp_min
    
    # make a list of models for each center
    model_list = []
    for center in center_list:
        model_list.append(make_model(q_max, q_min, center, sig, amp))
    
    return(model_list)  

def run_model(sliced_q, sliced_I, model, pars):
    model_result = model.fit(sliced_I, pars, x = sliced_q, nan_policy = 'omit')
    return(model_result)

def make_center_list(center, sig): #pass in sig for imporved fit

    center_low = center - (sig*1)
    center_high = center + (sig*1)
    center_list = [center_low, center, center_high]
    
    return center_list

def iterate_centers(center_list):
    # Iterate through center options --> [(a1, a2, a3),(b1, b2, b3)] to [(a1, b1), (a1, b2)...(a3, b3)]       
    target_center_list = list(it.product(*center_list))
    new_target_center_list = []
    for i in range(len(target_center_list)):
        my_list = list(target_center_list[i])
        new_target_center_list.append(my_list)
        
    #print(new_target_center_list)
    #print(type(new_target_center_list[0]))
        
    return new_target_center_list

def make_target_model(q_max, q_min, model_centers, sig_list, amp_list):
    background = LinearModel(prefix=('b' + '_'))  
    pars = background.make_params()
    
    model = background
    
    # initial guesses     
    slope1 = 0 
    int1 = 50
    
    # For linear background
    pars = background.make_params()
    pars['b' + '_slope'].set(slope1)
    pars['b' + '_intercept'].set(int1)
    
    index = 0
    for peak, center in enumerate(model_centers): 
        # create prefex for each peak
        pref = 'v'+str(peak)+'_'
        #peak = GaussianModel(prefix=pref)
        peak = VoigtModel(prefix=pref)
        # set the parimiters for each peak
        pars.update(peak.make_params())
        pars[pref+'center'].set(value=center, min=q_min, max=q_max)
        pars[pref+'sigma'].set(value=sig_list[index], max = 0.2)
        pars[pref+'amplitude'].set(amp_list[index], min = 0) #THIS IS APPARENTLY THE AREA
        pars[pref+'gamma'].set(value=sig_list[index], vary=True, expr='', min = 0)
        pars[pref+'height'].set(value=100, vary=True, expr='', max = 500)
        print('the height did not error')
        index += 1
        
        model = model + peak

    return (model, pars)

def targeted_model(new_center_list, sig_list, amp_list, q_max, q_min, sliced_q, sliced_I):
        model_list = []
        for i in range(len(new_center_list)):
            center_combo = new_center_list[i]
            for center in center_combo:
                model_list.append(make_target_model(q_max, q_min, center_combo, sig_list, amp_list))
    
        
        model_result_list = []
        for model in range(len(model_list)):
            model = model_list[i][0]
            pars = model_list[i][1]
            model_result_list.append(run_model(sliced_q, sliced_I, model, pars))
        
        results_sorted = sorted(model_result_list, key=lambda model: model.chisqr)
        best_model = results_sorted[0]
        chisqr = best_model.chisqr
        print('chi squared: ' + str(chisqr))
        
        return best_model


def user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name):
    good = 'n'
    print("\n\nfit not found")
    print('The chisqr is ', best_model.chisqr)
    show_params = False
    if show_params == True:
        model_param_list = []
        comps = best_model.eval_components(x=sliced_q)
        # get centers from the best model
        for prefex in comps.keys():
            if prefex != 'b_':
                model_param_list.append([best_model.params[str(prefex)+'center'].value, best_model.params[str(prefex)+'amplitude'], best_model.params[str(prefex)+'sigma']])
        
        print(model_param_list)
    
    best_values = best_model.best_values

    plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name)
    #best_model.plot()
    plt.pause(1)
    
    good = input('if its good enter y\n')

    while good != 'y':  
        try:
            centers =  input('Enter peak centers separated by comma \n')
            amp_list = input('Enter amplitude of peaks separated by comma (150 ~ 5)\n')
            sig_list = input('Enter the approximate standard deviations separated by comma (~0.005) \n')
            centers = centers.split(',')
            amp_list = amp_list.split(',')
            sig_list = sig_list.split(',')
            
            center_list = []
            
            for i in range(len(centers)): 
                centers[i] = float(centers[i])
                amp_list[i] = float(amp_list[i])
                sig_list[i] = float(sig_list[i])
                get_center_list = make_center_list(centers[i], sig_list[i])
                center_list.append(get_center_list) 
            
            
            #need to iterate through just the centers 
            new_center_list = iterate_centers(center_list)
    
            best_model = targeted_model(new_center_list, sig_list, amp_list, q_max, q_min, sliced_q, sliced_I)
                
            #best_model.plot()
            plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name)
            plt.pause(1)
            
            if chisqu_fit_value >= best_model.chisqr:
                good = 'y'
            else: 
                good = input('enter y to continue. To try again enter n.\n')
        
        except:
            print('operation filed with the following messege')
            print('Note for Ben. Add function so this prints error message. Also Hope your science is going well!')
    
    return best_model


def fit_data(sliced_q, sliced_I, q_max, q_min, num_of_centers, sig, amp, chisqu_fit_value, Li_q_max, Li_q_min, x_motor, y_motor, peak_name):
    chisqr = 1000000000
    num_peaks = 1
    more_peaks = False
    #assign the max number of peaks allowed (1 plus that number so if there can be 3 peaks put 4 here)
    if peak_name == 'NMC':
        max_peak_allowed = 2
    elif peak_name == 'Li':
        max_peak_allowed = 2
    else:
        max_peak_allowed = 3
    
    while chisqr >= chisqu_fit_value:

        if more_peaks is True and num_peaks >= max_peak_allowed:
            #print("TURN THE USER FIT BACK ON")
            best_model = user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name)
            print('chi squared: ' + str(chisqr))
            return best_model
        
        if num_peaks >= max_peak_allowed:
            num_peaks = 1
            num_of_centers = num_of_centers*2
            more_peaks = True
            print("THE THING HAPPENED MORE PEAKS")
 
            
        # returns a list of tuples. first value is the model second value is the pars. 
        # looks like this ((model, pars), (model, pars), ........)
        model_list = get_model_list(q_max, q_min, num_of_centers, num_peaks, sig, amp, peak_name,
                                    Li_q_max, Li_q_min)
        
        model_result_list = []

        for i in range(len(model_list)):
            model = model_list[i][0]
            pars = model_list[i][1]
            model_result_list.append(run_model(sliced_q, sliced_I, model, pars))
        
       # model_result_list = list(map(lambda model: run_model(sliced_q, sliced_I, model[0], model[1]), model_list))  
       # model_result_list = [run_model(sliced_q, sliced_I, model[0], model[1]) for model in model_list]
        
        results_sorted = sorted(model_result_list, key=lambda model: model.chisqr)
        best_model = results_sorted[0]
        chisqr = best_model.chisqr
        #print('chi squared: ' + str(chisqr))
        num_peaks += 1
        
        model_center_list = []
        comps = best_model.eval_components(x=sliced_q)
        
        # get centers from the best model
        for prefex in comps.keys():
            if prefex != 'b_':
                model_center_list.append(best_model.params[str(prefex)+'center'].value)
        
        # if the centers are too close together, the model likely miss fit 1 peak as 2
        # this sets the model to call the user fit if the centers are too close together 
        if len(model_center_list) > 1:
            center_dif = float(model_center_list[1]) - float(model_center_list[0])
            
            if center_dif < 0.01:
                max_peak_allowed = max_peak_allowed - 1
                center = (float(model_center_list[1]) + float(model_center_list[0])) / 2
                center = [center]
                (model, pars) = make_model(q_max, q_min, center, sig, amp)
                
                best_model = run_model(sliced_q, sliced_I, model, pars)
                plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name)
                plt.pause(1)
                
                print('Reduced the number of peaks, new chi sqrd: ' + str(chisqr))
                
                if chisqr > chisqu_fit_value:
                    go_to_user_model = input('Is the fit good (enter y)? If no, go to the user model (enter n) \n')
                    if go_to_user_model != 'y':
                        best_model = user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name)
                        plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name)

                return best_model
                
    
    plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name)
    #best_model.plot()

    plt.pause(1)
    return best_model


def get_values(best_model, sliced_q, sliced_I):
         
    # a list of tuples with 4 values. the peak data, fwhm, and center.
    # Looks like ((peak_data, fwhm, center, guess), (peak_data, fwhm, center, guess), ........)
    comps_list = []
    #print(best_model.fit_report())
 #   
 #   print("")
    comps = best_model.eval_components(x=sliced_q)
    
  #  ax.plot(x,best_model, label='Model')
    for prefex in comps.keys():
        if prefex != 'b_':
            comps_list.append(((comps[str(prefex)]), best_model.params[str(prefex)+'fwhm'].value, best_model.params[str(prefex)+'center'].value, 1.75))
    
    integral_list = []
    fwhm_list = []
    peak_center_list = []
    
    for vals in comps_list:
        integral_val = integrate_model(sliced_q, sliced_I, vals[0], vals[2], vals[3])
        integral_list.append(integral_val)
        # get_fwhm_center function not needed
        # fwhm_list, peak_center_list = get_fwhm_center(integral_val, vals[1], vals[2], vals[3])
        fwhm_list.append(vals[1])
        peak_center_list.append(vals[2])
        
    return integral_list, fwhm_list, peak_center_list


def integrate_model(sliced_q, sliced_I, Gaussian, center_raw, q_guess):
    
    # Define model
    model = Gaussian
    
    # Select the data to integrate over
    #q_range = df_cut['q'].tolist()
    

    # Caclulate the integral based on the direct data using Simpson's rule
    integral = integrate.simpson(model, sliced_q, even='avg')
    return integral

    
def plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name):
    
    comps = best_model.eval_components(x=sliced_q)
    
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    
    ax.scatter(sliced_q,sliced_I, label='Data', color='black')  
    ax.plot(sliced_q,best_model.best_fit, label='Model', color='gold')
    for prefix in comps.keys():
        ax.plot(sliced_q, comps[prefix], '--', label=str(prefix))

    ax.set_title(str(peak_name) + ' : (' + str(x_motor) + ',' + str(y_motor) + ')') 
    ax.set_xlabel('q [1/A]')
    ax.set_ylabel('I [au.]')
    ax.legend()  


def master_function(read_sample_file, num_of_centers,  data_path, q_min, q_max,  sample_name, sig, amp, chisqu_fit_value, peak_name, Li_q_max, Li_q_min, plot):
    
    # Make a dataframe of the entire XRD pattern
    df = make_dataframe(read_sample_file, data_path)
    
    # Normalize data
    #df_norm = normalize_data(df)
    df_norm = df
    
    # Get xy_motor positions
    x_motor, y_motor = get_xy_motor(read_sample_file, data_path)
    
    # Slice the dataframe to desired q range
    sliced_q, sliced_I = get_points(df_norm, q_min, q_max)

    # get the best fit for the data
    best_model = fit_data(sliced_q, sliced_I, q_max, q_min, num_of_centers, sig, amp, chisqu_fit_value, Li_q_max, Li_q_min, x_motor, y_motor, peak_name)

    if best_model is not None:
        integral_list, fwhm_list, peak_center_list = get_values(best_model, sliced_q, sliced_I)
    else:
        return sample_name, x_motor, y_motor
    
    if plot == True:
        plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name)
    
    return [sample_name, x_motor, y_motor, integral_list, fwhm_list, peak_center_list, best_model, sliced_q]


def save_fits(savePath_gen, get_integrals, element, list_of_files, i, sample_name):
  
    # find the cordanets of the sample and get rid of the dots file paths don't like that
    coordinates = (str(get_integrals[1]) + '_' + str(get_integrals[2])).replace('.', '-')
    # make it a file path
    savePath = os.path.join(savePath_gen, sample_name, element, coordinates)
    
    # if that foulder dosn't exist make it exist
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # name the file
    #y = str(i)
    #file_name = str(list_of_files[i])
    #file_name = file_name.replace(",", "_")
    #file_name = file_name[:len(file_name) - 5]
    file_name = sample_name
    fig_path = os.path.join(savePath, file_name)
    # save the file! that wasn't at all convaluded was it?
    get_integrals[6].plot().savefig(fig_path)
    plt.close()
    
    #save the model fit
    os.chdir(savePath)
    best_model = get_integrals[6]
    save_modelresult(best_model, sample_name)
        
    return savePath



    