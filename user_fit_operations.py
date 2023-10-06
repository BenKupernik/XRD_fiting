# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:43:40 2023

@author: Elizabeth Allan-Cole
"""

import peak_fitter_functions as pf
from matplotlib import pyplot as plt
from lmfit import Model
from lmfit.models import LinearModel, GaussianModel, ExponentialModel, ConstantModel, PowerLawModel, PolynomialModel, LorentzianModel, VoigtModel
import itertools as it
from joblib import Parallel, delayed
import numpy as np
import scipy
from scipy.signal import find_peaks


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

def make_target_model(q_max, q_min, model_centers, sig_list, amp_list, b_slope):
    background = LinearModel(prefix=('b' + '_'))  
    pars = background.make_params()
    
    model = background
    
    # initial guesses     
    slope1 = (float(b_slope[0]) + float(b_slope[1]))/2
    int1 = 50
    
    # For linear background
    pars = background.make_params()
    pars['b' + '_slope'].set(slope1, vary = True, min = float(b_slope[0]), max = float(b_slope[1]))
    pars['b' + '_intercept'].set(int1, vary = True)
    
    index = 0
    for peak, center in enumerate(model_centers): 
        # create prefex for each peak
        pref = 'v'+str(peak)+'_'
        #peak = GaussianModel(prefix=pref)
        peak = VoigtModel(prefix=pref)
        # set the parimiters for each peak
        pars.update(peak.make_params())
        #pars[pref+'center'].set(value=center, min=q_min, max=q_max)
        pars[pref+'center'].set(value=center, min= center - 0.025, max= center + 0.025)
        pars[pref+'sigma'].set(value=sig_list[index], max = sig_list[index] * 2)
        pars[pref+'amplitude'].set(amp_list[index], min = 0, max = amp_list[index] * 2) #THIS IS APPARENTLY THE AREA
        pars[pref+'gamma'].set(value=sig_list[index], vary=True, expr='', min = 0)
        index += 1
        
        model = model + peak

    return (model, pars)

def make_linear_model(q_max, q_min, b_slope):
    background = LinearModel(prefix=('b' + '_'))  
    pars = background.make_params()
    
    model = background
    
    # initial guesses     
    slope1 = (float(b_slope[0]) + float(b_slope[1]))/2
    int1 = 50
    
    # For linear background
    pars = background.make_params()
    pars['b' + '_slope'].set(slope1, vary = True, min = float(b_slope[0]), max = float(b_slope[1]))
    pars['b' + '_intercept'].set(int1, vary = True, min = 0)
    
    return (model, pars)


def targeted_model(new_center_list, sig_list, amp_list, q_max, q_min, sliced_q, sliced_I, b_slope):
    model_list = []
    for i in range(len(new_center_list)):
        center_combo = new_center_list[i]
        for center in center_combo:
            model_list.append(make_target_model(q_max, q_min, center_combo, sig_list, amp_list, b_slope))
    
    model_result_list = []
    model_result_list = Parallel(n_jobs=2)(delayed(pf.run_model)(sliced_q, sliced_I, model[0], model[1])for model in model_list)
    
    results_sorted = sorted(model_result_list, key=lambda model: model.chisqr)
    best_model = results_sorted[0]
    chisqr = best_model.chisqr
    print('chi squared: ' + str(chisqr))
    
    return best_model

# def get_peak_style(sliced_I):

#     peaks, properties = scipy.signal.find_peaks(sliced_I, prominence = (1, None))
    
#     if properties['prominences'][0] > properties['prominences'][1]:
#         peak_style = 'BigSmall'
#     elif properties['prominences'][1] > properties['prominences'][0]:
#         peak_style = 'SmallBig'
#     else: 
#         peak_style = 'unknown'
     
#     return peak_style
    


def user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot):
    good = 'n'
    print("\n\nfit not found")
    print('The chisqr is ', best_model.chisqr)
    print('\n\nOriginal Fit Report: \n\n', best_model.fit_report())
    
    pf.plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
    #plt.pause(1)
   
    peaks, properties = scipy.signal.find_peaks(sliced_I, prominence = (1, None))
    print(properties['prominences'])
    len_prominence = len(properties['prominences'])
    if len_prominence != 0:
        val_promenence = max(properties['prominences']) 
    
    if len_prominence == 0 or val_promenence < 0.75:
        
        if peak_name == 'Graphite-LiC12':
            b_slope = '-150, 0'
        elif peak_name == 'Li':
            b_slope = '0, 150'
        else: 
            b_slope = input('Enter background slope min and max separated by comma \n') 
            
           
        b_slope = b_slope.split(',')
        model = make_linear_model(q_max, q_min, b_slope)
        best_model = pf.run_model(sliced_q, sliced_I, model[0], model[1])
        chisqr = best_model.chisqr
        if chisqr <= 3 * chisqu_fit_value: 
            return best_model
    
    print('\nPeak class options: "y"- GoodFit , "1"-BigSmall, "2"-SmallBig,"3"-OneBig, "4"-OneSmall , "5"-Line, "6"-TwoSmall ,"7"-TwoTogether , "8"-Li-NMC,"9"-NMC-no-Li ,"10"-3peak,"n"- other \n')
    peak_style = input('Is the fit good? If yes enter "y", otherwise enter peak class to fit. \n')

    while peak_style != 'y':  
        # try:
        
        #peak_style = get_peak_style(sliced_I)
        #print('Peak class options: "1"-BigSmall, "2"-SmallBig,"3"-OneBig, "4"-OneSmall , "5"-Line, "6"-TwoSmall ,"7"-TwoTogether , "8"-Li-NMC,"9"-NMC-no-Li ,"10"-3peak-Li,"n"- other \n')
        #peak_style = input('Enter peak class: \n')
        
        # Need to add in automation to go back to being able to fit these if necessary 
        # Another note, need to add in function to go back on and only correct one of the input parameters
        
        if peak_name == 'Graphite-LiC12':
            b_slope = '-150, 0'
        elif peak_name == 'LiC6':
            b_slope = '-450, -90'
        elif peak_name == 'Li':
            b_slope = '0, 150'
        else: 
            b_slope = input('Enter background slope min and max separated by comma \n')    
        
        peaks, properties = find_peaks(sliced_I, prominence = (1, None))
        
        if peak_style == '1': # "BigSmall" One larger peak with a smaller detached peak at a higher q
            #center_list should be the q value corresponding to peaks(peaks is the index for the peaks found with find_peaks using peak prominence)
            centers = np.take(sliced_q, peaks)
            if len(centers) != 2:
                centers =  input('Enter peak centers separated by comma \n')
            amp_list = '3, 0.104'
            sig_list = '0.005, 0.0031'
        
        elif peak_style == '2': # "SmallBig" One smaller peak, then a larger peaks at a higher q
            centers = np.take(sliced_q, peaks)
            if len(centers) != 2:
                centers =  input('Enter peak centers separated by comma \n')
            amp_list = '0.2, 3'
            sig_list = '0.0031, 0.005'
            
        elif peak_style == '3':
            # Not fitting really big peaks
            centers = np.take(sliced_q, peaks)
            if len(centers) > 1:
                center_index = peaks[np.argmax(properties['prominences'])]
                centers = [np.take(sliced_q, center_index)]
            
            if peak_name == 'LiC6':
                amp_list = '1.45'
                sig_list = '0.019'
                print('Center Guess: ', centers)
            else: 
                amp_list = '5'
                sig_list = '0.005'
                
            
        elif peak_style == '4':
            # Doesn't look like it is setup to fit really small peaks as one small
            centers = np.take(sliced_q, peaks)
            if len(centers) > 1:
                center_index = peaks[np.argmax(properties['prominences'])]
                centers = [np.take(sliced_q, center_index)] 
            amp_list = '5'
            sig_list = '0.005'
        
        elif peak_style == '5':
            b_slope = b_slope.split(',')
            model = make_linear_model(q_max, q_min, b_slope)
            best_model = pf.run_model(sliced_q, sliced_I, model[0], model[1])
            #print(best_model.fit_report())
            #pf.plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)

            return best_model
        
        elif peak_style == '6':
            
            #center_list should be the q value corresponding to peaks(peaks is the index for the peaks found with find_peaks using peak prominence)
            centers = np.take(sliced_q, peaks)
            #centers =  input('Enter peak centers separated by comma \n')
            amp_list = '0.1, 0.1'
            sig_list = '0.003, 0.003'
            # need to add condition for peaks being too close, maybe add min and max closer to centers
                
        elif peak_style == '7':
            centers = np.take(sliced_q, peaks)
            
            if peak_name == 'LiC6':
                amp_list = '2, 0.75'
                sig_list = '0.015, 0.003'
                

                if len(centers) == 1:
                    center = float(centers[0])
                    centers = [center - 0.01, center + 0.01]
                elif len(centers) != 1:
                    centers =  input('Enter peak centers separated by comma \n')
            
            elif peak_name == 'Graphite-LiC12':
                amp_list = '1, 1'
                sig_list = '0.003, 0.003'
                
                if len(centers) == 1:
                    center = float(centers[0])
                    centers = [center - 0.01, center + 0.01]
                elif len(centers) != 1:
                    centers =  input('Enter peak centers separated by comma \n')

            else: 
                print('Fuck!')
        
        
        elif peak_style == '8':
            if peak_name =='Li':
                last_I = sliced_I[-1]
                max_I = max(sliced_I)               
                if max_I == last_I: 
                    amp_list = '0.2, 5'
                    sig_list = '0.003,0.005'
                    centers = [2.535, 2.605]
            else:
                centers =  input('Enter peak centers separated by comma \n')
                amp_list = input('Enter area (amplitude) of peaks separated by comma (~5 graphite)\n')
                sig_list = input('Enter the approximate standard deviations separated by comma (~0.005 graphite) \n')
                # centers = centers.split(',')
                # for i in range(len(centers)): 
                #     centers[i] = float(centers[i])
                    
        elif peak_style == '9':
            if peak_name =='Li':
                last_I = sliced_I[-1]
                max_I = max(sliced_I)               
                if max_I == last_I: 
                    amp_list = '5'
                    sig_list = '0.005'
                    centers = [2.605]
                                 
        elif peak_style == '10':
            if peak_name =='Li':
                amp_list = '1, 5, 3'
                sig_list = '0.003,0.005, 0.005'
                centers = [2.535, 2.56, 2.57]
            else:
                centers =  input('Enter peak centers separated by comma \n')
                amp_list = input('Enter area (amplitude) of peaks separated by comma (~5 graphite)\n')
                sig_list = input('Enter the approximate standard deviations separated by comma (~0.005 graphite) \n')
                # centers = centers.split(',')
                # for i in range(len(centers)): 
                #     centers[i] = float(centers[i])
        else:
            #b_slope = input('Enter background slope min and max separated by comma \n') 
            centers =  input('Enter peak centers separated by comma \n')
            amp_list = input('Enter area (amplitude) of peaks separated by comma (~5 graphite)\n')
            sig_list = input('Enter the approximate standard deviations separated by comma (~0.005 graphite) \n')
            # centers = centers.split(',')
            # for i in range(len(centers)): 
            #     centers[i] = float(centers[i])
        
        
        # NEED TO CHECK THIS!!!
        if type(centers[0]) == str:
            centers = centers.split(',')
            for i in range(len(centers)): 
                centers[i] = float(centers[i])
        
        amp_list = amp_list.split(',')
        sig_list = sig_list.split(',')
        b_slope = b_slope.split(',')
        
        center_list = []
        
        for i in range(len(centers)): 
            amp_list[i] = float(amp_list[i])
            sig_list[i] = float(sig_list[i])
            get_center_list = make_center_list(centers[i], sig_list[i])
            center_list.append(get_center_list) 
        
        
        #need to iterate through just the centers 
        new_center_list = iterate_centers(center_list)
        best_model = targeted_model(new_center_list, sig_list, amp_list, q_max, q_min, sliced_q, sliced_I, b_slope)
            
        #best_model.plot()
        pf.plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
        plt.pause(1)
        
        #print('\n\nNew Fit Report: \n\n', best_model.fit_report())
        
        if best_model.chisqr <= chisqu_fit_value * 10: # change back to 2
            peak_style = 'y'
        else: 
            peak_style = input('enter "y" to continue. \n To try again enter "n" to input paramters manually or enter peak class # from above.\n')
        
        # except:
        #      print('operation filed with the following messege')
        #      print('Note for Ben. Add function so this prints error message. Also Hope your science is going well!')
    
    return best_model