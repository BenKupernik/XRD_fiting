# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:43:36 2023

@author: Elizabeth Allan-Cole
"""
import peak_fitter_functions as pf
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
import math
import time
import itertools as it


startTime = time.time()

# Sample info
sample_name = 'S1_LN_10psi_Ch10_0120922_map_01-4'
plot = True

# numper of centers to try
num_of_centers = 8

# the range you want lmift to use for centers for a lithium peak. 
Li_q_max = 2.525
Li_q_min = 2.54

#Setup dataframe 
df_integrals = pd.DataFrame(columns=['Sample', 'file_name', 'x motor', 'y motor',  'Gaussian1', 'FWHM1', 'Center1',
                                     'Gaussian2', 'FWHM2', 'Center2', 'Gaussian3', 'FWHM3', 'Center3'])
# path to all the tiff files
general_input_folder = r'D:\NSLS-II Winter 2023'
input_folder = os.path.join(general_input_folder, sample_name, 'integration')

general_output_folder = r'C:\Users\Elizabeth Allan-Cole\Desktop\XRD Data Processing\NSLS-II Winter 2023\Processing\Initial_fit'
output_folder = os.path.join(general_output_folder,  'Output',  sample_name)
plot_folder = os.path.join(general_output_folder, 'Plot Output')

# if that folder dosn't exist make it exist
if not os.path.exists(output_folder):
     os.makedirs(output_folder)

#Set isolated peak q range dict: [q_min, q_max, chi squared, sigma, amplitude]
#q_range_dict = {'Graphite-LixC6':[1.75, 1.9, 1000, 0.1, 150], 'NMC':[1.25, 1.36, 5000, 0.2, 500], 'Li': [2.49, 2.55, 10, 0.05, 1]}
#q_range_dict = {'Graphite-LixC6':[1.75, 1.9, 50, 0.1, 100], 'NMC':[1.25, 1.36, 30000, 0.1, 1000], 'Li': [2.49, 2.55, 10, 0.05, 1]}
#q_range_dict = {'Graphite-LixC6':[1.6, 1.9, 50, 0.1, 100], 'Li': [2.45, 2.57, 2, 0.05, 1]}

# Graphite/LiC12 only
#q_range_dict = {'Graphite-LixC6':[1.75, 1.9, 50, 0.1, 100]}

# LiC6 only
#q_range_dict = {'LiC6':[1.6, 1.75, 10, 0.1, 100]}

# nmc peaks only
q_range_dict = {'NMC':[1.25, 1.36, 150, 0.1, 1500]}

# Li peaks only
#q_range_dict = {'Li': [2.45, 2.57, 2, 0.05, 1]}

# Make a list of all files names in folder
list_of_files = [files for files in listdir(input_folder) if isfile(join(input_folder, files))]


for element in q_range_dict.keys():
    df_integrals_temp = pd.DataFrame(columns=['Sample', 'file_name', 'x motor', 'y motor',  'Gaussian1', 'FWHM1', 'Center1',
                                     'Gaussian2', 'FWHM2', 'Center2', 'Gaussian3', 'FWHM3', 'Center3'])
    q_min = q_range_dict.get(element)[0]
    q_max = q_range_dict.get(element)[1]
    sig = q_range_dict.get(element)[3]
    amp =q_range_dict.get(element)[4]
    chisqu_fit_value = q_range_dict.get(element)[2]
    print("finding ", element, " peaks! Hold on to your socks!")
    print("qmax is " ,q_min, "q min is ", q_max)

    n = 0
    # loop through the list of files and append df_integrals --> Troubleshoot the peak fitting, getting weird numbers! 
    for i in range(len(list_of_files)):
        if i == 5:
            break
        if 'mean_q' in list_of_files[i]:
            
            #Call the master function to get the integral values for the specified peak
            # returns [sample_name, x_motor, y_motor, integral_list, fwhm_list, peak_center_list, best_model]
            get_integrals = pf.master_function(list_of_files[i], num_of_centers, input_folder, q_min, q_max, 
                                            sample_name, sig, amp, chisqu_fit_value, element, Li_q_max, Li_q_min, plot)
            
            
            # save the plots for the best fit if you want
            pf.save_fits(plot_folder, get_integrals, element, list_of_files, i, sample_name)
            
            
            # this just prints the number of files we've cronked through
            print(n)
            n += 1
            
            # uncomment me to see the fits!! Comment out for processing the whole data set. 
            #print(get_integrals[6].plot())
            print('chisqr is ', get_integrals[6].chisqr)
            print('x_motor and y motor are: (' + str(get_integrals[1]) + ',' + str(get_integrals[2]) + ')')
            
            
            
            # zips the integral_list, fwhm_list, peak_center_list together to make a list of lists
            # ie ((integral_1, fwhm_1, center_1), (integral_2, fwhm_2, center_2))
            vals_list = list(zip(get_integrals[3], get_integrals[4], get_integrals[5]))
            print('peak centers: ', get_integrals[5])
            
            #flatten the list to just a list (integral_1, fwhm_1, center_1, integral_2, fwhm_2, center_2)
            vals_list = [item for sublist in vals_list for item in sublist]
            
            
            # add the sample and position info sample_name, x_motor, y_motor
            info_list = [get_integrals[0], get_integrals[1], get_integrals[2]]
            # add the filename 
            info_list.insert(1, list_of_files[i])
            # add then together
            info_list = info_list + vals_list
            # Find the number of nan vales we add to make this list have 12 values so we can slap it in a dataframe
            num_nans = df_integrals_temp.shape[1] - len(info_list)
            
            # Add a bunch of nans
            x = 0
            while x < num_nans:
                info_list.append(np.nan)
                x += 1
                
            # find the last row in the df    
            max_row = df_integrals_temp.shape[0]
            # slap our list of values in the dataframe!
            df_integrals_temp.loc[max_row + 1,] = info_list
            
    break       
    # after each peak is run save the data frame
    file_name = str(get_integrals[0] + '_' + element + '.csv')
    output_file = os.path.join(output_folder, file_name)
    df_integrals_temp.to_csv(output_file)

    # add data to the master data frame
    if df_integrals.empty:
        df_integrals = df_integrals_temp
    else:
        df_integrals = pd.concat([df_integrals, df_integrals_temp])

# save the master dataframe
file_name = str(get_integrals[0]) + '_all_data.csv'
output_file = os.path.join(output_folder, file_name)
df_integrals.to_csv(output_file)

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
