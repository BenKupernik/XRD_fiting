# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:42:17 2023

@author: Elizabeth Allan-Cole
"""

import peak_fitter_functions as pf
import Li_background_functions as lb
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from pathlib import Path
from os import listdir, chdir
from os.path import isfile, join
import regex as re
import math
import time
import itertools as it
from lmfit.model import load_modelresult
from scipy import stats


def get_model_path(x_motor, y_motor, df):
    
    df_temp = df.drop(df[df.x_motor != x_motor].index)
    df_temp = df_temp.drop(df_temp[df_temp.y_motor != y_motor].index)
    
    return list(df_temp['Model Path'])[0]


def load_model_file(model_path, short_q, sample_name):

    os.chdir(model_path)
    model = load_modelresult(sample_name) 
    
    return model

def partition_data (df):
    integrals = df['Gaussian1'] 
    df['Percentile'] = df['Gaussian1'].apply(lambda x: stats.percentileofscore(integrals, x))
   
    return df

def get_percentile(x_motor, y_motor, df):
      
    df_temp = df.drop(df[df.x_motor != x_motor].index)
    df_temp = df_temp.drop(df_temp[df_temp.y_motor != y_motor].index)
    
    return list(df_temp['Percentile'])[0]

sample_name = 'S1_LN_10psi_Ch10_0120922_map_01-4' #charged state

# path to all the tiff files
general_input_folder = r'D:\NSLS-II Winter 2023'
input_folder = os.path.join(general_input_folder, sample_name, 'integration')

general_path = r'C:\Users\Elizabeth Allan-Cole\Desktop\XRD Data Processing\NSLS-II Winter 2023\Processing\Initial_fit\Output'

# Path to get pull output df from main peak fitter

csv_path = os.path.join(general_path, sample_name, str(sample_name) + '_Li_test.csv')
df = pd.read_csv(csv_path)
df = df.rename(columns={"x motor": "x_motor", "y motor": "y_motor"})
df = partition_data(df)


peak_name = 'Li(110)'
q_min_long = 2.45
q_max_long = 2.58
q_min_short = 2.5
q_max_short = 2.57

percentile_list = ((5, 10), (47,53), (95,100))

# Make a list of all files names in folder
list_of_files = [files for files in listdir(input_folder) if isfile(join(input_folder, files))]

#setbounds for the NMC electrrode
x_min, x_max = 92, 102.5
y_min, y_max = 66, 70.5


n=0

for values in percentile_list:
    percentile_min = values[0]
    percentile_max = values[1]
    
    
    for i in range(len(list_of_files)):
        raw_file_name = list_of_files[i]
        data_path = input_folder
    
        #if i == 6:
            #break
    
        if 'mean_q' in list_of_files[i]:
            
            df_q = pf.make_dataframe(raw_file_name, data_path)
            x_motor, y_motor = pf.get_xy_motor(raw_file_name, data_path)
            percentile = get_percentile(x_motor, y_motor, df)
            
            if percentile >= percentile_min and percentile <= percentile_max:
                model_path = get_model_path(x_motor, y_motor, df)
                long_q, long_I = pf.get_points(df_q,q_min_long,q_max_long)
                short_q, short_I = pf.get_points(df_q,q_min_short,q_max_short)
                model = load_model_file(model_path, short_q, sample_name)
            
                if x_motor >= x_min and x_motor <= x_max:
                    if y_motor >= y_min and y_motor <= y_max:
                        print(n)
            
                        fig, ax = plt.subplots(1,1, figsize=(7,7))
                        ax.scatter(long_q, long_I, label='Data', color='red')
                        ax.plot(short_q, model.best_fit, label='Model', color='blue')
                        ax.set_title(str(peak_name) + ' : (' + str(x_motor) + ',' + str(y_motor) + ')' + ' Percentile: ' + str("{:.2f}".format(percentile))) 
                        ax.set_xlabel('q [1/A]')
                        ax.set_ylabel('I [au.]')
                        
                        ax.legend()
                        
                        plt.show()
                        n+=1
