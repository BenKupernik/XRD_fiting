# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:38:13 2023

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



sample_name = 'S1_LN_10psi_Ch10_0120922_map_01-4' #charged state

# path to all the tiff files
general_input_folder = r'D:\NSLS-II Winter 2023'
input_folder = os.path.join(general_input_folder, sample_name, 'integration')

peak_name = 'Li(110)'
q_min = 2.45
q_max = 2.59

# Make a list of all files names in folder
list_of_files = [files for files in listdir(input_folder) if isfile(join(input_folder, files))]

x_min, x_max = 92, 102.5
y_min, y_max = 66, 70.5
n=0
for i in range(len(list_of_files)):
    sample_name = list_of_files[i]
    data_path = input_folder

    #if i == 5:
        #break

    if 'mean_q' in list_of_files[i]:
        
        df = pf.make_dataframe(sample_name, data_path)
        x_motor, y_motor = pf.get_xy_motor(sample_name, data_path)
        
        if x_motor >= x_min and x_motor <= x_max:
            if y_motor >= y_min and y_motor <= y_max:
                print(n)
    
                sliced_q, sliced_I = pf.get_points(df,q_min,q_max)
    
                fig, ax = plt.subplots(1,1, figsize=(7,7))
                ax.scatter(sliced_q,sliced_I, label='Data', color='black')
                ax.set_title(str(peak_name) + ' : (' + str(x_motor) + ',' + str(y_motor) + ')') 
                ax.set_xlabel('q [1/A]')
                ax.set_ylabel('I [au.]')
                ax.legend()
                
                plt.show()
                n+=1
