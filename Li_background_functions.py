# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 19:49:01 2023

@author: Elizabeth Allan-Cole
"""
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy import optimize
from scipy import integrate
from scipy import stats
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
import peak_fitter_functions as pf


def get_q(input_folder, q_min, q_max):
    list_of_files = [files for files in listdir(input_folder) if isfile(join(input_folder, files))]
    df_q = pd.DataFrame(columns = ['q', 'I'])
    
    for i in range(len(list_of_files)):
        if i == 2:
            break
        if 'mean_q' in list_of_files[i]:
            df_q = pf.make_dataframe(list_of_files[i], input_folder)
            df_q = df_q.drop(['I'], axis=1)
            df_q_cut = df_q[(df_q['q'] >= q_min) & (df_q['q'] <= q_max)]
    
    q_array = df_q_cut.to_numpy().flatten()
    
    return q_array

# takes a peak center returns what elements that peak is
def what_peak(center):
    if center is np.nan:
        return 'f) unknown'
    elif center >= 1.25 and center <= 1.36:
        return 'e) NMC(003) Peak Width'
    # check this!!! looks like we might be missing graphite
    elif center >= 1.75 and center <= 1.8:
        return 'c) Lithiated Graphite'
    elif center >= 1.8 and center <= 1.85:
        return 'b) Partially Lithiated Graphite'
    elif center >= 1.85 and center <= 1.9:
        return 'a) Graphite'
    elif center >= 2.525 and center <= 2.54:
        return 'd) Li(110)'
    else:
        return 'f) unknown'
    

def format_df(df):
    # add columns for peak names 
    df['peak1'] = df['Center1'].apply(what_peak)
    df['peak2'] = df['Center2'].apply(what_peak)
    df['peak3'] = df['Center2'].apply(what_peak)
    
    
    # convert the df from wide to long
    # for peak1
    df_long1 = df.drop(['Gaussian2', 'FWHM2', 'Center2', 'Gaussian3', 'FWHM3',
           'Center3', 'peak2', 'peak3'], axis=1)
    df_long1 = df_long1.rename(columns={"Gaussian1": "Gaussian", "FWHM1": "FWHM", "Center1":"Center", "peak1": "peak"})
    
    
    # for peak 2
    df_long2 = df.drop(['Gaussian1', 'FWHM1', 'Center1', 'Gaussian3', 'FWHM3',
           'Center3', 'peak1', 'peak3'], axis=1)
    df_long2 = df_long2.rename(columns={"Gaussian2": "Gaussian", "FWHM2": "FWHM", "Center2":"Center", "peak2": "peak"})
    
    
    # for peak 3
    df_long3 = df.drop(['Gaussian1', 'FWHM1', 'Center1', 'Gaussian2', 'FWHM2',
           'Center2', 'peak1', 'peak2'], axis=1)
    df_long3 = df_long3.rename(columns={"Gaussian3": "Gaussian", "FWHM3": "FWHM", "Center3":"Center", "peak3": "peak"})
    
    #print(df_long.columns)
    df_long = pd.concat([df_long1, df_long2, df_long3]).reset_index(drop = True)
    # group the data by point and file FWHM and center gets averaged, peak integral gets sumed
    df_long = df_long.groupby(['Sample', 'x motor', 'y motor', 'file_name', 'peak', 'Model Path'], as_index=False).agg({'Gaussian':'sum', 'FWHM':'mean', 'Center':'mean'})
    
    df_long = df_long.rename(columns={"x motor": "x_motor", "y motor": "y_motor"})
    
    return df_long


def filter_df (df_long, peak_name):
    df_filter = df_long.drop(df_long[df_long.peak != peak_name].index)
    return df_filter


def partition_data (df_filter):
    integrals = df_filter['Gaussian'] 
    df_filter['Percentile'] = df_filter['Gaussian'].apply(lambda x: stats.percentileofscore(integrals, x))
   
    return df_filter
                                  

def filter_by_partition (df_filter, min_percentile, max_percentile):
    
    df_percentile = df_filter.drop(df_filter[df_filter.Percentile < min_percentile].index)
    df_percentile = df_percentile.drop(df_percentile[df_percentile.Percentile >= max_percentile].index).reset_index(drop = True)
    
    # # Temp sort by x and y mortor position 
    # x_min = 92
    # x_max = 102.5
    
    # y_min = 66
    # y_max = 71
    
    # df_percentile = df_percentile.drop(df_percentile[df_percentile.x_motor < x_min].index)
    # df_percentile = df_percentile.drop(df_percentile[df_percentile.x_motor > x_max].index)
    # df_percentile = df_percentile.drop(df_percentile[df_percentile.y_motor < y_min].index)
    # df_percentile = df_percentile.drop(df_percentile[df_percentile.y_motor > y_max].index).reset_index(drop = True)
    
    #print(df_percentile)
    return df_percentile


def get_background(result, q_array):
    
    comps = result.eval_components(x = q_array)

    for prefex in comps.keys():
        if prefex == 'b_':
            background = comps[prefex]

    return background


def load_model(df_percentile, q_array):
    
    model_path_list = df_percentile['Model Path'].values.tolist()
    sample_name = list(df_percentile['Sample'].unique())[0]
    results_dict = {}

    for i in range(len(model_path_list)):
        model_path = model_path_list[i]
        os.chdir(model_path)
        model = load_modelresult(sample_name)
        result = get_background(model, q_array)
        results_dict[model_path_list[i]] = result
        
    return results_dict
        

def background_df(q_array, results_dict):
    
    df_linear = pd.DataFrame(columns = q_array)
    for value in results_dict.values():
        max_row = df_linear.shape[0]
        df_linear.loc[max_row + 1,] = value
    
    return df_linear
    
    
def average(df_linear):
    
    df_mean = df_linear.mean(axis = 0).to_frame()
    df_mean['q'] = df_mean.index
    df_mean = df_mean.reset_index(drop=True)
    df_mean = df_mean.rename(columns={0: 'I'})
    
    return df_mean


def plot_background (df_linear, q_array):
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    for row in range(len(df_linear)):
        ax.scatter(q_array, df_linear.iloc[row], label='Data', color='black')
        
        


def master_background(df, input_folder, q_min, q_max, peak_name, plot):
    
    q_array = get_q(input_folder, q_min, q_max)
    
    df_long = format_df(df)
    
    df_filter = filter_df(df_long, peak_name)
    
    df_filter = partition_data(df_filter)
    
    df_percentile = filter_by_partition(df_filter, 0, 100)
    df_percentile.to_csv('partition_test_new.csv')

    results_dict = load_model(df_percentile, q_array)
    
    df_linear = background_df(q_array, results_dict)
    
    df_mean = average(df_linear)
    
    if plot == True:
        plot_background(df_linear, q_array)
    
    return df_mean


