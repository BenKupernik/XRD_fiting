# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 19:59:07 2023

@author: Elizabeth Allan-Cole
"""
import Li_background_functions as lb
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
from lmfit.model import save_modelresult, load_modelresult
import math
import time
import itertools as it



# path to the csv the peak fiter scrip9 t made
sample_name = 'S1_LN_10psi_Ch10_0120922_map_01-4'

# path to all the tiff files
general_input_folder = r'D:\NSLS-II Winter 2023'
input_folder = os.path.join(general_input_folder, sample_name, 'integration')

general_path = r'C:\Users\Elizabeth Allan-Cole\Desktop\XRD Data Processing\NSLS-II Winter 2023\Processing\Initial_fit\Output'

# Path to get background
csv_path = os.path.join(general_path, sample_name, str(sample_name) + '_Li_test.csv')

# peak of interest, should be Li
peak_name = 'd) Li(110)'
q_min = 2.5
q_max = 2.56

df = pd.read_csv(csv_path)

df_mean = lb.master_background(df, input_folder, q_min, q_max, peak_name, plot = True)
