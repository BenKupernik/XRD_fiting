# -*- coding: utf-8 -*-
"""
Created on Mon May 15 20:16:28 2023

@author: Elizabeth Allan-Cole
"""
import peak_fitter_functions as pf
from lmfit import Model
from lmfit.models import LinearModel, GaussianModel, PolynomialModel, LorentzianModel, VoigtModel, PseudoVoigtModel

def make_Li_model(q_max, q_min, model_centers, sig, amp):
    
    # Hard set to bound over the NMC electrode
    # x_min, x_max = 92, 102.5
    # y_min, y_max = 66, 70.5
    
    # if x_motor >= x_min and x_motor <= x_max:
    #     if y_motor >= y_min and y_motor <= y_max:
            
    linear = LinearModel(prefix=('b' + '_'))  
    pars = linear.make_params()
    
    model = linear
    
    # initial guesses     
    slope1 = 0 
    int1 = 50
    
    # For linear background
    pars = linear.make_params()
    pars['b' + '_slope'].set(slope1)
    pars['b' + '_intercept'].set(int1)
    
    # For NMC peak background
    pref_v = 'v_b_'
    voigt = VoigtModel(prefix = pref_v)
    pars.update(voigt.make_params())
    pars[pref_v+'center'].set(value=1.6, min=q_max, max=q_max + 0.2)
    pars[pref_v+'sigma'].set(value=0.005, max = sig * 2)
    pars[pref_v+'amplitude'].set(value = 5, min = 0)
    pars[pref_v+'gamma'].set(value=0.005, vary=True, expr='', min = 0)
    
    model = linear + voigt
    
    for peak, center in enumerate(model_centers):
        # create prefex for each peak
        pref = 'v'+str(peak)+'_'
        #peak = GaussianModel(prefix=pref)
        peak = VoigtModel(prefix=pref)
        # set the parimiters for each peak
        pars.update(peak.make_params())
        pars[pref+'center'].set(value=center, min=2.52, max=2.55)
        pars[pref+'sigma'].set(value=0.003, max = sig * 2)
        pars[pref+'amplitude'].set(value=0.2, min = 0)
        #FOR PSEUDO VOIGT pars[pref+'alpha'].set(value=sig, vary=True, expr='', min = 0)
        pars[pref+'gamma'].set(value=sig, vary=True, expr='', min = 0)
        #pars[pref+'height'].set(value=height, vary=True, expr='', min = 0, max = height * 2)
        #pars[pref+'fwhm'].set(value=fwhm, vary=True, expr='', min = 0.0000001, max = fwhm * 1.5)
                
        model = model + peak
    
        return (model, pars)