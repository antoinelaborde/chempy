# -*- coding: utf-8 -*-
"""
Created on Wed Oct 03 2018

@author: ALA
Code for preprocessing
"""


"""
process_type: {'Name_of_the_preprocessing':parameter_dict}

example:
    {'SNV':None}
    {'MeanReduction':None}
    {'SG':{'window':11,'deriv':1,'polyorder':2}}

"""

import numpy as np
import chempy.utils.util

"""
def process(div, process_type):
def savitzky_golay(div, window_size, order, deriv):
def snv(div):
def meancenter(div):
def standardize(div):
    
"""

def process(div, process_type):
    """
    process data in div
    Parameters
    ----------
    div: div structure (mandatory)
        div to process
    process_type: dict (mandatory)
        dictionnary for processing
    Returns
    -------
    d,i,v: d data np array, i list of strings names of the row 'individuals' ,
    v list of strings names of the columns ('variables')
    """
    DICT_PROCESS = {
        'snv':[],
        'meancenter':[],
        'standardize':[],
        'sg':['window','deriv','polyorder']
    }

    # Check the process_type dictionary
    process_name = list(process_type.keys())[0]
    if process_name not in DICT_PROCESS.keys():
        raise ValueError(process_name + ' is not a valid preprocessing name. Available preprocessings: ' + str(list(DICT_PROCESS.keys())))
    
    if process_type[process_name] is None:
        list_parameters = None
    else:
        list_parameters = list(process_type[process_name].keys())
        for param in list_parameters:
            if param not in DICT_PROCESS[process_name]:
                raise ValueError(param + ' is not a valid parameter name. For ' + process_name + ' preprocessing, you may inform those parameters: ' + str(DICT_PROCESS[process_name]))
    
    process_div = util.copy(div)

    if process_name == 'snv':
        process_div = snv(process_div)
    elif process_name == 'meancenter':
        process_div = meanreduction(process_div)
    elif process_name == 'standardize':
        process_div = standardize(process_div)
    elif process_name == 'sg':
        process_div = np.apply_along_axis(savitzky_golay, 1, process_div ,process_type[process_name]['window'],process_type[process_name]['polyorder'], process_type[process_name]['deriv'])

    
    # Update the .p field of the div object
    process_div.p.append(process_type)

    return process_div




def savitzky_golay(div, window_size, order, deriv):
    """
    savitzky_golay performs filter on y data
    Parameters
    ----------
    div: div structure (mandatory)
        data to process
    window_size: int (mandatory)
        size of the window
    order: int (mandatory)
        order of the polynomial
    deriv: int (mandatory)
        order of the derivation
    
    Returns
    -------
    div_processed
    """
    #Copy div
    div_processed = util.copy(div)
    y = div_processed.d
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in
                   range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv]

    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    div_processed.d = np.convolve(m, y, mode='valid')

    return div_processed

def snv(div):
    """
    standard normal variate preprocessing
    Parameters
    ----------
    div: div structure (mandatory)
        data to process
    
    Returns
    -------
    div_processed
    """
    #Copy div
    div_processed = util.copy(div)
    x = div_processed.d

    # Mean calculation
    mean_vec = np.mean(x, axis=1)
    mean_vec = np.expand_dims(mean_vec, 1)
    mean_removal = np.repeat(mean_vec, x.shape[1], axis=1)

    # Standard deviation calulation
    std_vec = np.std(x, axis=1)
    std_vec = np.expand_dims(std_vec, 1)
    std_removal = np.repeat(std_vec, x.shape[1], axis=1)

    # SNV calculation
    x = x - mean_removal
    x = np.divide(x, std_removal)

    div_processed.d = x
    
    return div_processed

def meancenter(div):
    """
    meancenter preprocessing
    Parameters
    ----------
    div: div structure (mandatory)
        data to process
    
    Returns
    -------
    div_processed
    """
    #Copy div
    div_processed = util.copy(div)
    x = div_processed.d
    # Mean calculation
    mean_vec = np.mean(x, axis=1)
    mean_vec = np.expand_dims(mean_vec, 1)
    mean_removal = np.repeat(mean_vec, x.shape[1], axis=1)

    x = x - mean_removal
    div_processed.d = x
    return div_processed

def standardize(div):
    """
    divide each column by its standard deviation 
    input argument 
    ----------------
    div: an instance of class Div
    output argument
    ----------------
    div1: standardized data
    """
    #Copy div
    div_processed = util.copy(div)
    x = div_processed.d

    xstd = np.std(x,axis=0)
    x = x / xstd  
    #div_processed.id=div_processed.id + ' standardize'  
    div_processed.d = x
    return div_processed   