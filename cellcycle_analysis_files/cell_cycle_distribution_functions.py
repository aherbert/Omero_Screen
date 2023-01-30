# %% Import libraries

import pandas as pd
import numpy as np
import scipy
from scipy import signal
import math
import matplotlib.pyplot as plt

# %% Defining data normalisation function

def fun_normalise(data, values):
    """
    The function takes the parameters of the input values and normalizes the corresponding column of input data

    :param data: DataFrame
    :param values: list, default values=["DAPI_total", "intensity_mean_EdU_cell", "intensity_mean_H3P_cell", "area_cell"]
           the specify parameters in  values should be same as corresponding column name of input data
           for example, the parameter "DAPI_total"  is the column name in data.
    :return: DataFrame
           the data will include extra more val+'_log10',val+'_norm' as columns. For example 'DAPI_total_log10','DAPI_total_norm'
    """
    tmp_output = pd.DataFrame()
    
    for experiment in data["experiment"].unique() :
        
        for cell_line in data.loc[data["experiment"] == experiment]["cell_line"].unique() :
            
            tmp_data = data.copy().loc[(data["experiment"] == experiment) &
                                       (data["cell_line"] == cell_line)]
            tmp_bins = 100
            
            for val in values :

                tmp_data[val + "_log10"] = np.log10(tmp_data[val])
                tmp_data_hist = pd.cut(tmp_data[val + "_log10"], tmp_bins).value_counts().sort_index().reset_index()
                tmp_data_hist.rename(columns = {"index": "interval"}, inplace = True)
                tmp_data_hist = tmp_data_hist.sort_values(val + "_log10", ascending = False)
                tmp_denominator = 10 ** tmp_data_hist["interval"].values[0].mid
                tmp_data[val + "_norm"] = tmp_data[val] / tmp_denominator
                
            tmp_output = pd.concat([tmp_output, tmp_data])

    return(tmp_output)

# %% Defining cell cycle phase asignment function # 2
""" This function assigns a cell cycle phase
to each cell based on normalised EdU and DAPI intensities."""

def fun_CellCycle (data,
                   DAPI_col = "DAPI_total_norm",
                   EdU_col = "intensity_mean_EdU_cell_norm",
                   H3P_col = "intensity_mean_H3P_cell_norm") :
    
    tmp_output = pd.DataFrame()
   
    thresholds = {
        
        "DAPI_low_threshold" : [0.6],
        "DAPI_mid_threshold" : [1.4],
        "DAPI_high_threshold" : [2.6],
        "EdU_threshold" : [1.4],
        "H3P_threshold" : [1.4]} 
        
    """ Storing established thresholds in a dataframe """
    
    tmp_data_thresholds = pd.DataFrame(thresholds)

     
    """
    Step 3 - Defining the function which uses established thresholds and assigns a cell cycle phase to each cell.
    
    - The function generates two outputs:
        Output 1 - The input data with assigned cell cycle phases
        Output 2 - A dataframe containing all threshold values
               
    """
                      
    def fun_thresholding (dataset) :
                
        if (dataset[DAPI_col] < thresholds["DAPI_low_threshold"][0]) :
            return "Sub-G1"
                
        elif (dataset[DAPI_col] >= thresholds["DAPI_low_threshold"][0]) and (dataset[DAPI_col] < thresholds["DAPI_mid_threshold"][0]) and (dataset[EdU_col] < thresholds["EdU_threshold"][0]) :
            return "G1"
                
        elif (dataset[DAPI_col] >= thresholds["DAPI_mid_threshold"][0]) and (dataset[DAPI_col] < thresholds["DAPI_high_threshold"][0]) and (dataset[EdU_col] < thresholds["EdU_threshold"][0]) and (dataset[H3P_col] < thresholds["H3P_threshold"][0]) :
            return "G2"
                
        elif (dataset[DAPI_col] >= thresholds["DAPI_mid_threshold"][0]) and (dataset[DAPI_col] < thresholds["DAPI_high_threshold"][0]) and (dataset[EdU_col] < thresholds["EdU_threshold"][0]) and (dataset[H3P_col] >= thresholds["H3P_threshold"][0]) :
            return "M"
                
        elif (dataset[DAPI_col] >= thresholds["DAPI_low_threshold"][0]) and (dataset[DAPI_col] < thresholds["DAPI_mid_threshold"][0]) and (dataset[EdU_col] >= thresholds["EdU_threshold"][0]) :
            return "Early S"
                
        elif (dataset[DAPI_col] >= thresholds["DAPI_mid_threshold"][0]) and (dataset[DAPI_col] < thresholds["DAPI_high_threshold"][0]) and (dataset[EdU_col] >= thresholds["EdU_threshold"][0]) :
            return "Late S"
                
        elif (dataset[DAPI_col] >= thresholds["DAPI_high_threshold"][0] and (dataset[EdU_col] < thresholds["EdU_threshold"][0])) :
            return "Polyploid"
                
        elif (dataset[DAPI_col] >= thresholds["DAPI_high_threshold"][0] and (dataset[EdU_col] >= thresholds["EdU_threshold"][0])) :
            return "Polyploid (replicating)"
                
        else :
            return "Unasigned"
                      
    data["cell_cycle_detailed"] = data.apply(fun_thresholding, axis = 1)
    
    data["cell_cycle"] = np.where(data["cell_cycle_detailed"].isin(["G2", "M"]), "G2/M",
                                  np.where(data["cell_cycle_detailed"].isin(["Early S", "Late S"]), "S",
                                           np.where(data["cell_cycle_detailed"].isin(["Polyploid (replicating)"]), "Polyploid", data["cell_cycle_detailed"])))

    tmp_output = pd.concat([tmp_output, data])
        
    return(tmp_output, tmp_data_thresholds)
