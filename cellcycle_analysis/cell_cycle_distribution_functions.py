# %% Import libraries

import pandas as pd
import numpy as np
import scipy
from scipy import signal
import math
import matplotlib.pyplot as plt

# %% Defining data normalisation function

def fun_normalise(data, values, ctr_col, ctr_cond) :
    
    tmp_output = pd.DataFrame()
    
    for experiment in data["experiment"].unique() :
        
        for cell_line in data["cell_line"].unique() :
            
            tmp_data = data.copy().loc[(data["experiment"] == experiment) &
                                       (data["cell_line"] == cell_line)]
            
            tmp_data_subset = tmp_data.copy().loc[tmp_data[ctr_col] == ctr_cond]

            tmp_bins = 100
            
            for val in values :
                print(val)
                tmp_data_subset[val + "_log10"] = np.log10(tmp_data[val])
                #print(tmp_data_subset[val + "_log10"].values())
                #print(tmp_data_subset.loc[tmp_data_subset[val + "_log10"].math.isnan()])
                tmp_data_hist = pd.cut(tmp_data_subset[val + "_log10"], tmp_bins).value_counts().sort_index().reset_index()
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
                   mode = "automatic",
                   ctr_col = "condition",
                   ctr_cond = "0.0",
                   DAPI_col = "DAPI_total_norm",
                   EdU_col = "EdU_total_norm",
                   H3P_col = "H3P_total_norm") :
    
    tmp_output = pd.DataFrame()
    tmp_output_thresholds = pd.DataFrame()
   
    for experiment in data["experiment"].unique() :

        for cell_line in data["cell_line"].unique() :

            """ 
            Constructing the subset (control) dataset
            
            - Creating working (tmp_data) and normalisation (tmp_data_subset) dataframes.
            - Calculating log10-transformed normalised integrated DAPI abd mean EdU intensities."""
            
            tmp_data = data.copy().loc[(data["experiment"] == experiment) &
                                       (data["cell_line"] == cell_line)]
            
            tmp_data[DAPI_col + "_log10"] = np.log10(tmp_data[DAPI_col])
            tmp_data[EdU_col + "_log10"] = np.log10(tmp_data[EdU_col])
            tmp_data[H3P_col + "_log10"] = np.log10(tmp_data[H3P_col])
            tmp_data_subset = tmp_data.loc[tmp_data[ctr_col] == ctr_cond]
            tmp_bins = 100
            
            """
            Step 1 - Establishing threshold values of normalised integrated DAPI intensities
            
            1.1) Creating histograms of log10-transformed normalised integrated DAPI intensities.
            1.2) Creating kernel density estimation (KDE).
            1.3) Finding two most prominent KDE peaks of repressenting centers of G1 and G2 populations.
                 Establishing the half-distance between G1 and G2 peaks.
            1.4) Establishing the "expand" parameter, which is implemented in determination of the low threshold.
                 The "expand" parameter represents the ratio of G1 and G2 maxima.
                 Discrepancy in hights of G1 and G2 peaks is translated into the widths of their bases.
            1.5) Determining the three thresholds (low, mid, high).
                 DAPI_low_threshold = "position of the G1 peak" minus "the half distance of G1 and G2 peaks" minus "the expand parameter". 
                 DAPI_mid_threshold = "position of the G1 peak" plus "the half distance of G1 and G2 peaks"
                 DAPI_high_threshold = "position of the G1 peak" plus "the half distance of G1 and G2 peaks" plus 1 (compensation for math.floor used in determination of the half-distance).
            """

            """ 1.1 """ 
            DAPI_hist = pd.cut(tmp_data_subset[DAPI_col + "_log10"], int(tmp_bins)).value_counts().sort_index().reset_index()
            DAPI_hist.rename(columns = {"index" : "interval"}, inplace = True)
                      
            """1.2"""
            DAPI_min = min(tmp_data_subset[DAPI_col + "_log10"])
            DAPI_max = max(tmp_data_subset[DAPI_col + "_log10"])
            tmp_kde = scipy.stats.gaussian_kde(tmp_data_subset[DAPI_col + "_log10"])
            DAPI_hist["kde"] = tmp_kde.evaluate(np.linspace(DAPI_min, DAPI_max, tmp_bins))
            
            """1.3"""
            DAPI_peaks_all = list(signal.find_peaks(DAPI_hist["kde"])[0])
            DAPI_peaks_selection = sorted(DAPI_hist.iloc[DAPI_peaks_all].sort_values("kde", ascending = False).head(n = 2).index)
            DAPI_peaks_distance_half = math.floor((DAPI_peaks_selection[1] - DAPI_peaks_selection[0]) / 2)
            
            """1.4"""
            tmp_expand = int(np.ceil(DAPI_hist["kde"][DAPI_peaks_selection[0]] / DAPI_hist["kde"][DAPI_peaks_selection[1]]))

            """1.5"""
            DAPI_low_threshold = DAPI_hist.iloc[(DAPI_peaks_selection[0] - DAPI_peaks_distance_half - tmp_expand)]["interval"].left
            DAPI_mid_threshold = DAPI_hist.iloc[(DAPI_peaks_selection[0] + DAPI_peaks_distance_half)]["interval"].mid
            DAPI_high_threshold = DAPI_hist.iloc[(DAPI_peaks_selection[1] + DAPI_peaks_distance_half + 1)]["interval"].right
            

            """
            Step 2 - Establishing threshold values of normalised mean EdU intensities
            
            2.1) Creating histograms of log10-transformed normalised mean EdU intensities.
            2.2) Creating kernel density estimation (KDE).
            2.3) Finding two most prominent KDE peaks repressenting centers of non-replicating and replicating populations.
                 Establishing the local minimum between two peaks, which represents the "background".
            2.4) Subtracting the local minimum value from the KDE distribution and transforming negative values into 0.
            2.5) The EdU trheshold is determined by applying the signal.peak_widths function on the first (non-replicative) KDE peak.

            """

            """ 2.1 """
            EdU_hist = pd.cut(tmp_data_subset[EdU_col + "_log10"], tmp_bins).value_counts().sort_index().reset_index()
            EdU_hist.rename(columns = {"index" : "interval"}, inplace = True)
            
            """2.2"""
            EdU_min = min(tmp_data_subset[EdU_col + "_log10"])
            EdU_max = max(tmp_data_subset[EdU_col + "_log10"])
            tmp_kde = scipy.stats.gaussian_kde(tmp_data_subset[EdU_col + "_log10"])
            EdU_hist["kde"] = tmp_kde.evaluate(np.linspace(EdU_min, EdU_max, tmp_bins))
            
            """2.3"""
            EdU_peaks_all = signal.find_peaks(EdU_hist["kde"])[0]
            EdU_peaks_selection = sorted(EdU_hist.iloc[EdU_peaks_all].sort_values("kde", ascending = False).head(n = 2).index)
            EdU_local_min = min(EdU_hist.iloc[EdU_peaks_selection[0]:EdU_peaks_selection[1]]["kde"])
            
            """2.4"""
            EdU_hist["kde"] = EdU_hist["kde"] - EdU_local_min
            EdU_hist["kde"] = np.where(EdU_hist["kde"] <= 0, 0, EdU_hist["kde"])
            EdU_hist.loc[0, "kde"] = 0
            
            """2.5"""
            EdU_threshold_index = round(signal.peak_widths(EdU_hist["kde"], EdU_peaks_selection, rel_height = 0.85)[3][0])
            EdU_threshold = EdU_hist["interval"].tolist()[EdU_threshold_index].mid

            """
            Step 4 – Establishing threshold values of normalised mean H3-P intensities
            3.1) Creating histograms of log10-transformed normalised mean H3-P intensities.
            3.2) Creating kernel density estimation (KDE).
            3.3) Finding the most prominent KDE peak repressenting the center of non-mitotic population.
            3.4) The H3P trheshold is determined by applying the signal.peak_widths function on the non-mitotic KDE peak.
    
            """
            
            """3.1"""
            H3P_hist = pd.cut(tmp_data_subset[H3P_col + "_log10"], tmp_bins).value_counts().sort_index().reset_index()
            H3P_hist.rename(columns = {"index" : "interval"}, inplace = True)
            
            """3.2"""
            H3P_min = min(tmp_data_subset[H3P_col + "_log10"])
            H3P_max = max(tmp_data_subset[H3P_col + "_log10"])
            tmp_kde = scipy.stats.gaussian_kde(tmp_data_subset[H3P_col + "_log10"])
            H3P_hist["kde"] = tmp_kde.evaluate(np.linspace(H3P_min, H3P_max, tmp_bins))
            
            """3.3"""
            H3P_peaks_all = signal.find_peaks(H3P_hist["kde"])[0]
            H3P_peaks_selection = sorted(H3P_hist.iloc[H3P_peaks_all].sort_values("kde", ascending = False).head(n = 1).index)
            
            """3.4"""
            H3P_threshold_index = round(signal.peak_widths(H3P_hist["kde"], H3P_peaks_selection, rel_height = 1)[3][0])
            H3P_threshold = H3P_hist["interval"].tolist()[H3P_threshold_index].mid
            
            """ Storing established thresholds in a dataframe """
            
            tmp_data_thresholds = pd.DataFrame({
                
                "experiment" : [experiment],
                "cell_line" : [cell_line],
                "EdU_threshold_log10" :  [EdU_threshold],
                "EdU_threshold" : [10 ** EdU_threshold],
                "DAPI_low_threshold_log10" : [DAPI_low_threshold],
                "DAPI_low_threshold" : [10 ** DAPI_low_threshold],
                "DAPI_mid_threshold_log10" : [DAPI_mid_threshold],
                "DAPI_mid_threshold" : [10 ** DAPI_mid_threshold],
                "DAPI_high_threshold_log10" : [DAPI_high_threshold],
                "DAPI_high_threshold" : [10 ** DAPI_high_threshold],
                "H3P_threshold_log10" : [H3P_threshold],
                "H3P_threshold" : [10 ** H3P_threshold]
                })
     
            """
            Step 3 - Defining the function which uses established thresholds and assigns a cell cycle phase to each cell.
            
            - The function generates two outputs:
                Output 1 - The input data with assigned cell cycle phases
                Output 2 - A dataframe containing all threshold values
               
            """
                      
            def fun_thresholding (dataset) :
                
                if (dataset[DAPI_col + "_log10"] < DAPI_low_threshold) :
                    return "Sub-G1"
                
                elif (dataset[DAPI_col + "_log10"] >= DAPI_low_threshold) and (dataset[DAPI_col + "_log10"] < DAPI_mid_threshold) and (dataset[EdU_col + "_log10"] < EdU_threshold) :
                    return "G1"
                
                elif (dataset[DAPI_col + "_log10"] >= DAPI_mid_threshold) and (dataset[DAPI_col + "_log10"] < DAPI_high_threshold) and (dataset[EdU_col + "_log10"] < EdU_threshold) and (dataset[H3P_col + "_log10"] < H3P_threshold) :
                    return "G2"
                
                elif (dataset[DAPI_col + "_log10"] >= DAPI_mid_threshold) and (dataset[DAPI_col + "_log10"] < DAPI_high_threshold) and (dataset[EdU_col + "_log10"] < EdU_threshold) and (dataset[H3P_col + "_log10"] >= H3P_threshold) :
                    return "M"
                
                elif (dataset[DAPI_col + "_log10"] >= DAPI_low_threshold) and (dataset[DAPI_col + "_log10"] < DAPI_mid_threshold) and (dataset[EdU_col + "_log10"] >= EdU_threshold) :
                    return "Early S"
                
                elif (dataset[DAPI_col + "_log10"] >= DAPI_mid_threshold) and (dataset[DAPI_col + "_log10"] < DAPI_high_threshold) and (dataset[EdU_col + "_log10"] >= EdU_threshold) :
                    return "Late S"
                
                elif (dataset[DAPI_col + "_log10"] >= DAPI_high_threshold and (dataset[EdU_col + "_log10"] < EdU_threshold)) :
                    return "Polyploid"
                
                elif (dataset[DAPI_col + "_log10"] >= DAPI_high_threshold and (dataset[EdU_col + "_log10"] >= EdU_threshold)) :
                    return "Polyploid (replicating)"
                
                else :
                    return "Unasigned"
                      
            tmp_data["cell_cycle_detailed"] = tmp_data.apply(fun_thresholding, axis = 1)
            
            tmp_data["cell_cycle"] = np.where(tmp_data["cell_cycle_detailed"].isin(["G2", "M"]), "G2/M",
                                              np.where(tmp_data["cell_cycle_detailed"].isin(["Early S", "Late S"]), "S",
                                                       np.where(tmp_data["cell_cycle_detailed"].isin(["Polyploid", "Polyploid (replicating)"]), "Polyploid", tmp_data["cell_cycle_detailed"])))

            tmp_output = pd.concat([tmp_output, tmp_data])
            
            tmp_output_thresholds = pd.concat([tmp_output_thresholds, tmp_data_thresholds])
        
    return(tmp_output, tmp_output_thresholds)
