#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 09:24:19 2023

@author: rz73
"""

# %% Selecting parameters of interest and aggregating counts of nuclei and total cellular DAPI signal

data_IF = data_raw.groupby(["experiment", "plate_id", "well", "well_id", "image_id",
                            "Cell_Line", "Condition", "Cyto_ID", "cell_id", "area_cell",

# !!! Include cytoplasmic EdU and H3P intensities          

                            "intensity_mean_EdU_cyto",
                            "intensity_mean_H3P_cyto"]).agg(
                                         
                                         nuclei_count = ("label", "count"),
                                         area_nucleus = ("area_nucleus", "sum"),
                                         DAPI_total = ("integrated_int_DAPI", "sum"),
                                         EdU_mean = ("intensity_mean_EdU_nucleus", "mean"),
                                         H3P_mean = ("intensity_mean_H3P_nucleus", "mean")).reset_index()

# !!! Correct nuclear EdU and H3P intensities using respective cytoplasmic intensities
                                
data_IF["EdU_mean_corr"] = data_IF["EdU_mean"] / data_IF["intensity_mean_EdU_cyto"]
data_IF["H3P_mean_corr"] = data_IF["H3P_mean"] / data_IF["intensity_mean_H3P_cyto"]
data_IF["Condition"] = data_IF["Condition"].astype(str)


# !!! The normalisation function is slightly changed
# !!! Specify values to normalise (use corrected values for EdU and H3P)
# !!! Specify DAPI_col (this is only done to make G1 values 2 and G2 values 4 instead of 1 and 2)

data_IF = fun_normalise(data = data_IF,
                        values = ["DAPI_total", "EdU_mean_corr", "H3P_mean_corr", "area_cell", "area_nucleus"],
                        DAPI_col = "DAPI_total")

# !!! Specify DAPI, EdU and H3P columns (use normalised corrected values for EdU and H3P)

data_IF, data_thresholds = fun_CellCycle(data = data_IF, DAPI_col = "DAPI_total_norm", EdU_col = "EdU_mean_corr_norm", H3P_col = "H3P_mean_corr_norm")

# %% Defining data normalisation function (Updated)

def fun_normalise(data, values, DAPI_col) :
    
    tmp_output = pd.DataFrame()
    
    for experiment in data["experiment"].unique() :
        
        for cell_line in data.loc[data["experiment"] == experiment]["Cell_Line"].unique() :
            
            tmp_data = data.copy().loc[(data["experiment"] == experiment) &
                                       (data["Cell_Line"] == cell_line)]
            tmp_bins = 100
            
            for val in values :

                tmp_data[val + "_log10"] = np.log10(tmp_data[val])
                tmp_data_hist = pd.cut(tmp_data[val + "_log10"], tmp_bins).value_counts().sort_index().reset_index()
                tmp_data_hist.rename(columns = {"index": "interval"}, inplace = True)
                tmp_data_hist = tmp_data_hist.sort_values(val + "_log10", ascending = False)
                tmp_denominator = 10 ** tmp_data_hist["interval"].values[0].mid
                tmp_data[val + "_norm"] = tmp_data[val] / tmp_denominator
                
                if val == DAPI_col :
                    
                    tmp_data[val + "_norm"] = tmp_data[val + "_norm"] * 2
                
                tmp_data = tmp_data.drop([val + "_log10"], axis = 1)
                tmp_data[val + "_norm_log2"] = np.log2(tmp_data[val + "_norm"])
                
            tmp_output = pd.concat([tmp_output, tmp_data])
    
    return(tmp_output)

# %% Defining cell cycle phase asignment function (Updated, with hard-coded thresholds)

""" This function assigns a cell cycle phase
to each cell based on normalised EdU and DAPI intensities."""

def fun_CellCycle (data, DAPI_col, EdU_col, H3P_col) :
    
    tmp_output = pd.DataFrame()
   
    thresholds = {
        
        "DAPI_low_threshold" : [1.5],
        "DAPI_mid_threshold" : [3],
        "DAPI_high_threshold" : [5.25],
        "EdU_threshold" : [1.5],
        "H3P_threshold" : [1.25]}
        
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
                                  np.where(data["cell_cycle_detailed"].isin(["Early S", "Late S"]), "S", data["cell_cycle_detailed"]))

    tmp_output = pd.concat([tmp_output, data])
        
    return(tmp_output, tmp_data_thresholds)

# %% Plotting & exporting combined EdU ~ DAPI scatter plots (Updated with the 0.1 percentile cut-off)

x_axis = "DAPI_total_norm"
y_axis = "EdU_mean_corr_norm"
conditions = ["0.0", "0.25", "0.5", "1.0", "2.0", "4.0"]

x_limits = [np.percentile(data_IF[x_axis], 0.1), np.percentile(data_IF[x_axis], 99.9)]
y_limits = [np.percentile(data_IF[y_axis], 0.1), np.percentile(data_IF[y_axis], 99.9)]

data_figure_scatter = data_IF.copy().loc[(data_IF[x_axis] > x_limits[0]) &
                                         (data_IF[x_axis] < x_limits[1]) &
                                         (data_IF[y_axis] > y_limits[0]) &
                                         (data_IF[y_axis] < y_limits[1]) &
                                         (data_IF["Condition"].isin(conditions))]

for cell_line in data_figure_scatter["Cell_Line"].unique() :
    
    for condition in data_figure_scatter["Condition"].unique() :
                 
        print(cell_line + " : " + condition)
        
        tmp_data = data_figure_scatter.loc[(data_figure_scatter["experiment"] == experiment) &
                                           (data_figure_scatter["Cell_Line"] == cell_line) &
                                           (data_figure_scatter["Condition"] == condition)]
        
        sns.set_context(context = 'talk',
                        rc = {'font.size': 8.0,
                              'axes.labelsize': 8.0,
                              'axes.titlesize': 8.0,
                              'xtick.labelsize': 8.0,
                              'ytick.labelsize': 8.0,
                              'legend.fontsize': 3,
                              'axes.linewidth': 0.5,
                              'grid.linewidth': 0.5,
                              'lines.linewidth': 0.5,
                              'lines.markersize': 2.5,
                              'patch.linewidth': 0.5,
                              'xtick.major.width': 0.5,
                              'ytick.major.width': 0.5,
                              'xtick.minor.width': 0.5,
                              'ytick.minor.width': 0.5,
                              'xtick.major.size': 5.0,
                              'ytick.major.size': 5.0,
                              'xtick.minor.size': 2.5,
                              'ytick.minor.size': 2.5,
                              'legend.title_fontsize': 3})
        
        Figure = sns.JointGrid(
            
            ratio = 3,
            xlim = (data_figure_scatter[x_axis].min(),
                    data_figure_scatter[x_axis].max()),
            ylim = (data_figure_scatter[y_axis].min() - data_figure_scatter[y_axis].min() * 0.2,
                    data_figure_scatter[y_axis].max()))
            
        
        Figure.ax_joint.set_xscale("log", base = 2)
        Figure.ax_joint.set_yscale("log", base = 2)
        
        Figure.ax_joint.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        Figure.ax_joint.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        
        Figure.ax_joint.ticklabel_format(axis = "both", style = "plain")
        Figure.ax_joint.tick_params(axis = "x", labelrotation = - 90)
        
        Figure.ax_joint.set_xticks([0.5, 1, 2, 4, 8, 16, 32])
        Figure.ax_joint.set_yticks([0.5, 1, 2, 4, 8, 16])

        Figure.refline(y = data_thresholds["EdU_threshold"].values)
        Figure.refline(x = data_thresholds["DAPI_low_threshold"].values)
        Figure.refline(x = data_thresholds["DAPI_mid_threshold"].values)
        Figure.refline(x = data_thresholds["DAPI_high_threshold"].values)
            
        Figure.set_axis_labels("Integrated Hoechst intensity\n$(log_{2}$, normalised)",
                               '\nMean EdU intensity\n$(log_{2}$, normalised)')
  
        sns.scatterplot(
                
            data = tmp_data,
            marker = "o",
            x = x_axis,
            y = y_axis,
            color = '#000000',
            hue = "cell_cycle_detailed",
            palette = {"G1": "#6794db", "Early S": "#aed17d", "Late S": "#63a678", "G2": "#CC6677", "M": "#CC6677", "Polyploid" : "#b39bcf", "Polyploid (replicating)" : "#e3b344", "Sub-G1" : "#c7c7c7"},
            ec = "none",
            linewidth = 0,
            alpha = 0.1,
            legend = False,
            ax = Figure.ax_joint)
        
        sns.histplot(
        
            data = tmp_data,
            x = x_axis,
            color = "#ADACAC",
            ax = Figure.ax_marg_x,
            bins = 100,
            element = "step",
            stat = "density",
            fill = True)
        
        sns.histplot(
        
            data = tmp_data,
            y = y_axis,
            color = "#ADACAC",
            ax = Figure.ax_marg_y,
            bins = 100,
            element = "step",
            stat = 'density',
            fill = True)
        
        Figure.ax_joint.text(
            data_figure_scatter[x_axis].max() - data_figure_scatter[x_axis].min() * 0.4,
            data_figure_scatter[y_axis].max() - data_figure_scatter[y_axis].max() * 0.1,
            str(len(tmp_data)),
            horizontalalignment = "right",
            verticalalignment = "top",
            size = 7,
            color = "#000000",
            weight ="normal")
    
        Figure.fig.set_figwidth(2)
        Figure.fig.set_figheight(2)
    
        Figure.savefig(path_export + "EdU-DAPI" + "_" + cell_line + "_" + condition + ".pdf", dpi = 300)
        Figure.savefig(path_export + "EdU-DAPI" + "_" + cell_line + "_" + condition + ".png", dpi = 1000)
        del(tmp_data)
        
# %% Plotting & exporting distributions of H3-P signal in G2/M cells (Updated with the 0.1 percentile cut-off)

x_axis = "Cell_Line"
y_axis = "H3P_mean_corr_norm"
conditions = ["0.0", "0.25", "0.5", "1.0", "2.0", "4.0"]

y_limits = [np.percentile(data_IF[y_axis], 0.1), np.percentile(data_IF[y_axis], 99.9)]

data_figure_scatter = data_IF.copy().loc[(data_IF["cell_cycle_detailed"].isin(["G2", "M"])) &
                                         (data_IF["Condition"].isin(conditions))]

for cell_line in data_figure_scatter.loc[data_figure_scatter["experiment"] == experiment]["Cell_Line"].unique() :
    
    for condition in data_figure_scatter["Condition"].unique() :
        
        print(cell_line + " : " + condition)
        
        tmp_data = data_figure_scatter.loc[(data_figure_scatter["Cell_Line"] == cell_line) &
                                           (data_figure_scatter["Condition"] == condition) &
                                           (data_figure_scatter["experiment"] == experiment)]
            
        sns.set_context(context = 'talk',
                        rc = {'font.size': 8.0,
                              'axes.labelsize': 8.0,
                              'axes.titlesize': 8.0,
                              'xtick.labelsize': 8.0,
                              'ytick.labelsize': 8.0,
                              'legend.fontsize': 3,
                              'axes.linewidth': 0.5,
                              'grid.linewidth': 0.5,
                              'lines.linewidth': 0.5,
                              'lines.markersize': 2.5,
                              'patch.linewidth': 1.5,
                              'xtick.major.width': 0,
                              'ytick.major.width': 0.5,
                              'xtick.minor.width': 0,
                              'ytick.minor.width': 0.5,
                              'xtick.major.size': 5.0,
                              'ytick.major.size': 5.0,
                              'xtick.minor.size': 2.5,
                              'ytick.minor.size': 2.5,
                              'legend.title_fontsize': 3})
        
        Figure = sns.JointGrid(
            
            ratio = 3,

            ylim = (data_figure_scatter[y_axis].min() - data_figure_scatter[y_axis].min() * 0.2,
                    data_figure_scatter[y_axis].max())
            )

        Figure.ax_joint.set_yscale("log", base = 2) 
        Figure.ax_joint.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        Figure.ax_joint.ticklabel_format(axis = "y", style = "plain")
        Figure.ax_joint.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16, 32])

        Figure.refline(y = data_thresholds["H3P_threshold"].values)
        
        
        if len(tmp_data) == 0 :
        
            sns.scatterplot(
                
                data = pd.DataFrame({x_axis : ["G2/M"],
                                     y_axis : [np.nan]}),
                x = x_axis,
                y = y_axis,
                color = '#000000',
                palette = {"G1": "#6794db", "Early S": "#aed17d", "Late S": "#63a678", "G2": "#CC6677", "M": "#fcba03", "Polyploid" : "#9230d9", "Debris" : "#a8a8a8"},
                ec = "none",
                linewidth = 0,
                alpha = 0,
                legend = False,
                ax = Figure.ax_joint)
            
            sns.histplot(
                    
                data = tmp_data,
                y = y_axis,
                color = "#ADACAC",
                ax = Figure.ax_marg_y,
                bins = 100,
                element = "step",
                stat = 'density',
                fill = True,
                lw = 0.5)
            
            Figure.ax_joint.set_xticklabels("G2/M")
            
        else :
        
            sns.scatterplot(
                
                data = tmp_data,
                x = x_axis,
                y = y_axis,
                color = '#000000',
                hue = "cell_cycle_detailed",
                palette = {"G1": "#6794db", "Early S": "#aed17d", "Late S": "#63a678", "G2": "#CC6677", "M": "#fcba03", "Polyploid" : "#9230d9", "Debris" : "#a8a8a8"},
                ec = "none",
                linewidth = 0,
                alpha = 0,
                legend = False,
                ax = Figure.ax_joint)

            sns.histplot(
                
                data = tmp_data,
                y = y_axis,
                color = "#ADACAC",
                ax = Figure.ax_marg_y,
                bins = 100,
                element = "step",
                stat = 'density',
                fill = True,
                lw = 0.5)

            sns.stripplot(
                
                data = tmp_data,
                x = x_axis,
                y = y_axis,
                color = '#000000',
                hue = "cell_cycle_detailed",
                palette = {"G1": "#6794db", "Early S": "#aed17d", "Late S": "#63a678", "G2": "#CC6677", "M": "#fcba03", "Polyploid" : "#9230d9", "Debris" : "#a8a8a8"},
                ec = "none",
                size = 3,
                linewidth = 0,
                alpha = 0.1,
                ax = Figure.ax_joint)
            
            Figure.ax_joint.legend_.remove()
        
        Figure.set_axis_labels(" \n \n",
                               "\nMean H3-P intensity\n$(log_{2}$, normalised)")
        
        Figure.ax_joint.set_xticklabels(["G2/M"])
        
        Figure.fig.set_figwidth(0.5)
        Figure.fig.set_figheight(2)
            
        Figure.savefig(path_export + "H3P_" + cell_line + "_" + condition + ".pdf", dpi = 300)
        Figure.savefig(path_export + "H3P_" + cell_line + "_" + condition + ".png", dpi = 1000)
        del(tmp_data)
 


