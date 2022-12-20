import seaborn as sns
from data_phase_summary import assign_cell_cycle_phase,dict_wells_corr

def plot_scatter_Edu_G2(data_dir,path_export,conn,):
    """
    # %% Plotting & exporting combined EdU ~ DAPI scatter plots
    :param data_dir: str, the path
    :param path_export: Path of the directory to save plot to
    :param kwargs:dict, Parameters of sns.set_context, mappings to override the values in the preset seaborn context dictionaries.
    :return: A folder to save the plots
    """
    data_IF, data_thresholds = assign_cell_cycle_phase(dict_wells_corr(data_dir,conn),"experiment", "plate_id", "well", "well_id", "image_id",
                            "cell_line", "condition", "Cyto_ID", "cell_id", "area_cell",
                            "intensity_mean_EdU_cell",
                            "intensity_mean_H3P_cell")

    for experiment in data_IF["experiment"].unique():
        for cell_line in data_IF.loc[data_IF["experiment"] == experiment]["cell_line"].unique():
            for condition in data_IF.loc[(data_IF["experiment"] == experiment) &
                                             (data_IF["cell_line"] == cell_line)]["condition"].unique():
                tmp_data = data_IF.loc[(data_IF["experiment"] == experiment) &
                                           (data_IF["cell_line"] == cell_line) &
                                           (data_IF["condition"] == condition)]
                tmp_thresholds = data_thresholds.loc[data_thresholds["cell_line"] == cell_line]
                sns.set_context(context='talk',
                                rc={'font.size': 8.0,
                                    'axes.labelsize': 8.0,
                                    'axes.titlesize': 8.0,
                                    'xtick.labelsize': 8.0,
                                    'ytick.labelsize': 8.0,
                                    'legend.fontsize': 3,
                                    'axes.linewidth': 0.5,
                                    'grid.linewidth': 0.5,
                                    'lines.linewidth': 0.5,
                                    'lines.markersize': 2,
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
                Figure = sns.JointGrid(ratio=3,ylim=(
                    data_IF["intensity_mean_H3P_cell_norm"].min() - data_IF["intensity_mean_H3P_cell_norm"].min() * 0.2,
                    data_IF["intensity_mean_EdU_cell_norm"].max()), )
                Figure.ax_joint.set_xscale("log")
                Figure.ax_joint.set_yscale("log")
                Figure.refline(y=tmp_thresholds["EdU_threshold"].values)
                Figure.refline(x=tmp_thresholds["DAPI_low_threshold"].values)
                Figure.refline(x=tmp_thresholds["DAPI_mid_threshold"].values)
                Figure.refline(x=tmp_thresholds["DAPI_high_threshold"].values)
                Figure.set_axis_labels("Integrated Hoechst intensity\n(normalised)\n",
                                       '\nMean EdU intensity\n(normalised)')
                sns.scatterplot(
                    data=tmp_data,
                    x="DAPI_total_norm",
                    y="intensity_mean_EdU_cell_norm",
                    color='#000000',
                    hue="cell_cycle_detailed",
                    palette={"G1": "#6794db", "Early S": "#aed17d", "Late S": "#63a678", "G2": "#CC6677",
                             "M": "#CC6677", "Polyploid": "#b39bcf", "Polyploid (replicating)": "#e3b344",
                             "Sub-G1": "#c7c7c7"},
                    ec="none",
                    linewidth=0,
                    alpha=0.1,
                    legend=False,
                    ax=Figure.ax_joint)
                sns.histplot(
                    data=tmp_data,
                    x="DAPI_total_norm",
                    color="#ADACAC",
                    ax=Figure.ax_marg_x,
                    bins=100,
                    element="step",
                    stat="density",
                    fill=True)
                sns.histplot(
                    data=tmp_data,
                    y="intensity_mean_EdU_cell_norm",
                    color="#ADACAC",
                    ax=Figure.ax_marg_y,
                    bins=100,
                    element="step",
                    stat='density',
                    fill=True)
                Figure.ax_joint.text(
                    data_IF["DAPI_total_norm"].min() + data_IF["DAPI_total_norm"].min() * 0.4,
                    data_IF["intensity_mean_EdU_cell_norm"].max() - data_IF["intensity_mean_EdU_cell_norm"].max() * 0.5,
                    f"{cell_line}\n{condition} µM",
                    horizontalalignment="left",
                    size=7,
                    color="#000000",
                    weight="normal")
                Figure.ax_joint.text(
                    data_IF["DAPI_total_norm"].max() - data_IF["DAPI_total_norm"].min() * 0.4,
                    data_IF["intensity_mean_EdU_cell_norm"].max() - data_IF["intensity_mean_EdU_cell_norm"].max() * 0.3,
                    str(len(tmp_data)),
                    horizontalalignment="right",
                    size=7,
                    color="#000000",
                    weight="normal")
                Figure.fig.set_figwidth(2)
                Figure.fig.set_figheight(2)

                Figure.savefig(path_export + "EdU-DAPI_" + experiment + "_" + cell_line + "_" + condition + ".pdf",
                               dpi=300)
                Figure.savefig(path_export + "EdU-DAPI_" + experiment + "_" + cell_line + "_" + condition + ".png",
                               dpi=1000)
                del (tmp_data)

def plot_distribution_H3_P(path_export,data_dir,conn,):
    """
    # %% Plotting & exporting distributions of H3-P signal in G2/M cells
    :param path_export:Path of the directory to save plot to
    :param data_dir:
    :param kwargs:dict, Parameters of sns.set_context, mappings to override the values in the preset seaborn context dictionaries.
    :return: A folder to save the plots
    """
    data_IF, data_thresholds = assign_cell_cycle_phase(dict_wells_corr(data_dir,conn),"experiment", "plate_id", "well", "well_id", "image_id",
                            "cell_line", "condition", "Cyto_ID", "cell_id", "area_cell",
                            "intensity_mean_EdU_cell",
                            "intensity_mean_H3P_cell")
    for experiment in data_IF["experiment"].unique():
        for cell_line in data_IF.loc[data_IF["experiment"] == experiment]["cell_line"].unique():

            for condition in data_IF.loc[(data_IF["experiment"] == experiment) &
                                         (data_IF["cell_line"] == cell_line)]["condition"].unique():
                print(experiment + " – " + cell_line + " : " + condition)

                tmp_data = data_IF.loc[(data_IF["cell_line"] == cell_line) &
                                       (data_IF["condition"] == condition) &
                                       (data_IF["experiment"] == experiment) &
                                       (data_IF["cell_cycle_detailed"].isin(["G2", "M"]))]

                tmp_thresholds = data_thresholds.loc[data_thresholds["cell_line"] == cell_line]
                sns.set_context(context='talk',
                                rc={'font.size': 8.0,
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
                ylim = (data_IF["intensity_mean_H3P_cell_norm"].min() - data_IF["intensity_mean_H3P_cell_norm"].min() * 0.2,
                        data_IF["intensity_mean_EdU_cell_norm"].max()),)

                Figure.ax_joint.set_yscale("log")

                Figure.refline(y=tmp_thresholds["H3P_threshold"].values)

                Figure.set_axis_labels(" \n \n",
                                       "\nMean H3-P intensity\n(normalised)")

                sns.scatterplot(

                    data=tmp_data,
                    x="cell_line",
                    y="intensity_mean_H3P_cell_norm",
                    color='#000000',
                    hue="cell_cycle_detailed",
                    palette={"G1": "#6794db", "Early S": "#aed17d", "Late S": "#63a678", "G2": "#CC6677",
                             "M": "#fcba03", "Polyploid": "#9230d9", "Debris": "#a8a8a8"},
                    ec="none",
                    linewidth=0,
                    alpha=0.25,
                    legend=False,
                    ax=Figure.ax_joint)

                Figure.ax_joint.set_xticks([cell_line])
                Figure.ax_joint.set_xticklabels(["G2/M"])

                sns.histplot(

                    data=tmp_data,
                    y="intensity_mean_H3P_cell_norm",
                    color="#ADACAC",
                    ax=Figure.ax_marg_y,
                    bins=100,
                    element="step",
                    stat='density',
                    fill=True,
                    lw=0.5)

                Figure.fig.set_figwidth(0.5)
                Figure.fig.set_figheight(2)

                Figure.savefig(path_export + "H3P_" + experiment + "_" + cell_line + "_" + condition + ".pdf", dpi=300)
                Figure.savefig(path_export + "H3P_" + experiment + "_" + cell_line + "_" + condition + ".png", dpi=1000)
                del (tmp_data)

