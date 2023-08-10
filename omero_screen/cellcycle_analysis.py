import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from omero_screen.general_functions import save_fig


STYLE = Path("../data/Style_01.mplstyle")
plt.style.use(STYLE)
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]


norm_colums = (
    "integrated_int_DAPI",
    "intensity_mean_EdU_nucleus",
)  # Default columns for cell cycle normalisation


def cellcycle_analysis(
    df: pd.DataFrame, H3: bool = False, cyto: bool = True
) -> pd.DataFrame:
    """
    Function to normalise cell cycle data using normalise and assign_ccphase functions for each cell line
    :param df: single cell data from omeroscreen
    :param cyto: True if cytoplasmic data is present
    :return: dataframe with cell cycle and cell cycle detailed columns
    """
    df1 = df.copy()
    if H3:
        values = [
            "integrated_int_DAPI",
            "intensity_mean_EdU_nucleus",
            "intensity_mean_H3P_nucleus",
        ]
        df1["intensity_mean_H3P_nucleus"] = (
            df1["intensity_mean_H3P_nucleus"] - df1["intensity_min_H3P_nucleus"] + 1
        )
    else:
        values = ["integrated_int_DAPI", "intensity_mean_EdU_nucleus"]
    df1["intensity_mean_EdU_nucleus"] = (
        df1["intensity_mean_EdU_nucleus"] - df1["intensity_min_EdU_nucleus"] + 1
    )
    if cyto:
        df_agg = agg_multinucleates(df1)
        df_agg_corr = delete_duplicates(df_agg)
    else:
        df_agg_corr = df1.copy()
    tempfile = pd.DataFrame()
    for cell_line in df_agg_corr["cell_line"].unique():
        df1 = df_agg_corr.loc[df_agg_corr["cell_line"] == cell_line]
        df_norm = normalise(df1, values)
        df_norm["integrated_int_DAPI_norm"] = df_norm["integrated_int_DAPI_norm"] * 2
        tempfile = pd.concat([tempfile, df_norm])
    return assign_ccphase(data=tempfile, H3=H3)


# Helper Functions for cell cycle normalisation
def agg_multinucleates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to aggregate multinucleates by summing up the nucleus area and DAPI intensity
    :param df: single cell data from omeroscreen
    :return: corrected df with aggregated multinucleates
    """
    num_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)
    str_cols = list(df.select_dtypes(include=["object"]).columns)
    # define the aggregation functions for each column
    agg_functions = {}
    for col in num_cols:
        if col in ["integrated_int_DAPI", "area_nucleus"]:
            agg_functions[col] = "sum"
        elif "max" in col and "nucleus" in col:
            agg_functions[col] = "max"
        elif "min" in col and "nucleus" in col:
            agg_functions[col] = "min"
        else:
            agg_functions[col] = "mean"
    return df.groupby(str_cols + ["image_id", "Cyto_ID"], as_index=False).agg(
        agg_functions
    )


def delete_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to delete duplicates from the agg_multinucleate dataframe
    :param df: dataframe from agg_multinucleates function
    :return: df with deleted duplicates
    """
    temp_data = pd.DataFrame()
    for image in df["image_id"].unique():
        image_data = df.loc[df.image_id == image].drop_duplicates()
        temp_data = pd.concat([temp_data, image_data])
    return temp_data


def normalise(df: pd.DataFrame, values) -> pd.DataFrame:
    """
    Data normalisation function: Identifies the most frequent intensity value and sets it to
    1 by division. For DAPI data this is set to two, to reflect diploid (2N) state of chromosomes
    :param df: dataframe from delete_duplicates function
    :param values:
    :return:
    """
    norm_df = pd.DataFrame()
    for cell_line in df["cell_line"].unique():
        tmp_data = df.copy().loc[(df["cell_line"] == cell_line)]
        tmp_bins = 10000
        for value in values:
            y, x = np.histogram(tmp_data[value], bins=tmp_bins)
            max_value = x[np.where(y == np.max(y))]
            tmp_data[f"{value}_norm"] = tmp_data[value] / max_value[0]
        norm_df = pd.concat([norm_df, tmp_data])
    return norm_df


def assign_ccphase(data: pd.DataFrame, H3) -> pd.DataFrame:
    """
    Assigns a cell cycle phase to each cell based on normalised EdU and DAPI intensities.
    :param data: dataframe from normalise function
    :return: dataframe with cell cycle assignment
    (col: cellcycle (Sub-G1, G1, S, G2/M Polyploid
    and col: cellcycle_detailed with Early S/Late S and Polyploid (non-replicating)
    Polyploid (replicating))
    """
    if H3:
        data["cell_cycle_detailed"] = data.apply(thresholdingH3, axis=1)
    else:
        data["cell_cycle_detailed"] = data.apply(thresholding, axis=1)
    data["cell_cycle"] = data["cell_cycle_detailed"]
    data["cell_cycle"] = data["cell_cycle"].replace(["Early S", "Late S"], "S")
    data["cell_cycle"] = data["cell_cycle"].replace(
        ["Polyploid (non-replicating)", "Polyploid (replicating)"], "Polyploid"
    )
    return data


def thresholding(
    data: pd.DataFrame,
    DAPI_col: str = "integrated_int_DAPI_norm",
    EdU_col="intensity_mean_EdU_nucleus_norm",
) -> str:
    """
    Function to assign cell cycle phase based on thesholds of normalised EdU and DAPI intensities
    :param data: data from assign_ccphase function
    :param DAPI_col: default 'integrated_int_DAPI_norm'
    :param EdU_col: default 'intensity_mean_EdU_nucleus_norm'
    :return: string indicating cell cycle phase
    """
    if data[DAPI_col] <= 1.5:
        return "Sub-G1"

    elif 1.5 < data[DAPI_col] < 3 and data[EdU_col] < 3:
        return "G1"

    elif 3 <= data[DAPI_col] < 5.5 and data[EdU_col] < 3:
        return "G2/M"

    elif 1.5 < data[DAPI_col] < 3 and data[EdU_col] > 3:
        return "Early S"

    elif 3 <= data[DAPI_col] < 5.5 and data[EdU_col] > 3:
        return "Late S"

    elif data[DAPI_col] >= 5.5 and data[EdU_col] < 3:
        return "Polyploid (non-replicating)"

    elif data[DAPI_col] >= 5.5 and data[EdU_col] > 3:
        return "Polyploid (replicating)"

    else:
        return "Unassigned"


def thresholdingH3(
    data: pd.DataFrame,
    DAPI_col: str = "integrated_int_DAPI_norm",
    EdU_col="intensity_mean_EdU_nucleus_norm",
    H3P_col="intensity_mean_H3P_nucleus_norm",
) -> str:
    """
    Function to assign cell cycle phase based on thresholds of normalised EdU, DAPI and H3P intensities
    :param data: data from assign_ccphase function
    :param DAPI_col: default 'integrated_int_DAPI_norm'
    :param EdU_col: default 'intensity_mean_EdU_nucleus_norm'
    :param H3P_col: default 'intensity_mean_H3P_nucleus_norm'
    :return: string indicating cell cycle phase
    """
    if data[DAPI_col] <= 1.5:
        return "Sub-G1"

    elif 1.5 < data[DAPI_col] < 3 and data[EdU_col] < 3:
        return "G1"

    elif 3 <= data[DAPI_col] < 5.5 and data[EdU_col] < 3 and data[H3P_col] < 5:
        return "G2"

    elif 3 <= data[DAPI_col] < 5.5 and data[EdU_col] < 3 and data[H3P_col] > 5:
        return "M"

    elif 1.5 < data[DAPI_col] < 3 and data[EdU_col] > 3:
        return "Early S"

    elif 3 <= data[DAPI_col] < 5.5 and data[EdU_col] > 3:
        return "Late S"

    elif data[DAPI_col] >= 5.5 and data[EdU_col] < 3:
        return "Polyploid (non-replicating)"

    elif data[DAPI_col] >= 5.5 and data[EdU_col] > 3:
        return "Polyploid (replicating)"

    else:
        return "Unassigned"


# 3 Cell Cycle Proportion Analysis


def cellcycle_prop(df: pd.DataFrame, cell_cycle: str = "cell_cycle") -> pd.DataFrame:
    """
    Function to calculate the proportion of cells in each cell cycle phase
    :param df_norm: dataframe from assign_ccphase function
    :param cell_cycle: choose column cell_cycle or cell_cycle_detailed, default 'cell_cycle'
    :return: grouped dataframe with cell cycle proportions
    """
    df_ccphase = (
        df.groupby(["plate_id", "well", "cell_line", "condition", cell_cycle])[
            "experiment"
        ].count()
        / df.groupby(["plate_id", "well", "cell_line", "condition"])[
            "experiment"
        ].count()
        * 100
    )
    return df_ccphase.reset_index().rename(columns={"experiment": "percent"})


def prop_pivot(df: pd.DataFrame, well, H3):
    """
    Function to pivot the cell cycle proportion dataframe and get the mean and std of each cell cycle phase
    This will be the input to plot the stacked barplots with errorbars.
    :param df_prop: dataframe from cellcycle_prop function
    :param conditions: list of conditions sort the order of data
    :param H3: boolean, default False, if True the function will use M phase instead of G2/M based on H3 staining
    :return: dataframe to submit to the barplot function
    """
    df_prop = cellcycle_prop(df)
    if H3:
        cc_phases = ["Sub-G1", "G1", "S", "G2", "M", "Polyploid"]
    else:
        cc_phases = ["Sub-G1", "G1", "S", "G2/M", "Polyploid"]

    df_prop1 = df_prop.loc[
        df_prop["well"] == well, ["well", "cell_cycle", "percent"]
    ].pivot_table(columns=["cell_cycle"], index=["well"])
    df_prop1.columns = df_prop1.columns.droplevel(0)

    # Reindex the DataFrame to include all cell cycle phases and fill missing values with 0
    df_prop1 = df_prop1.reindex(columns=cc_phases, fill_value=0)

    return df_prop1


def plot_histogram(ax, data):
    sns.histplot(data=data, x="integrated_int_DAPI_norm", ax=ax)
    ax.set_xlabel(None)
    ax.set_xscale("log", base=2)
    ax.set_xlim([1, 16])
    ax.xaxis.set_visible(False)
    ax.set_ylabel("Frequency")


def plot_scatter(ax, data, H3: bool):
    if H3:
        phases = ["Sub-G1", "G1", "S", "G2", "M", "Polyploid"]
    else:
        phases = ["Sub-G1", "G1", "S", "G2/M", "Polyploid"]
    sns.scatterplot(
        data=data,
        x="integrated_int_DAPI_norm",
        y="intensity_mean_EdU_nucleus_norm",
        hue="cell_cycle",
        hue_order=phases,
        s=5,
        alpha=0.8,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.grid(False)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(int(x))))
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xlim([1, 16])
    ax.set_xlabel("norm. DNA content")
    ax.set_ylabel("norm. EdU intesity")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(int(x))))
    ax.legend().remove()
    ax.axvline(x=3, color="black", linestyle="--")
    ax.axhline(y=3, color="black", linestyle="--")
    sns.kdeplot(
        data=data,
        x="integrated_int_DAPI_norm",
        y="intensity_mean_EdU_nucleus_norm",
        fill=True,
        cmap="rocket_r",
        alpha=0.3,
        ax=ax,
    )


def cellcycle_barplot(ax, df, well, H3):
    df_mean = prop_pivot(df, well, H3)
    df_mean.plot(kind="bar", stacked=True, width=0.75, ax=ax)
    ax.set_ylim(0, 110)
    ax.set_xlabel("")  # Remove the x-axis label)
    if H3:
        legend = ax.legend(
            ["Sub-G1", "G1", "S", "G2", "M", "Polyploid"], title="CellCyclePhase"
        )
        ax.set_ylabel("% of population")
    else:
        legend = ax.legend(
            ["Sub-G1", "G1", "S", "G2/M", "Polyploid"], title="CellCyclePhase"
        )
    # Get current handles and labels
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]
    # Clear the current legend
    legend.remove()
    # Create a new legend with the reversed handles and labels
    legend = ax.legend(handles, labels, title="CellCyclePhase")
    frame = legend.get_frame()
    frame.set_alpha(0.5)
    ax.set_ylabel("% of population")
    ax.grid(False)


def combplot(
    df,
    well,
    H3=False,
):
    df1 = df[df.well == well]

    fig = plt.figure(figsize=(5, 5))
    gs = GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1])
    y_max = df["intensity_mean_EdU_nucleus_norm"].quantile(0.99) * 1.5
    y_min = df["intensity_mean_EdU_nucleus_norm"].quantile(0.01) * 0.8

    ax = fig.add_subplot(gs[0, 0])
    plot_histogram(ax, df1)
    ax.set_title(f"{well}, {len(df1)} cells", size=12, weight="bold")
    ax = fig.add_subplot(gs[1, 0])
    plot_scatter(ax, df1, H3)
    ax.set_ylim([y_min, y_max])
    ax.grid(visible=False)
    # Add the subplot spanning both rows
    ax_last = fig.add_subplot(gs[:, -1])
    ax_last.grid(visible=False)
    # Add the barplot to the subplot
    cellcycle_barplot(ax_last, df1, well, H3)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    df = pd.read_csv("/Users/hh65/Desktop/test_plate01/test_plate01_final_data.csv")
    df_cc = cellcycle_analysis(df)
    fig = combplot(df_cc, "C2", H3=False)
    plt.show()
