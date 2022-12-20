import os
import pandas as pd
from cell_cycle_distribution_functions import fun_normalise, fun_CellCycle
from os import listdir
from omero.gateway import BlitzGateway

def plate_get_well(plate_lists,conn):
    """  get the corresponding well_id according to plate and stored in dictionary

    :param plate_list: the list of plate id
    :param conn: connect to omero
    :return: A dictionary, the keys are plate id, the values are list of well id

    for example myDict['948']=[10620,10621,...]
    """
    plate_well_dict={}
    for plate in plate_lists:
        plate_well_dict[str(plate)] = []
        plate = conn.getObject("Plate", plate)
        for well in plate.listChildren():
            plate_well_dict[str(plate.id)].append(well.id)
    return plate_well_dict


def dict_wells_corr(F_dir,conn):
    """
    # %% Importing RAW data from path of file & excluding inconsistent wells

    :param F_dir: the path of RAW
    :return: A dataframe and adding a columns called the cell_id that group by "plate_id", "well_id", "image_id", "Cyto_ID", which make sure that each segmented cell will have individual number.
    """

    list_files = list(filter(lambda file: ".csv" in file, listdir(F_dir + "/single_cell_data/")))
    data_raw = pd.DataFrame()
    for file in list_files:
        tmp_data = pd.read_csv(F_dir + "single_cell_data/" + file, sep=",")
        if 'plate_id' and 'well_id' in tmp_data.columns.values.tolist():
            tmp_plates = tmp_data["plate_id"].unique()
            Dict_plate_well = plate_get_well(tmp_plates,conn=conn)
            for tmp_plate in tmp_plates:
                if tmp_plate in Dict_plate_well.keys():
                    tmp_data = tmp_data.copy().loc[tmp_data["well_id"].isin(Dict_plate_well[tmp_plate])]
                data_raw = pd.concat([data_raw, tmp_data])
        else:
            print('Not Exist: plate_id, well_id')
        # del ([tmp_data, file])
    data_raw.loc[:, "cell_id"] = data_raw.groupby(["plate_id", "well_id", "image_id", "Cyto_ID"]).ngroup()
    return data_raw


def assign_cell_cycle_phase(data, *args):
    """
    # %% Selecting parameters of interest and aggregating counts of nuclei and total cellular DAPI signal
      %% Normalising selected parameters & assigning cell cycle phases

    :param data:A data frame that including the necessary parameters
    :param args:interesting parameters used for group data frame to aggregating counts of nuclei and total cellular DAPI signal
    :return: data_IF (A dataframe assigned a cell cycle phase to each cell), data_thresholds (threshold values of normalised integrated DAPI intensities)
    """

    data_IF = data.groupby(list(args)).agg(
    nuclei_count=("label", "count"),
    nucleus_area=("area_nucleus", "sum"),
    DAPI_total=("integrated_int_DAPI", "sum")).reset_index()
    data_IF["Condition"] = data_IF["Condition"].astype(str)

    data_IF = fun_normalise(data=data_IF, values=["DAPI_total", "intensity_mean_EdU_cell", "intensity_mean_H3P_cell",
                                                  "area_cell"])
    data_IF, data_thresholds = fun_CellCycle(data=data_IF, ctr_col="Condition", ctr_cond="NT")
    return data_IF, data_thresholds

def cell_cycle_summary(data_dir,conn):
    """
    # %% Establishing proportions (%) of cell cycle phases
    calling functions of dict_wells_corr and assign_cell_cycle_phase to get the data_IF, data_thresholds
    :param data_dir: the path of RAW data frame
    :return: A dataframe summarized each cell cycle phase
    """
    data_IF, data_thresholds = assign_cell_cycle_phase(dict_wells_corr(data_dir, conn),"experiment", "plate_id", "well_id", "image_id",
                            "Cell_Line", "Condition", "Cyto_ID", "cell_id", "area_cell",
                            "intensity_mean_EdU_cell",
                            "intensity_mean_H3P_cell")
    data_cell_cycle = pd.DataFrame()
    for experiment in data_IF["experiment"].unique():
        for cell_line in data_IF.loc[data_IF["experiment"] == experiment]["Cell_Line"].unique():
            for condition in data_IF.loc[(data_IF["experiment"] == experiment) &
                                             (data_IF["Cell_Line"] == cell_line)]["Condition"].unique():
                tmp_data = data_IF.loc[(data_IF["experiment"] == experiment) &(data_IF["Cell_Line"] == cell_line) &
                                             (data_IF["Condition"] == condition)]
                n = len(tmp_data)

                tmp_data = tmp_data.groupby(["experiment", "plate_id", "Cell_Line", "Condition", "cell_cycle"],
                                                as_index=False).agg(
                        count=("cell_id", "count"),
                        nuclear_area_mean=("nucleus_area", "mean"),
                        DAPI_total_mean=("DAPI_total_norm", "mean"),
                        area_cell_mean=("area_cell_norm", "mean"))
                # calculate proportions for each cell cycle phase
            tmp_data["n"] = n
            tmp_data["percentage"] = (tmp_data["count"] / tmp_data["n"]) * 100
            data_cell_cycle = pd.concat([data_cell_cycle, tmp_data])
    return data_cell_cycle.groupby(["Cell_Line", "cell_cycle", "Condition"], as_index=False).agg(percentage_mean=("percentage", "mean"), percentage_sd=("percentage", "std"))


def save_folder(Path_data,exist_ok=True):
    """  Establishing path to the data and creating a folder to save exported .pdf files

    :param Path_data: the path used to save the exported .pdf files
    :return: This method does not return any value.
    """
    # path_data = "/Users/Lab/Desktop/CDK1ArrestCheck_20hr_1/"
    path_export = Path_data+ "/Figures/"
    if exist_ok==True:
        os.makedirs(path_export, exist_ok=True)
    else:
        isExist = os.path.exists(path_export)
        try:
            if isExist==False:
                os.makedirs(path_export)
        except FileExistsError:
            print('File already exists')
    return path_export




if __name__ == '__main__':

    conn = BlitzGateway('hy274', 'omeroreset', host='ome2.hpc.susx.ac.uk')
    conn.connect()
    df = cell_cycle_summary('/Users/hh65/Desktop/221128_DepMap_Exp8_siRNAscreen_Plate1_72hrs/',conn=conn)
    # df=dict_wells_corr(F_dir='/Users/hh65/Desktop/221128_DepMap_Exp8_siRNAscreen_Plate1_72hrs/',conn=conn)
    #
    # df_2=assign_cell_cycle_phase(df,"experiment", "plate_id", "well_id", "image_id",
    # #                         "Cell_Line", "Condition", "Cyto_ID", "cell_id", "area_cell",
    # #                         "intensity_mean_EdU_cell",
    # #                         "intensity_mean_H3P_cell")
    df.to_csv('~/Desktop/test.csv')



