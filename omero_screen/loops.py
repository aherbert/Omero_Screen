import omero_screen
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from omero_screen import Defaults
from omero_screen.metadata import MetaData, ProjectSetup
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.image_analysis import Image, ImageProperties
from omero_screen.image_analysis_nucleus import NucImage, NucImageProperties
from omero_screen.omero_functions import load_fig, load_csvdata, delete_annotations
from omero_screen.quality_figure import quality_control_fig
from omero_screen.cellcycle_analysis import cellcycle_analysis, combplot, cellcycle_prop
from omero_screen.general_functions import omero_connect

from omero.gateway import BlitzGateway
import tqdm
import torch
import pandas as pd
import pathlib
from typing import Tuple
import logging

logger = logging.getLogger("omero-screen")
# Functions to loop through well object, assemble data for images and ave quality control data


def well_loop(conn, well, metadata, project_data, flatfield_dict):
    print(f"\nSegmenting and Analysing Images\n")
    df_well = pd.DataFrame()
    df_well_quality = pd.DataFrame()
    image_number = len(list(well.listChildren()))
    for number in tqdm.tqdm(range(image_number)):
        omero_img = well.getImage(number)
        if "Tub" in metadata.channels.keys():
            image = Image(conn, well, omero_img, metadata, project_data, flatfield_dict)
            image_data = ImageProperties(well, image, metadata)
        else:
            image = NucImage(
                conn, well, omero_img, metadata, project_data, flatfield_dict
            )
            image_data = NucImageProperties(well, image, metadata)
        df_image = image_data.image_df
        df_image_quality = image_data.quality_df
        df_well = pd.concat([df_well, df_image])
        df_well_quality = pd.concat([df_well_quality, df_image_quality])
    return df_well, df_well_quality


def plate_loop(plate_id: int, conn: BlitzGateway):
    """
    Main loop to process a plate.
    :param plate_id: ID of the plate
    :param conn: Connection to OMERO
    :return: Two DataFrames containing the final data and quality control data
    """
    logger.info(f"Processing plate {plate_id}")
    metadata = MetaData(conn, plate_id=plate_id)
    logger.debug(f"Chennel Metadata: {metadata.channels}")
    plate_name = metadata.plate_obj.getName()
    project_data = ProjectSetup(plate_id, conn)
    flatfield_dict = flatfieldcorr(metadata, project_data, conn)

    # Add plate name to summary file
    with open(
        Defaults["DEFAULT_DEST_DIR"] + "/" + Defaults["DEFAULT_SUMMARY_FILE"], "a"
    ) as f:
        print(plate_name, file=f)

    print_device_info()

    df_final, df_quality_control = process_wells(
        metadata, project_data, flatfield_dict, conn
    )
    logger.debug(f"Final data sample: {df_final.head()}")
    # check conditiosn for cell cycle analysis
    keys = metadata.channels.keys()

    if "EdU" in keys:
        try:
            H3 = "H3P" in keys
            cyto = "Tub" in keys

            if H3 and cyto:
                df_final_cc = cellcycle_analysis(df_final, H3=True, cyto=True)
            elif H3:
                df_final_cc = cellcycle_analysis(df_final, H3=True)
            elif not cyto:
                df_final_cc = cellcycle_analysis(df_final, cyto=False)
            else:
                df_final_cc = cellcycle_analysis(df_final)
            wells = list(metadata.plate_obj.listChildren())
            add_welldata(wells, df_final_cc, conn)
        except Exception as e:
            print(e)
            df_final_cc = None
    else:
        df_final_cc = None

    save_results(df_final, df_final_cc, df_quality_control, metadata, plate_name, conn)
    return df_final, df_final_cc, df_quality_control


def print_device_info() -> None:
    """
    Print whether the code is using Cellpose with GPU or CPU.
    """
    if torch.cuda.is_available():
        print("Using Cellpose with GPU.")
    else:
        print("Using Cellpose with CPU.")


from typing import Tuple


def process_wells(
    metadata: MetaData,
    project_data: ProjectSetup,
    flatfield_dict: dict,
    conn: BlitzGateway,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process the wells of the plate.
    :param wells: Wells to be processed
    :param conn: Connection to OMERO
    :param metadata: Metadata associated with the plate
    :param project_data: Project setup data
    :param flatfield_dict: Dictionary containing flatfield correction data
    :return: Two DataFrames containing the final data and quality control data
    """
    df_final = pd.DataFrame()
    df_quality_control = pd.DataFrame()
    for count, well in enumerate(list(metadata.plate_obj.listChildren())):
        ann = well.getAnnotation(Defaults["NS"])
        try:
            cell_line = dict(ann.getValue())["cell_line"]
        except KeyError:
            cell_line = dict(ann.getValue())["Cell_Line"]
        if cell_line != "Empty":
            message = f"{metadata.separator}\nAnalysing well row:{well.row}/col:{well.column} - {count + 1} of {metadata.plate_length}."
            print(message)
            well_data, well_quality = well_loop(
                conn, well, metadata, project_data, flatfield_dict
            )
            df_final = pd.concat([df_final, well_data])
            df_quality_control = pd.concat([df_quality_control, well_quality])

    return df_final, df_quality_control


def save_results(
    df_final: pd.DataFrame,
    df_final_cc,
    df_quality_control: pd.DataFrame,
    metadata: MetaData,
    plate_name: str,
    conn: BlitzGateway,
) -> None:
    """
    Save the results to CSV files.
    :param df_final: DataFrame containing the final data
    :param df_quality_control: DataFrame containing quality control data
    :param plate_name: Name of the plate
    """
    # delete pre-existing data
    delete_annotations(metadata.plate_obj, conn)
    #save and load final data and if available cell cycle data
    path = (
        pathlib.Path(Defaults["DEFAULT_DEST_DIR"]) / f"{metadata.plate_obj.getName()}"
    )
    path.mkdir(exist_ok=True)
    file_path = save_data_csv(
        path, plate_name, '_final_data.csv', df_final
    )
    load_csvdata(metadata.plate_obj, file_path, conn)
    if df_final_cc is not None:
        file_path_cc = save_data_csv(
            path, plate_name, '_final_data_cc.csv', df_final_cc
        )
        load_csvdata(metadata.plate_obj, file_path_cc, conn)
    # load quality control figure
    df_quality_control.to_csv(path / f"{plate_name}_quality_ctr.csv")
    quality_fig = quality_control_fig(df_quality_control)
    load_fig(quality_fig, metadata.plate_obj, f"{plate_name}_quality_ctr", conn)


# TODO Rename this here and in `save_results`
def save_data_csv(path, plate_name, df_name, df):
    result = path / f"{plate_name}{df_name}"

    cols = df.columns.tolist()
    i = cols.index("experiment")
        # save csv files to project directory
    df.to_csv(result, columns=cols[i:] + cols[:i])
    return result


def add_welldata(wells, df_final, conn):
    """
    Add well data to OMERO plate.
    :param plate_id: ID of the plate
    :param conn: Connection to OMERO
    :param df_final: DataFrame containing the final data
    """
    for well in wells:
        well_pos = well.getWellPos()
        if len(df_final[df_final["well"] == well_pos]) > 100:
            fig = combplot(df_final, well_pos)
            delete_annotations(well, conn)
            load_fig(fig, well, well_pos, conn)
        else:
            print(f"Insufficient data for {well_pos}")


if __name__ == "__main__":

    @omero_connect
    def test_well(plate_id, well_id, conn=None):
        well = conn.getObject("Well", well_id)
        metadata = MetaData(conn, plate_id=plate_id)
        project_data = ProjectSetup(plate_id, conn)
        flatfield_dict = flatfieldcorr(metadata, project_data, conn)
        df_well, df_well_quality = well_loop(
            conn, well, metadata, project_data, flatfield_dict
        )

    test_well(1927, 34975)
