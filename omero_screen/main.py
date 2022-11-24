from omero_screen.general_functions import omero_connect
from omero_screen.data_structure import MetaData, ExpPaths
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen import EXCEL_PATH
from omero_screen.omero_loop import *
import pandas as pd


@omero_connect
def main(excel_path=EXCEL_PATH, conn=None):
    meta_data = MetaData(excel_path)
    exp_paths = ExpPaths(conn, meta_data)
    df_final = pd.DataFrame()
    df_quality_control = pd.DataFrame()
    for count, ID in enumerate(meta_data.plate_layout["Well_ID"]):
        print(f"Analysing well {meta_data.well_pos(ID)} - {count + 1} of {meta_data.plate_length}.\n{SEPARATOR}")
        well = conn.getObject("Well", ID)
        flatfield_dict = flatfieldcorr(well, meta_data, exp_paths)
        well_data, well_quality = well_loop(well, meta_data, exp_paths, flatfield_dict)
        df_final = pd.concat([df_final, well_data])
        df_quality_control = pd.concat([df_quality_control, well_quality])
    df_final = pd.concat([df_final.iloc[:,- 7:], df_final.iloc[:,:-7]], axis=1)
    df_final.to_csv(exp_paths.final_data / f"{exp_paths.plate_name}_final_data.csv")
    df_quality_control.to_csv(exp_paths.quality_ctr / f"{exp_paths.plate_name}_quality_data.csv")


if __name__ == '__main__':
    main()
