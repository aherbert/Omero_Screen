from omero_screen.general_functions import omero_connect
from omero_screen.data_structure import ExperimentData
from omero_screen.flatfield_corr import FlatFieldCorr
from omero_screen import EXCEL_PATH, SEPARATOR
from omero_loop import *
import pandas as pd




@omero_connect
def main(conn=None, excel_path=EXCEL_PATH):
    df_final = pd.DataFrame()
    df_quality_control = pd.DataFrame()
    exp_data = ExperimentData(excel_path, conn=conn)
    total = len(exp_data.plate_layout["Well_ID"] + 1)
    for count, ID in enumerate(exp_data.plate_layout["Well_ID"]):
        well = conn.getObject("Well", ID)
        well_pos = get_well_pos(exp_data.plate_layout, well.getId())
        flatfield_corr = FlatFieldCorr(well, exp_data, well_pos)
        print(f"\nAnalysing well {well_pos} / {count + 1} of {total}\n{SEPARATOR}")
        well_data, well_quality = well_loop(exp_data, well, flatfield_corr)
        df_final = pd.concat([df_final, well_data])
        df_quality_control = pd.concat([df_quality_control, well_quality])
    # switch columns around
    df_final = df_final[
        list(df_final.columns.values[- 7:]) + list(df_final.columns.values[:len(df_final.columns.values) - 7])]
    df_final.to_csv(exp_data.final_data_path / f"{exp_data.plate_name}final_data.csv")
    df_quality_control.to_csv(exp_data.quality_ctr_path / f"{exp_data.plate_name}quality_data.csv")


if __name__ == '__main__':
    main()
