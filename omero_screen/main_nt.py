from omero_screen import Defaults
from omero_screen.general_functions import omero_connect
from omero_screen.metadata import MetaData, ExpPaths
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.loops import well_loop
import pandas as pd
import torch


@omero_connect
def main(plate_id, options=None, conn=None):
    # if dict:
    #     Defaults.update(options)

    meta_data = MetaData(plate_id, conn)
    exp_paths = ExpPaths(meta_data)
    flatfield_dict = flatfieldcorr(meta_data, exp_paths, conn)

    with open(
        Defaults["DEFAULT_DEST_DIR"] + "/" + Defaults["DEFAULT_SUMMARY_FILE"], "a"
    ) as f:
        print(meta_data.plate, file=f)

    if torch.cuda.is_available():
        print("Using Cellpose with GPU.")
    else:
        print("Using Cellpose with CPU")
    df_final = pd.DataFrame()
    df_quality_control = pd.DataFrame()
    for count, well in enumerate(list(meta_data.plate_obj.listChildren())):
        ann = well.getAnnotation(Defaults["NS"])
        try:
            cell_line = dict(ann.getValue())["cell_line"]
        except KeyError:
            cell_line = dict(ann.getValue())["Cell_Line"]
        if cell_line != "Empty":
            message = f"{exp_paths.separator}\nAnalysing well row:{well.row}/col:{well.column} - {count + 1} of {meta_data.plate_length}."
            print(message)
            well_data, well_quality = well_loop(
                well, meta_data, exp_paths, flatfield_dict
            )
            df_final = pd.concat([df_final, well_data])
            df_quality_control = pd.concat([df_quality_control, well_quality])
    cols = df_final.columns.tolist()
    i = cols.index("experiment")
    df_final.to_csv(
        exp_paths.final_data / f"{meta_data.plate}_final_data.csv",
        columns=cols[i:] + cols[:i],
    )
    df_quality_control.to_csv(
        exp_paths.final_data / f"{meta_data.plate}_quality_ctr.csv"
    )


if __name__ == "__main__":
    main(1237)
