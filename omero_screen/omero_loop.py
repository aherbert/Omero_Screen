from omero_screen import SEPARATOR
from omero_screen.image_analysis import Image, ImageProperties
import tqdm
import pandas as pd
import pathlib




# Functions to loop through well object, assemble data for images and ave quality control data

def well_loop(well, meta_data, exp_paths, flatfield_dict):
    well_pos = f"row_{well.row}_col{well.column}"

    df_well_path = exp_paths.temp_well_data / f'{well_pos}_df_well'
    df_well_quality_path = exp_paths.temp_well_data / f'{well_pos}_df_well_quality'
    # check if file already exists to load dfs and move on
    if pathlib.Path.exists(df_well_path) and pathlib.Path.exists(df_well_quality_path):
        print(f"\nWell has already been analysed, loading data\n{SEPARATOR}")
        df_well = pd.read_pickle(str(df_well_path))
        df_well.rename(columns={'Cell_Line': 'cell_line', 'Condition': 'condition'}, inplace=True)
        df_well_quality = pd.read_pickle(str(df_well_quality_path))
    # analyse the images to generate the dfs
    else:
        print(f"\nSegmenting and Analysing Images\n{SEPARATOR}")
        df_well = pd.DataFrame()
        df_well_quality = pd.DataFrame()
        image_number = len(list(well.listChildren()))
        for number in tqdm.tqdm(range(image_number)):
            omero_img = well.getImage(number)
            image = Image(well, omero_img, meta_data, exp_paths, flatfield_dict)
            image_data = ImageProperties(well, image, meta_data, exp_paths)
            df_image = image_data.image_df
            df_image_quality = image_data.quality_df
            df_well = pd.concat([df_well, df_image])
            df_well_quality = pd.concat([df_well_quality, df_image_quality])
            df_well.to_pickle(str(df_well_path))
            df_well_quality.to_pickle(str(df_well_quality_path))
    return df_well, df_well_quality
