from omero_screen.image_analysis import Image, ImageProperties
from omero_screen.image_analysis_nucleus import NucImage, NucImageProperties
from omero_screen import Defaults
import tqdm
import pandas as pd
import pathlib




# Functions to loop through well object, assemble data for images and ave quality control data

def well_loop(conn, well, meta_data, exp_paths, flatfield_dict):
    print(f"\nSegmenting and Analysing Images\n")
    df_well = pd.DataFrame()
    df_well_quality = pd.DataFrame()
    image_number = len(list(well.listChildren()))
    for number in tqdm.tqdm(range(image_number)):
        omero_img = well.getImage(number)
        if 'Tub' in meta_data.channels.keys():
            image = Image(conn, well, omero_img, meta_data, exp_paths, flatfield_dict)
            image_data = ImageProperties(well, image, meta_data, exp_paths)
        else:
            image = NucImage(well, omero_img, meta_data, exp_paths, flatfield_dict)
            image_data = NucImageProperties(well, image, meta_data, exp_paths)
        df_image = image_data.image_df
        df_image_quality = image_data.quality_df
        df_well = pd.concat([df_well, df_image])
        df_well_quality = pd.concat([df_well_quality, df_image_quality])
        if number == 1 and Defaults['DEBUG']:
            image.segmentation_figure()
            image.save_example_tiff()
    return df_well, df_well_quality
