from omero_screen import SEPARATOR
from omero_screen.image_analysis import Image, ImageProperties
import tqdm
import pandas as pd


# Functions to loop through well object, assemble data for images and ave quality control data

def well_loop(well, meta_data, exp_paths, flatfield_dict):
    print(f"Segmenting and Analysing Images\n{SEPARATOR}")
    df_well = pd.DataFrame()
    df_well_quality = pd.DataFrame()
    image_number = len(list(well.listChildren()))
    for number in tqdm.tqdm(range(image_number)):
        omero_img = well.getImage(number)
        image = Image(well, omero_img, meta_data, exp_paths, flatfield_dict)
        image_data = ImageProperties(well, image, meta_data, exp_paths)
        if number == 1:
            image.segmentation_figure()
            image.save_example_tiff()
        df_image = image_data.image_df
        df_image_quality = image_data.quality_df
        df_well = pd.concat([df_well, df_image])
        df_well_quality = pd.concat([df_well_quality, df_image_quality])
    return df_well, df_well_quality
