import numpy as np
from omero_screen.general_functions import scale_img, save_fig, color_label, get_well_pos
from omero_screen.feature_extraction import Image, ImageProperties
from skimage import io
import matplotlib.pyplot as plt
import tqdm
import random
import pandas as pd


# Functions to loop through well object, assemble data for images and ave quality control data

def well_loop(exp_data, well, flatfield_corr):
    df_well = pd.DataFrame()
    df_well_quality = pd.DataFrame()
    image_number = len(list(well.listChildren()))
    print(
        "\nSegmenting and Analysing Images\n------------------------------------------------------------------------------------------------------\n")
    for number in tqdm.tqdm(range(image_number)):
        omero_img = well.getImage(number)
        image = Image(well, omero_img, exp_data, flatfield_corr)
        segmentation_quality_control(exp_data, well, image)
        data = ImageProperties(well, image, exp_data)
        df_well = pd.concat([df_well, data.image_properties])
        df_well_quality = pd.concat([df_well_quality, image.quality_df])
    return df_well, df_well_quality


def segmentation_quality_control(exp_data, well, image):
    well_pos = get_well_pos(exp_data.plate_layout, well.getId())
    path = exp_data.quality_ctr_path
    index = well.countWellSample()
    #number = random.randint(0, index - 1)  # option to choose random number
    # decided to choose first image in well. Saves me finding the image ID
    number = 1
    if index == number:
        segmentation_image(path, well_pos, image.corr_img_dict, image.n_mask, image.cyto_mask)
        save_example_tiff(path, image.corr_img_dict, well_pos)


def segmentation_image(path: 'pathlib.Path object', well_pos: str, img_dict: dict, n_mask: np.ndarray,
                       cyto_mask: np.ndarray) -> None:
    """
    Generate matplotlib image for segmentation check and save to path (quality control)
    :param path: path
    :param well_pos: string for well position from exp_data object
    :param img_dict: dictionary of iamges from Image object
    :param n_mask: nuclear segmentation mask from Image object
    :param cyto_mask: cytoplasm segmentation mask from Image object
    :return: None, just saves the pdf file using well number as ID
    """
    img_list = get_img_list(img_dict, n_mask, cyto_mask)
    title_list = ["DAPI image", "Tubulin image", "DAPI segmentation", "Tubulin image"]
    fig, ax = plt.subplots(ncols=4, figsize=(16, 7))
    for i in range(4):
        ax[i].axis('off')
        ax[i].imshow(img_list[i])
        ax[i].title.set_text(title_list[i])
        save_fig(path, f'{well_pos}_segmentation_check')


def get_img_list(img_dict: dict, n_mask: np.ndarray, cyto_mask: np.ndarray) -> list:
    """
    Prepare images for segmentation check
    :param img_dict: image dictionary from Image object
    :param n_mask: nuclear segmentation mask from Image object
    :param cyto_mask: cytoplasm segmentation mask from Image object
    :return: list of images to be fed to matplotlib function
    """
    dapi_img = scale_img(img_dict['DAPI'])
    tub_img = scale_img(img_dict['Tub'])
    dapi_color_labels = color_label(n_mask, dapi_img)
    tub_color_labels = color_label(cyto_mask, dapi_img)
    return [dapi_img, tub_img, dapi_color_labels, tub_color_labels]


def save_example_tiff(path: 'pathlib.Path object', img_dict: dict, well_pos: str) -> None:
    """
    Combines arrays from image_dict and saves images as tif files
    :param path: path to quality control folder
    :param img_dict: dictionary with images from Image object
    :param well_pos: well position from well object
    :return: None, just saves tif file
    """
    comb_image = np.dstack(list(img_dict.values()))
    io.imsave(path, f'{well_pos}_segmentation_check.tif', comb_image)

