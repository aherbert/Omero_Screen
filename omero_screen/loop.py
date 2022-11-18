from omero_screen.general_functions import scale_img, save_fig, color_label, get_well_pos, omero_connect
from omero_screen.data_structure import ExperimentData
from omero_screen.feature_extraction import Image, ImageProperties
from omero_screen.flatfield_corr import FlatFieldCorr
import matplotlib.pyplot as plt
from skimage import color, io
import tqdm
import random
import pandas as pd



def get_img_list(img_dict, n_mask, cyto_mask):
    dapi_img = scale_img(img_dict['DAPI'])
    tub_img = scale_img(img_dict['Tub'])
    dapi_color_labels = color_label(n_mask, dapi_img)
    tub_color_labels = color_label(cyto_mask, dapi_img)
    return [dapi_img, tub_img, dapi_color_labels, tub_color_labels]


def segmentation_image(path, well_pos, img_dict, n_mask, cyto_mask):
    img_list = get_img_list(img_dict, n_mask, cyto_mask)
    title_list = ["DAPI image", "Tubulin image", "DAPI segmentation", "Tubulin image"]
    fig, ax = plt.subplots(ncols=4, figsize=(16, 7))
    for i in range(4):
        ax[i].axis('off')
        ax[i].imshow(img_list[i])
        ax[i].title.set_text(title_list[i])
    save_fig(path, f'{well_pos}_segmentation_check')


def segmentation_quality_control(exp_data, well, image):
    well_pos = get_well_pos(exp_data.plate_layout, well.getId())
    path = exp_data.quality_ctr_path
    index = well.countWellSample()
    random_number = random.randint(0, index)
    if index == random_number:
        segmentation_image(path, well_pos, image.corr_img_dict, image.n_mask, image.cyto_mask)


def well_loop(exp_data, well):
    df_well = pd.DataFrame()
    image_number = len(list(well.listChildren()))
    print("\nSegmenting and Analysing Images\n"
          "------------------------------------------------------------------------------------------------------\n")
    for number in tqdm.tqdm(range(image_number)):
        omero_img = well.getImage(number)
        image = Image(well, omero_img, exp_data, flatfield_corr)
        segmentation_quality_control(exp_data, well, image)
        data = ImageProperties(well, image, exp_data)
        df_well = pd.concat([df_well, data.image_properties])
    return df_well


excel_path = '/Users/hh65/Desktop/221102_cellcycle_exp6.xlsx'



@omero_connect
def omero_screen(conn=None, excel_path = excel_path):
    df_final = pd.DataFrame()
    df_quality_control = pd.DataFrame()
    exp_data = ExperimentData(excel_path, conn=conn)
    total = len(exp_data.plate_layout["Well_ID"] + 1)
    for count, ID in enumerate(exp_data.plate_layout["Well_ID"]):
        well = conn.getObject("Well", ID)
        flatfield_corr = FlatFieldCorr(well, exp_data)
        print(f"\nAnalysing well {count+1} of {total}\n"
              "------------------------------------------------------------------------------------------------------\n")
        well_data = well_loop(exp_data, well)
        df_final = pd.concat([df_final, well_data])
    df_final = df_final[
        list(df_final.columns.values[- 7:]) + list(df_final.columns.values[:len(df_final.columns.values) - 7])]
    df_final.to_csv(exp_data.final_data_path / f"{exp_data.plate_name}final_data.csv")

if __name__ == '__main__':
    omero_screen()
