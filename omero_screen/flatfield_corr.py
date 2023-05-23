#!/usr/bin/env python
from omero_screen.aggregator import ImageAggregator
from omero_screen.general_functions import save_fig, scale_img, generate_image, \
    omero_connect
from omero_screen.data_structure import MetaData, ExpPaths
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib
from skimage import io
import platform
import random
import glob

if platform.system() == 'Darwin':
     matplotlib.use('MacOSX')  # avoid matplotlib warning about interactive backend
def flatfieldcorr(meta_data, exp_paths, conn) -> dict:
    """

    :return:
    """

    plate = meta_data.plate_obj
    channels = meta_data.channels
    template_path = exp_paths.flatfield_templates
    if len(glob.glob(f"{str(template_path)}/*.tif")) == len(channels):
        return load_corr_dict(template_path, channels)
    else:
        return generate_corr_dict(plate, channels, template_path, conn)

def load_corr_dict(path, channels):
    """
    Loads flatfield correction masks from file
    :param path: path to flatfield correction masks
    :param channels: dictionary of channels
    :return: dictionary of channel_name : flatfield correction masks
    """
    print("\nLoading Flatfield Correction Masks from File\n")
    corr_img_list = glob.glob(f'{str(path)}/*.tif')
    array_list = list(map(io.imread, corr_img_list))
    channel_list = list(channels.keys())
    return dict(zip(channel_list, array_list))
def generate_corr_dict(plate, channels, template_path, conn):
    """
    Saves each flat field mask file with well position and channel name
    :return: a dictionary with channel_name : flatfield correction masks
    """

    print(f"\nAssembling Flatfield Correction Masks for {len(channels)} Channels\n")
    corr_dict = {}
    img_list = random_imgs(plate)
    for channel in list(channels.items()):
        corr_img_id = f"corr_{channel[0]}"
        norm_mask = aggregate_imgs(img_list, channel, conn)
        io.imsave(template_path / f"{corr_img_id}_flatfield_masks.tif", norm_mask)
        example = gen_example(img_list, channel, norm_mask, conn)
        example_fig(example, channel, template_path)
        corr_dict[channel[0]] = norm_mask  # associates channel name with flatfield mask
    return corr_dict

def random_imgs(plate):
     """
    Generate a random image from each well in the plate
    :param plate: omero plate object
    :return: random image from each well
    """
     # Get all the wells associated with the plate
     wells = plate.listChildren()
     img_list = []
     for well in wells:
         index = well.countWellSample()
         for index in range(0, index):
             img_list.append(well.getImage(index).getId())

     return img_list if len(img_list) <= 100 else random.sample(img_list, 100)


def aggregate_imgs(img_list, channel, conn):
     """
    Aggregate images in well for specified channel and generate correction mask using the Aggregator Module
    :param channel: dictionary from self.exp_data.channels
    :return: flatfield correction mask for given channel
    """
     agg = ImageAggregator(60)
     for img_id in tqdm(img_list):
        image = conn.getObject("Image", img_id)
        image_array = generate_image(image, channel[1])
        agg.add_image(image_array)
     blurred_agg_img = agg.get_gaussian_image(30)
     return blurred_agg_img / blurred_agg_img.mean()


def gen_example(img_list, channel, mask, conn):
    random_id = random.choice(img_list)
    image = conn.getObject("Image", random_id)
    example_img = generate_image(image, channel[1])
    scaled = scale_img(example_img)
    corr_img = example_img / mask
    bgcorr_img = corr_img - np.percentile(corr_img, 0.2) + 1
    corr_scaled = scale_img(bgcorr_img)
    # order all images for plotting
    return [(scaled, 'original image'), (np.diagonal(example_img), 'diag. intensities'),
            (corr_scaled, 'corrected image'), (np.diagonal(corr_img), 'diag. intensities'),
            (mask, 'flatfield correction mask')]


def example_fig(data_list, channel, path):
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    for i, data_tuple in enumerate(data_list):
        plt.sca(ax[i])
        if i in [0, 2, 4]:
            plt.imshow(data_tuple[0], cmap='gray')
        else:
            plt.plot(data_tuple[0])
            plt.ylim(0, 10 * data_tuple[0].min())
        plt.title(data_tuple[1])
    # save and close figure
    fig_id = f"{channel[0]}_flatfield_check"  # using channel name
    save_fig(path, fig_id)
    plt.close(fig)


# test


if __name__ == "__main__":
    @omero_connect
    def flatfield_test(conn=None):
        meta_data = MetaData(948, conn)
        exp_paths = ExpPaths(meta_data)
        well = conn.getObject("Well", 10636)
        return flatfieldcorr(meta_data, exp_paths)


    flatfield_corr = flatfield_test()
    print(flatfield_corr['DAPI'].shape)
