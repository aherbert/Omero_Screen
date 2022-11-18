from omero.gateway import BlitzGateway
import numpy as np
from skimage import exposure, color
import time
import functools
import matplotlib.pyplot as plt
import json
import random


def save_fig(path, fig_id, tight_layout=True, fig_extension="pdf", resolution=300):
    dest = path / f"{fig_id}.{fig_extension}"
    # print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(dest, format=fig_extension, dpi=resolution)


def omero_connect(func):
    @functools.wraps(func)
    def wrapper_omero_connect(*args, **kwargs):
        with open('../secrets/config.json') as file:
            data = json.load(file)
        username = data['username']
        password = data['password']
        conn = BlitzGateway(username, password, host="ome2.hpc.sussex.ac.uk")
        conn.connect()
        print('connecting to Omero')
        value = func(*args, **kwargs, conn=conn)
        conn.close()
        print('disconnecting from Omero')
        return value

    return wrapper_omero_connect


def time_it(func):
    @functools.wraps(func)
    def wrapper_time_it(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = "%.4f" % (time.time() - start_time)
        print(f"{func.__name__!r} took {end_time} seconds to execute")
        return value, end_time

    return wrapper_time_it


def scale_img(img: np.array, percentile: tuple[float, float] = (1, 99)) -> np.array:
    """Increase contract by scaling image to exclude lowest and higest intensities"""
    percentiles = np.percentile(img, (percentile[0], percentile[1]))
    return exposure.rescale_intensity(img, in_range=tuple(percentiles))


def generate_image(image: 'Omero image object', channel: int) -> np.ndarray:
    """
    Turn Omero Image Object from Well into numpy nd-array that is returned

    :param well: Omero Image Object
    :param img_number: number to access the image in the wel object
    :param channel: channel number
    :return: numpy.ndarray
    """
    pixels = image.getPrimaryPixels()
    return pixels.getPlane(0, channel, 0)  # using channel number


def generate_random_image(well: 'Omero well Object', channel: dict) -> np.ndarray:
    """
    Choose random image from well and feed to generate image function

    :param well: Omero well object
    :param channel: channel of image
    :return: an numpy ndarray from generate image function
    """
    index = well.countWellSample()
    random_img_num = random.randint(0, index - 1)  # to select random image for flatfield test
    random_image = well.getImage(random_img_num)
    return generate_image(random_image, channel[1])


def color_label(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Generate color labels for matplotlib to show segmentation
    :param mask: segmented mask
    :param img:
    :return: color labels for matplotlib
    """
    return color.label2rgb(mask, img, alpha=0.4, bg_label=0, kind='overlay')


def get_well_pos(df, id):
    return df[df['Well_ID'] == id]['Well'].iloc[0]
