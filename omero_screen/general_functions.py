import pandas as pd
from omero.gateway import BlitzGateway
import pathlib
import numpy as np
from skimage import exposure, color
from skimage.segmentation import clear_border
import time
import functools
import matplotlib.pyplot as plt
import json
import random
import getpass


def save_fig(path: pathlib, fig_id: str, tight_layout=True, fig_extension="pdf", resolution=300) -> None:
    """
    coherent saving of matplotlib figures as pdfs (default)
    :param path: path for saving
    :param fig_id: name of saved figure
    :param tight_layout: option, default True
    :param fig_extension: option, default pdf
    :param resolution: option, default 300dpi
    :return: None, saves Figure in poth
    """
    dest = path / f"{fig_id}.{fig_extension}"
    # print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(dest, format=fig_extension, dpi=resolution)


def omero_connect(func):
    """
    decorator to log in and generate omero connection
    :param func: function to be decorated
    :return: wrapper function: passes conn to function and closes it after execution
    """

    @functools.wraps(func)
    def wrapper_omero_connect(*args, **kwargs):
        try:
            with open('../data/secrets/config.json') as file:
                data = json.load(file)
            username = data['username']
            password = data['password']
        except IOError:
            username = input("Username: ")
            password = getpass.getpass(prompt='Password: ')
        conn = BlitzGateway(username, password, host="ome2.hpc.sussex.ac.uk")
        value = None
        try:
            print('Connecting to Omero')
            if conn.connect():
                value = func(*args, **kwargs, conn=conn)
                print('Disconnecting from Omero')
            else:
                print('Failed to connect to Omero: %s' % conn.getLastError())
        finally:
            # No side effects if called without a connection
            conn.close()
        return value

    return wrapper_omero_connect


def time_it(func):
    """
    decorator to time functions
    :param func: function
    :return: wrapper that prints  times
    """

    @functools.wraps(func)
    def wrapper_time_it(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = "%.4f" % (time.time() - start_time)
        print(f"{func.__name__!r} took {end_time} seconds to execute")
        return value, end_time

    return wrapper_time_it


def scale_img(img: np.array, percentile: tuple[float, float] = (1, 99)) -> np.array:
    """Increase contrast by scaling image to exclude lowest and highest intensities"""
    percentiles = np.percentile(img, (percentile[0], percentile[1]))
    return exposure.rescale_intensity(img, in_range=tuple(percentiles))


def generate_image(image: 'Omero image object', channel: int) -> np.ndarray:
    """
    Turn Omero Image Object from Well into numpy nd-array that is returned

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


def filter_segmentation(mask: np.ndarray) -> np.ndarray:
    """
    removes border objects and filters large abd small objects from segmentation mask
    :param mask: unfiltered segmentation mask
    :return: filtered segmentation mask
    """
    cleared = clear_border(mask)
    sizes = np.bincount(cleared.ravel())
    mask_sizes = (sizes > 10)
    mask_sizes[0] = 0
    cells_cleaned = mask_sizes[cleared]
    return cells_cleaned * mask
