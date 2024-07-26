import omero
from omero.gateway import BlitzGateway, _ImageWrapper, _DatasetWrapper
from omero_screen import Defaults
from ezomero import get_image
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


def save_fig(
    fig,
    fig_id: str,
    path: pathlib,
    tight_layout=True,
    fig_extension="png",
    resolution=300,
    transparent=False,
) -> None:
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
        fig.tight_layout()
    plt.savefig(dest, format=fig_extension, dpi=resolution, transparent=transparent)


def omero_connect(func):
    """
    decorator to log in and generate omero connection
    :param func: function to be decorated
    :return: wrapper function: passes conn to function and closes it after execution
    """

    @functools.wraps(func)
    def wrapper_omero_connect(*args, **kwargs):
        if Defaults['SERVER_DATA']:
            username = Defaults['USERNAME']
            password = Defaults['PASSWORD']
            server = Defaults['SERVER']
        else:
            username = input("Username: ")
            password = getpass.getpass(prompt="Password: ")
            server = "ome2.hpc.sussex.ac.uk"
        conn = BlitzGateway(username, password, host=server)
        value = None
        try:
            print("Connecting to Omero")
            if conn.connect():
                value = func(*args, **kwargs, conn=conn)
                print("Disconnecting from Omero")
            else:
                print(f"Failed to connect to Omero: {conn.getLastError()}")
        finally:
            # No side effects if called without a connection

            conn.close()

        return value

    return wrapper_omero_connect


def omero_connect_test(func):
    """
    decorator to log in and generate omero connection
    :param func: function to be decorated
    :return: wrapper function: passes conn to function and closes it after execution
    """
    server = Defaults.SERVER
    @functools.wraps(func)
    def wrapper_omero_connect(*args, **kwargs):
        try:
            with open("../data/secrets/config_test.json") as file:
                data = json.load(file)
            username = data["username"]
            password = data["password"]
        except IOError:
            username = input("Username: ")
            password = getpass.getpass(prompt="Password: ")
        conn = BlitzGateway(username, password, host="localhost")
        value = None
        try:
            print("Connecting to Omero")
            if conn.connect():
                value = func(*args, **kwargs, conn=conn)
                print("Disconnecting from Omero")
            else:
                print(f"Failed to connect to Omero: {conn.getLastError()}")
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


def scale_img(img: np.array, percentile: tuple = (1, 99)) -> np.array:
    """Increase contrast by scaling image to exclude lowest and highest intensities"""
    percentiles = np.percentile(img, (percentile[0], percentile[1]))
    return exposure.rescale_intensity(img, in_range=tuple(percentiles))


def generate_image(image, channel: int) -> np.ndarray:
    """
    Turn Omero Image Object from Well into numpy nd-array that is returned

    :param channel: channel number
    :return: numpy.ndarray
    """
    pixels = image.getPrimaryPixels()
    return pixels.getPlane(0, channel, 0)  # using channel number


def generate_random_image(well: str, channel: dict) -> np.ndarray:
    """
    Choose random image from well and feed to generate image function

    :param well: Omero well object
    :param channel: channel of image
    :return: an numpy ndarray from generate image function
    """
    index = well.countWellSample()
    random_img_num = random.randint(
        0, index - 1
    )  # to select random image for flatfield test
    random_image = well.getImage(random_img_num)
    return generate_image(random_image, channel[1])


def color_label(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Generate color labels for matplotlib to show segmentation
    :param mask: segmented mask
    :param img:
    :return: color labels for matplotlib
    """
    return color.label2rgb(mask, img, alpha=0.4, bg_label=0, kind="overlay")


def filter_segmentation(mask: np.ndarray) -> np.ndarray:
    """
    removes border objects and filters large abd small objects from segmentation mask
    :param mask: unfiltered segmentation mask
    :return: filtered segmentation mask
    """
    cleared = clear_border(mask, buffer_size=5)
    sizes = np.bincount(cleared.ravel())
    mask_sizes = sizes > 10
    mask_sizes[0] = 0
    cells_cleaned = mask_sizes[cleared]
    return cells_cleaned * mask


def add_map_annotation(omero_object, key_value, conn=None):
    """

    :param omero_object:
    :param data_dict:
    :param conn:
    :return:
    """
    map_ann = omero.gateway.MapAnnotationWrapper(conn)
    map_ann.setNs(omero.constants.metadata.NSCLIENTMAPANNOTATION)
    map_ann.setValue(key_value)
    map_ann.save()
    omero_object.linkAnnotation(map_ann)

def process_mip(conn: BlitzGateway, image: _ImageWrapper) -> np.ndarray:
    """
    Generate maximum intensity projection of a z-stack image.add()
    The get_image function returns an array of the shale ((t, z, x, y, c))
    This function only takes arrays with a single time dimension and this gets
    eliminated by the squeeze function.
    :param image: _imageWrapper object
    :return: numpy array of maximum intensity projection (x, y, c)
    """
    img, array = get_image(conn, image.getId())
    array_squeezed = np.squeeze(array, axis=0)
    return np.max(array_squeezed, axis=0)

def image_generator(image_array):
    for c in range(image_array.shape[-1]):
        yield image_array[..., c]


def load_mip(conn: BlitzGateway, image: _ImageWrapper, dataset: _DatasetWrapper ) -> None:
    """
    Load the maximum intensity projection of a z-stack image to OMERO
    :param image: _ImageWrapper object
    :param _ImageWrapper: maximum intensity projection
    :return: None
    """
    
    mip = process_mip(image)
    channel_num = mip.shape[-1]
    mip_name = f"mip_{image.getId()}"
    img_gen = image_generator(mip)
    image = conn.createImageFromNumpySeq(
        img_gen, mip_name, 1, channel_num, 1, dataset=dataset
    )
    add_map_annotation(image, mip_name, conn=conn)