import numpy as np
import scipy
from scipy import ndimage
import skimage


def strel_disk(radius):
    """Create a disk structuring element for morphological operations
  
  radius - radius of the disk
  """
    iradius = int(radius)
    x, y = np.mgrid[-iradius: iradius + 1, -iradius: iradius + 1]
    radius2 = radius * radius
    strel = np.zeros(x.shape)
    strel[x * x + y * y <= radius2] = 1
    return strel


def median_filter(pixel_data, radius):
    """Perform median filter with the given radius"""
    filter_sigma = max(1, int(radius + 0.5))
    strel = strel_disk(filter_sigma)
    scale = 65535 / np.max(pixel_data);
    rescaled_pixel_data = pixel_data * scale
    rescaled_pixel_data = rescaled_pixel_data.astype(np.uint16)
    output_pixels = skimage.filters.median(rescaled_pixel_data, strel, behavior="rank")
    output_pixels = output_pixels / scale
    output_pixels = output_pixels.astype(pixel_data.dtype)
    return output_pixels


def gaussian_filter(pixel_data, sigma):
    """Perform gaussian filter with the given radius"""

    # Use the method to divide by the bleed over fraction
    # to remove edge artifacts
    def fn(image):
        return scipy.ndimage.gaussian_filter(
            image, sigma, mode="constant", cval=0
        )

    mask = np.ones(pixel_data.shape)
    bleed_over = fn(mask)
    smoothed_image = fn(pixel_data)
    output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
    return output_image


# https://github.com/CellProfiler/centrosome/blob/master/centrosome/cpmorphology.py
# centrosome.cpmorphology.block
def block(shape, block_shape):
    """Create a labels image that divides the image into blocks
    
    shape - the shape of the image to be blocked
    block_shape - the shape of one block
    
    returns a labels matrix and the indexes of all labels generated
    
    The idea here is to block-process an image by using SciPy label
    routines. This routine divides the image into blocks of a configurable
    dimension. The caller then calls scipy.ndimage functions to process
    each block as a labeled image. The block values can then be applied
    to the image via indexing. For instance:
    
    labels, indexes = block(image.shape, (60,60))
    minima = scind.minimum(image, labels, indexes)
    img2 = image - minima[labels]
    """
    shape = np.array(shape)
    block_shape = np.array(block_shape)
    i, j = np.mgrid[0: shape[0], 0: shape[1]]
    ijmax = (shape.astype(float) / block_shape.astype(float)).astype(int)
    ijmax = np.maximum(ijmax, 1)
    multiplier = ijmax.astype(float) / shape.astype(float)
    i = (i * multiplier[0]).astype(int)
    j = (j * multiplier[1]).astype(int)
    labels = i * ijmax[1] + j
    indexes = np.array(list(range(np.product(ijmax))))
    return labels, indexes


# https://github.com/CellProfiler/centrosome/blob/master/centrosome/cpmorphology.py
# centrosome.cpmorphology.fixup_scipy_ndimage_result
def fixup_scipy_ndimage_result(whatever_it_returned):
    """Convert a result from scipy.ndimage to a numpy array
    
    scipy.ndimage has the annoying habit of returning a single, bare
    value instead of an array if the indexes passed in are of length 1.
    For instance:
    scind.maximum(image, labels, [1]) returns a float
    but
    scind.maximum(image, labels, [1,2]) returns a list
    """
    if getattr(whatever_it_returned, "__getitem__", False):
        return np.array(whatever_it_returned)
    else:
        return np.array([whatever_it_returned])


class ImageAggregator():
    """ImageAggregator accumulates the image data from successive images and
    calculates the aggregate image when asked.
    """

    def __init__(self, block_size=0):
        """Create an instance
        block_size - the size of the block region for the minimum; ignored if
        not strictly positive.
        """
        super(ImageAggregator, self).__init__()
        self.__block_size = block_size
        self.__labels = None
        self.__indexes = None
        self.__dirty = False
        self.__image_sum = None
        self.__count = 0
        self.__cached_image = None

    def add_image(self, image):
        """Accumulate the data from the given image
        image - an instance of a 2D numpy array
        """
        self.__dirty = True
        # Optional aggregation of the minimum from blocks
        # See: https://github.com/CellProfiler/CellProfiler/blob/master/cellprofiler/modules/correctilluminationcalculate.py#L923       
        if self.__count == 0:
            self.__image_sum = np.zeros(image.shape)
            if self.__block_size > 0:
                self.__labels, self.__indexes = block(
                    image.shape[:2], (self.__block_size, self.__block_size)
                )
        if self.__block_size > 0:
            minima = fixup_scipy_ndimage_result(
                scipy.ndimage.minimum(image, self.__labels, self.__indexes)
            )
            pixel_data = minima[self.__labels]
        else:
            pixel_data = image
        self.__image_sum = self.__image_sum + pixel_data
        self.__count = self.__count + 1

    def get_image(self):
        """Get the aggregated image"""
        if self.__dirty:
            self.__cached_image = self.__image_sum / self.__count
            self.__dirty = False
        return self.__cached_image

    def get_median_image(self, radius):
        """Get the aggregated image after smoothing with a median filter"""
        im = self.get_image()
        return median_filter(im, radius)

    def get_gaussian_image(self, sigma):
        """Get the aggregated image after smoothing with a Gaussian filter"""
        im = self.get_image()
        return gaussian_filter(im, sigma)

    def reset(self):
        """Reset the aggregator"""
        self.__labels = None
        self.__indexes = None
        self.__dirty = False
        self.__image_sum = None
        self.__count = 0
        self.__cached_image = None
