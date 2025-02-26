import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from PIL import Image as PILImage
from io import BytesIO

def _create_heatmap_with_contours(image: np.ndarray, threshold_value: int = 10) -> np.ndarray:
    """
    Generates a heatmap with contours on a white background directly in memory.
    Note: Multi-channel images use the first 3 channels as RBG and converted to greyscale.
    :param image: Input grayscale image (single channel), or CYX multi-channel.
    :param threshold_value: Threshold value for contour detection.
    :return: np.ndarray: Processed image with heatmap and contours.
    """
    # Ensure the image is CYX for processing
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=0)
    elif len(image.shape) == 3:
        pass
    else:
        raise Exception(f'Unsupported image shape: {image.shape}')

    # Normalize the image to the range 0-255
    image_normalized = np.array([cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
      for x in image])

    # Ensure the image is single-channel
    s = image_normalized.shape
    if s[0] == 1:
        image_normalized = image_normalized[0]
    else:
        # multi-channel
        # Pad with a blank plane or crop to 3 channels and convert RGB to greyscale
        if s[0] == 2:
            image_normalized = np.concatenate([image_normalized, np.zeros((1, s[1], s[2]), dtype=np.uint8)])
        else:
            image_normalized = image_normalized[0:3]
        image_normalized = cv2.cvtColor(image_normalized.transpose((1,2,0)), cv2.COLOR_BGR2GRAY)

    # Thresholding to reduce noise
    _, thresholded_image = cv2.threshold(image_normalized, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a white background for the heatmap
    plt.figure(figsize=(5, 5))
    plt.imshow(np.ones_like(image_normalized) * 255, cmap='gray', vmin=0, vmax=255)  # White background
    plt.imshow(image_normalized, cmap='Reds', norm=Normalize(vmin=0, vmax=255), alpha=0.9)  # Heatmap

    # Draw contours
    for contour in contours:
        plt.plot(contour[:, :, 0], contour[:, :, 1], color='black', linewidth=1)  # Black contour lines

    plt.axis('off')  # Remove axes

    # Render the figure to a buffer in memory
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Convert the buffer to a NumPy array
    buf.seek(0)
    image_pil = PILImage.open(buf)
    processed_image = np.array(image_pil)
    buf.close()

    return processed_image

def _create_image(image: np.ndarray) -> np.ndarray:
    """
    Generates an image (M, N) or (M, N, 3).
    Note: Multi-channel images use the first 3 channels as RBG.
    :param image: Input grayscale image (single channel), or CYX multi-channel.
    :return: np.ndarray: Processed image.
    """
    # Ensure the image is CYX for processing
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=0)
    elif len(image.shape) == 3:
        pass
    else:
        raise Exception(f'Unsupported image shape: {image.shape}')

    # Normalize the image to the range 0-255
    image_normalized = np.array([cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
      for x in image])

    s = image_normalized.shape
    if s[0] == 1:
        # single-channel
        return image_normalized[0]

    # multi-channel
    # Pad with a blank plane or crop to 3 channels
    if s[0] == 2:
        image_normalized = np.concatenate([image_normalized, np.zeros((1, s[1], s[2]), dtype=np.uint8)])
    else:
        image_normalized = image_normalized[0:3]
    # (C,Y,X) -> (M,N,3)
    return image_normalized.transpose((1,2,0))

def create_gallery(images: list, grid_size: int):
    """
    Generates a gallery figure of the images in a grid.
    :param images: List of numpy 2D image arrays
    :param grid_size: Edge length of the grid
    :return: matplotlib.figure.Figure
    """
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20), facecolor="white")
    axs = axs.reshape(grid_size, grid_size)  # Ensure axs is a 2D grid

    for idx, ax in enumerate(axs.flat):
        if idx < len(images):
            im = _create_image(images[idx])
            ax.imshow(im)
            # Create contours using first channel and assuming non-masked pixels are > 0
            if len(im.shape) == 3:
                im = im[:, :, 0]
            _, thresholded_image = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # contour has shape n x 1 x 2
                p = Polygon(contour.squeeze(axis=1), fc='none', ec='cyan', lw=1)
                ax.add_patch(p)
        ax.axis('off')

    return fig
