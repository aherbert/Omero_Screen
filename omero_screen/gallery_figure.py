import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image as PILImage
from io import BytesIO

def _create_heatmap_with_contours(image: np.ndarray, threshold_value: int = 10) -> np.ndarray:
    """
    Generates a heatmap with contours on a white background directly in memory.
    Note: Multi-channel images use the first 3 channels as RBG and converted to greyscale.
    :param image: Input grayscale image (single channel), or CYX multi-channel.
    :param threshold_value: Threshold value for contour detection (default: 50).
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
            ax.imshow(_create_heatmap_with_contours(images[idx]))
        ax.axis('off')

    return fig
