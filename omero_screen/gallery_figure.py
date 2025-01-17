import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image as PILImage
from io import BytesIO

def _create_heatmap_with_contours(image: np.ndarray, threshold_value: int = 10) -> np.ndarray:
    """
    Generates a heatmap with contours on a white background directly in memory.
    :param image: Input grayscale image (single channel).
    :param threshold_value: Threshold value for contour detection (default: 50).
    :return: np.ndarray: Processed image with heatmap and contours.
    """
    # Normalize the image to the range 0-255
    image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Ensure the image is single-channel
    if len(image_normalized.shape) > 2:
        if image_normalized.shape[2] == 3:  # If the image is RGB
            image_normalized = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2GRAY)
        else:
            return image

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

def save_gallery(filename: str, images: list, grid_size: int):
    """
    Generates a gallery figure of the images in a grid.
    :param filename: Output filename
    :param images: List of numpy 2D image arrays
    :param grid_size: Edge length of the grid
    """
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20), facecolor="white")
    axs = axs.reshape(grid_size, grid_size)  # Ensure axs is a 2D grid

    for idx, ax in enumerate(axs.flat):
        if idx < len(images):
            ax.imshow(_create_heatmap_with_contours(images[idx]))
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", facecolor="white")
    plt.close()
