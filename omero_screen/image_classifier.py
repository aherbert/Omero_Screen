from skimage.measure import label, regionprops
import numpy as np
import matplotlib.pyplot as plt
import torch
import omero

from PIL import Image as PILImage
from tqdm import tqdm
import os
import json
from torchvision import transforms
import cv2
from matplotlib.colors import Normalize
from io import BytesIO
import logging
import pathlib

from omero_screen.models import ROIBasedDenseNetModel

logger = logging.getLogger("omero-screen")


class ImageClassifier:
    """
    Class to process a collection of images with various operations, including normalization,
    cropping (based on centroids), mask application, and duplicate removal. Each step visualizes the result.
    """

    def __init__(self, conn, model_name):
        """
        Initialize the processor with images, metadata path, and optional masks.
        Args:
            images (list[dict]): List of image dictionaries, each containing channel data.
            metadata_path (str): Path to the metadata JSON file.
            masks (list[np.ndarray]): List of masks corresponding to the images (optional).
        """
        self.image_data = None
        self.crop_size = 100
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.cropped_images = []
        self.processed_images = []  # Store processed image data for all images
        self.crop_coords_list = []  # Store crop coordinates for all images
        self.selected_channels = {}
        self.predicted_classes = []
        self.gallery_dict = {}

        self.model, self.active_channels, self.class_options = self.load_model_from_omero("CNN_Models", model_name, model_name+".pth", conn)

    def load_model_from_omero(self, project_name, dataset_name, 
                               model_filename, conn=None):
        
        file_path = pathlib.Path.home() / model_filename

        # If the model file does not exist locally
        if not os.path.exists(file_path):
            # Find the project in OMERO
            project = conn.getObject("Project", attributes={"name": project_name})
            if project is None:
                logger.warning(f"Project '{project_name}' not found in OMERO.")
                return None, None
    
            # Find the dataset in OMERO
            dataset = next((ds for ds in project.listChildren() if ds.getName() == dataset_name), None)
            if dataset is None:
                logger.warning(f"Dataset '{dataset_name}' not found in project '{project_name}'.")
                return None, None
    
            # Check annotations in the dataset
            model_found = False
            for attachment in dataset.listAnnotations():
                if isinstance(attachment, omero.gateway.FileAnnotationWrapper):
                    # Download the model file
                    if attachment.getFileName() == model_filename:
                        with open(file_path, "wb") as f:
                            for chunk in attachment.getFileInChunks():
                                f.write(chunk)
                        logger.info(f"Downloaded model file to {file_path}")
                        model_found = True
                        break
            if not model_found:
                logger.warning(f"File '{model_filename}' not found in dataset '{dataset_name}' under project '{project_name}'.")
                return None, None

        # If the model file is downloaded, download the Key-Value Pairs
        # Extract image channels
        active_channels, class_options = self.download_metadata_and_extract_channels(dataset_name, conn)

        if active_channels:
            print(f"Active Channels: {active_channels}")
        else:
            print("No active channels found.")

        self.gallery_dict = {class_name: [] for class_name in class_options}

        # Load the model
        model = ROIBasedDenseNetModel(num_classes=len(class_options), num_channels=len(active_channels))
        model.load_state_dict(torch.load(file_path, weights_only=True, map_location=torch.device('cpu')))
        model = model.to(self.device)  # Move model to the device
        model.eval()
        return model, active_channels, class_options
    
    def download_metadata_and_extract_channels(self, dataset_name, conn):
        """
        Download the metadata.json file associated with the model and extract active channels.
        
        Args:
            dataset_name (str): The name of the dataset in OMERO.
            model_name (str): The name of the model (used to locate metadata.json).
            conn: OMERO connection object.
        
        Returns:
            list: A list of active channels if found, otherwise an empty list.
        """
        metadata_file_name = "metadata.json"
        metadata_local_path = pathlib.Path.home() / metadata_file_name

        # Find the dataset in OMERO
        dataset = conn.getObject("Dataset", attributes={"name": dataset_name})
        if dataset is None:
            logger.warning(f"Dataset '{dataset_name}' not found in OMERO.")
            return []

        # Check for metadata.json in the dataset
        for annotation in dataset.listAnnotations():
            if isinstance(annotation, omero.gateway.FileAnnotationWrapper):
                if annotation.getFileName() == metadata_file_name:
                    # Download the metadata.json file
                    with open(metadata_local_path, "wb") as f:
                        for chunk in annotation.getFileInChunks():
                            f.write(chunk)
                    logger.info(f"Downloaded metadata file to {metadata_local_path}")
                    break
        else:
            logger.warning(f"Metadata file '{metadata_file_name}' not found in dataset '{dataset_name}'.")
            return []

        # Read the metadata.json file and extract active channels
        try:
            with open(metadata_local_path, "r") as f:
                metadata = json.load(f)
                if "user_data" in metadata and "channels" in metadata["user_data"]:
                    active_channels = metadata["user_data"]["channels"]
                    class_options = metadata["class_options"]
                    class_options.remove("unassigned")
                    logger.info(f"Active channels extracted: {active_channels}")
                    logger.info(f"Class options: {class_options}")
                    logger.info(f"Crop Size: {self.crop_size}")
                    return active_channels, class_options
                else:
                    logger.warning(f"Metadata file '{metadata_file_name}' does not contain 'channels' information.")
                    return []
        except Exception as e:
            logger.error(f"Error reading metadata file '{metadata_file_name}': {e}")
            return []
    
    def select_channels(self, image_data):
        self.image_data = image_data
        self.selected_channels = {channel: image_data[channel] for channel in self.active_channels}
        logger.info(f"Selected channels for classification: {self.selected_channels.keys()}")

    # Duplicate removal function
    def remove_duplicate_cyto_ids(self, images, cyto_id_column="Cyto_ID"):
        """
        Remove duplicate rows based on the specified Cyto_ID column.

        Args:
            images (pd.DataFrame): DataFrame containing the data.
            cyto_id_column (str): The name of the Cyto_ID column.

        Returns:
            pd.DataFrame: DataFrame with duplicates removed.
        """
        # Drop duplicates based on the Cyto_ID column
        unique_images = images.drop_duplicates(subset=cyto_id_column, keep='first')
        return unique_images
    
    def process_images(self, images, mask):

        original_images = images.copy()

        predicted_classes = []

        print("Images length before removing duplicates : ", len(images))
        images = self.remove_duplicate_cyto_ids(images)
        print("Images length after removing duplicates : ", len(images))

        for i in tqdm(range(len(images["centroid-0"]))):

            # Center the crop around the centroid coordinates with a 100x100 area
            # Assume YX are the last dimensions
            max_length_x = self.selected_channels[self.active_channels[0]].shape[-1]
            max_length_y = self.selected_channels[self.active_channels[0]].shape[-2]

            centroid_x = images["centroid-1_x"].iloc[i]
            centroid_y = images["centroid-0_y"].iloc[i]

            x0 = int(max(0, centroid_x - self.crop_size // 2))
            x1 = int(min(max_length_x, centroid_x + self.crop_size // 2))
            y0 = int(max(0, centroid_y - self.crop_size // 2))
            y1 = int(min(max_length_y, centroid_y + self.crop_size // 2))

            combined_channels = []

            for channel in self.selected_channels:

                # Crop image and mask
                cropped_image = self.crop(self.selected_channels[channel], x0, y0, x1, y1)

                cropped_mask = self.crop(mask, x0, y0, x1, y1)

                corrected_mask = self.erase_masks(cropped_mask)

                masked_image = self.extract_roi_multichannel(cropped_image, corrected_mask)
                
                target_size = (100, 100)  # Target size (height, width)

                padded_masked_image = self.add_padding(masked_image, target_size)

                # Add the masked image to the list for combining channels
                combined_channels.append(padded_masked_image)
            
            if len(combined_channels) > 1:
                # Combine channels if there is more than one channel
                latest_image = np.stack(combined_channels, axis=0)
                latest_image = np.squeeze(latest_image)
                latest_image = np.transpose(latest_image, (1, 2, 0))
            else:
                latest_image = padded_masked_image

            max_val = np.max(latest_image)

            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Lambda(lambda x: x / max_val),
                # transforms.Normalize([0.5], [0.5])
            ])

            image_tensor = data_transform(latest_image) 
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            predicted_class = self.classify(image_tensor)
            processed_image = self.create_heatmap_with_contours(latest_image)
            self.gallery_dict[predicted_class].append(processed_image)
            predicted_classes.append(predicted_class)

        # This gives an warning:
        # A value is trying to be set on a copy of a slice from a DataFrame.
        images["Class"] = predicted_classes

        original_images = original_images.merge(
            images[["Cyto_ID", "Class"]], 
            on="Cyto_ID", 
            how="left"
        )

        return original_images


    def classify(self, image_tensor):

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)

        class_names = self.class_options
        predicted_class = class_names[int(predicted.item())]
        self.predicted_classes.append(predicted_class)

        return predicted_class

    def transform_image(self, image):

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5], [0.5])
        ])

        image_tensor = data_transform(image)  # Apply transform
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)

        return image_tensor

    def crop(self, image, x0, y0, x1, y1):
        """
        Crops the input image using the provided coordinates.
        
        Args:
            image (numpy array): The image to be cropped.
            x0, y0 (int): Top-left coordinates for cropping.
            x1, y1 (int): Bottom-right coordinates for cropping.
        
        Returns:
            Cropped image as a numpy array.
        """
        # Ensure coordinates are valid
        if x1 < x0:
            x0, x1 = x1, x0  # Swap values if x1 is less than x0
        if y1 < y0:
            y0, y1 = y1, y0  # Swap values if y1 is less than y0

        # Crop with numpy to return a 2D image with YX as last dimensions
        return image.squeeze()[y0:y1, x0:x1]

        # # Convert image to PIL format if it's a numpy array
        # if isinstance(image, np.ndarray):
        #     image = PILImage.fromarray(image)
        
        # # Crop the image using the corrected coordinates
        # cropped_image = image.crop((x0, y0, x1, y1))
        
        # # Convert back to numpy array for further processing
        # return np.array(cropped_image)
    
    def extract_roi_multichannel(self, image, mask):
        """
        Extracts the ROI (Region of Interest) from a multi-channel image using the mask.
        Args:
            image (numpy array): Multi-channel input image (height, width, channels).
            mask (numpy array): Mask image (should have the same height and width).
        Returns:
            numpy array: Cropped ROI with masked regions.
        """

        if image.ndim == 2:  # Single channel (height, width)
            image = np.expand_dims(image, axis=-1) 

        # Convert mask to binary
        binary_mask = (mask > 0).astype(np.uint8)

        # Apply the mask to all channels
        roi = np.zeros_like(image)
        for channel in range(image.shape[-1]):
            roi[..., channel] = image[..., channel] * binary_mask

        # Crop the masked region
        coords = np.argwhere(binary_mask > 0)
        if len(coords) > 0:  # Ensure there are non-zero mask regions
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            roi_cropped = roi[y0:y1, x0:x1, :]
        else:
            roi_cropped = roi  # Return full image if no ROI is found
        return roi_cropped

    def add_padding(self, image, target_size, padding_value=0):
        """
        Adds padding to a NumPy array image to reach the target size.

        Args:
            image (np.ndarray): Input image as a NumPy array (H, W) or (H, W, C).
            target_size (tuple): Target size as (height, width).
            padding_value (int): Value to use for padding (default: 0).

        Returns:
            np.ndarray: Padded image with the desired target size.
        """
        current_height, current_width = image.shape[:2]
        target_height, target_width = target_size

        # Calculate the padding needed
        pad_height = max(0, target_height - current_height)
        pad_width = max(0, target_width - current_width)

        # Divide padding equally on all sides
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Determine padding configuration based on the number of dimensions
        if image.ndim == 2:  # Grayscale image
            padding_config = ((pad_top, pad_bottom), (pad_left, pad_right))
        elif image.ndim == 3:  # RGB or multi-channel image
            padding_config = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else:
            raise ValueError("Unsupported image dimensions. Image must be 2D or 3D.")

        # Apply padding
        padded_image = np.pad(image, padding_config, mode='constant', constant_values=padding_value)
        return padded_image
    
    def mask_image(self, image, mask):
        binary_mask = mask / mask.max()  # Normalize between 0 and 1
        masked_image = image * binary_mask
        return masked_image
    
    def erase_masks(self, cropped_label: np.ndarray) -> np.ndarray:
        """
        Erases all masks in the cropped_label that do not overlap with the centroid.
        """

        center_row, center_col = np.array(cropped_label.shape) // 2

        unique_labels = np.unique(cropped_label)
        for unique_label in unique_labels:
            if unique_label == 0:  # Skip background
                continue

            binary_mask = cropped_label == unique_label

            if np.sum(binary_mask) == 0:  # Check for empty masks
                continue

            label_props = regionprops(label(binary_mask))

            if len(label_props) == 1:
                cropped_centroid_row, cropped_centroid_col = label_props[
                    0
                ].centroid

                # Using a small tolerance value for comparing centroids
                tol = 15

                if (
                    abs(cropped_centroid_row - center_row) > tol
                    or abs(cropped_centroid_col - center_col) > tol
                ):
                    cropped_label[binary_mask] = 0

        return cropped_label
    
    def create_heatmap_with_contours(self, image: np.ndarray, threshold_value: int = 10) -> np.ndarray:
        """
        Generates a heatmap with contours on a white background directly in memory.

        Args:
            image (np.ndarray): Input grayscale image (single channel).
            threshold_value (int): Threshold value for contour detection (default: 50).

        Returns:
            np.ndarray: Processed image with heatmap and contours.
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
