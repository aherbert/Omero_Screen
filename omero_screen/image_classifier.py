from skimage.measure import regionprops
import numpy as np
import torch
import omero

from tqdm import tqdm
import json
import logging
import pathlib
import skimage
from random import randrange

logger = logging.getLogger("omero-screen")


class ImageClassifier:
    """
    Classify images using a model.
    """
    def __init__(self, conn, model_name, class_name='Class'):
        """
        Initialize the classifier.
        :param conn: OMERO connection.
        :param model_name: Name of model.
        """
        self.image_data = None
        self.crop_size = 0
        self.input_shape = None
        self.gallery_size = 0
        self.batch_size = 16
        self.class_name = class_name
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.cropped_images = []
        self.processed_images = []  # Store processed image data for all images
        self.crop_coords_list = []  # Store crop coordinates for all images
        self.selected_channels = []
        self.gallery_dict = {}

        self.model, self.active_channels, self.class_options = self._load_model_from_omero(
          "CNN_Models", model_name, conn)

    def _load_model_from_omero(self, project_name, dataset_name, conn=None):
        model_filename = self._download_file(project_name, dataset_name, dataset_name + ".pt", conn)
        if not model_filename:
            return None, None, None
        meta_filename = self._download_file(project_name, dataset_name, dataset_name + ".json", conn)
        if not meta_filename:
            return None, None, None

        # Extract image channels
        active_channels, class_options = self._extract_channels(meta_filename)

        if active_channels:
            print(f"Active Channels: {active_channels}")
        else:
            print("No active channels found.")
            return None, None, None

        # list of random samples, total number of items
        self.gallery_dict = {class_name: [[], 0] for class_name in class_options}

        # Load the model
        model = torch.jit.load(model_filename, map_location=torch.device('cpu'))
        model = model.to(self.device)
        model.eval()
        return model, active_channels, class_options

    def _download_file(self, project_name, dataset_name, file_name, conn):
        """
        Download the file attachment.
        :param project_name (str): The name of the project in OMERO.
        :param dataset_name (str): The name of the dataset in OMERO.
        :param file_name (str): The name of the file attachment in OMERO.
        :param conn: OMERO connection object.
        :return: str: Path to local file (or None).
        """
        local_path = pathlib.Path.home() / '.cache' / 'omero_screen' / file_name

        # If the model file does not exist locally
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
          
            # Find the project in OMERO
            project = conn.getObject("Project", attributes={"name": project_name})
            if project is None:
                logger.warning(f"Project '{project_name}' not found in OMERO.")
                return None

            # Find the dataset in OMERO
            dataset = next((ds for ds in project.listChildren() if ds.getName() == dataset_name), None)
            if dataset is None:
                logger.warning(f"Dataset '{dataset_name}' not found in project '{project_name}'.")
                return None

            # Check annotations in the dataset
            for attachment in dataset.listAnnotations():
                if isinstance(attachment, omero.gateway.FileAnnotationWrapper):
                    # Download the model file
                    if attachment.getFileName() == file_name:
                        with open(local_path, "wb") as f:
                            for chunk in attachment.getFileInChunks():
                                f.write(chunk)
                        logger.info(f"Downloaded model file to {local_path}")
                        return local_path

            logger.warning(f"File '{file_name}' not found in dataset '{dataset_name}' under project '{project_name}'.")
            return None
        # Already cached
        return local_path

    def _extract_channels(self, meta_filename):
        # Read the metadata.json file and extract active channels
        try:
            with open(meta_filename, "r") as f:
                metadata = json.load(f)
                active_channels = metadata.get('channels', [])
                class_options = metadata.get('labels', [])
                img_shape = metadata.get('img_shape', None)
                input_shape = metadata.get('input_shape', None)
                if active_channels and class_options and img_shape and input_shape:
                    # assume a square image for cropping
                    self.crop_size = img_shape[-1]
                    self.input_shape = tuple(input_shape[-2:])
                    logger.info(f"Active channels extracted: {active_channels}")
                    logger.info(f"Class options: {class_options}")
                    logger.info(f"Crop Size: {self.crop_size}")
                    logger.info(f"Input shape: {self.input_shape}")
                    return active_channels, class_options
                else:
                    logger.warning(f"Metadata file '{meta_filename}' does not contain required information.")
        except Exception as e:
            logger.error(f"Error reading metadata file '{meta_filename}': {e}")
        return [], []

    def select_channels(self, image_data):
        """
        Select the channels from the image_data to be used for classification.
        If this method returns False then the classifier is not able to process the images.
        :param image_data (Dict): Dictionary of images keyed by channel name.
        :return: bool: True if the channels were selected.
        """
        if self.active_channels:
            self.image_data = image_data
            self.selected_channels = [image_data[channel] for channel in self.active_channels]
            logger.info(f"Selected channels for classification: {self.active_channels}")
            return True
        return False

    def process_images(self, original_image_df, mask):
        if len(self.selected_channels) == 0:
            return

        predicted_classes = []

        # Entries may share the same cytoplasm ID, e.g. cells with multiple nuclei
        image_df = original_image_df.drop_duplicates(subset="Cyto_ID", keep='first')
        logger.info("Classification of %d cyto IDs (%d items)", len(image_df), len(original_image_df))

        img_size = (self.crop_size, self.crop_size)  # Target size (height, width)
        half_crop = self.crop_size // 2

        # Assume YX are the last dimensions
        max_length_x = self.selected_channels[0].shape[-1]
        max_length_y = self.selected_channels[0].shape[-2]

        # Batch processing
        total = len(image_df["centroid-0"])
        step = self.batch_size
        pbar = tqdm(total=total)
        for start in range(0, total, step):
            stop = min(start+step, total)
            pbar.n = stop
            pbar.refresh()

            batch = []
            for i in range(start, stop):
                # Center the crop around the centroid coordinates
                centroid_x = image_df["centroid-1_x"].iloc[i]
                centroid_y = image_df["centroid-0_y"].iloc[i]

                x0 = int(max(0, centroid_x - half_crop))
                x1 = int(min(max_length_x, centroid_x + half_crop))
                y0 = int(max(0, centroid_y - half_crop))
                y1 = int(min(max_length_y, centroid_y + half_crop))

                # Crop mask
                cropped_mask = self._crop(mask, x0, y0, x1, y1).copy()
                # Pass in the translated centroid allowing for the crop to clip
                cx = min(half_crop, int(centroid_x))
                cy = min(half_crop, int(centroid_y))
                corrected_mask = self._erase_masks(cropped_mask, cx, cy)
                # Convert mask to binary
                binary_mask = (corrected_mask > 0).astype(np.uint8)

                # Create cropped YXC image
                img = np.dstack([self._crop(i, x0, y0, x1, y1) for i in self.selected_channels])
                # Image normalisation copied from training repo cellclass:src/bin/create_dataset.py
                # Extract (1, 99) percentile and convert to 8-bit
                img = self._to_uint8(img)
                # Remove pixels outside the mask (transforms image to CYX)
                img = self._extract_roi(img, binary_mask)
                # Pad to required image size
                img = self._add_padding(img, img_size)
                batch.append(img)

            # Create tensor (B, C, H, W)
            batch = np.array(batch)

            # Value scaling: [0,255] -> [0,1]
            tensor = np.divide(batch, 255, dtype=np.float32)
            # Resize to model input size
            output_shape = batch.shape[0:2] + self.input_shape
            tensor = skimage.transform.resize(tensor, output_shape, mode='edge')
            # Convert to tensor in place
            tensor = torch.from_numpy(tensor)

            classes = self._classify(tensor)
            predicted_classes.extend(classes)

            # Optional gallery
            if self.gallery_size:
                for idx, predicted_class in enumerate(classes):
                    a = self.gallery_dict[predicted_class]
                    # list of random samples, total number of items
                    l, s = a
                    s = s + 1
                    a[1] = s
                    # Image is CYX
                    img = batch[idx]
                    if s <= self.gallery_size:
                        # Gallery size not yet reached
                        l.append(img)
                    else:
                        # Randomly replace a gallery image
                        i = randrange(s)
                        if i < self.gallery_size:
                            l[i] = img

        image_df.insert(len(image_df.columns), self.class_name, predicted_classes)

        return original_image_df.merge(
            image_df[["Cyto_ID", self.class_name]],
            on="Cyto_ID",
            how="left"
        )

    def _classify(self, image_tensor):

        # self.model.eval()
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            # Model takes a tensor of (B, C, H, W)
            outputs = self.model(image_tensor)
            # Find maximum of all elements along dim=1 (i.e. classification): (values, indices)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu()

        class_names = self.class_options
        predicted_class = [class_names[x] for x in predicted]

        return predicted_class

    def _crop(self, image, x0, y0, x1, y1):
        """
        Crops the input image using the provided coordinates.
        :param image (numpy array): The image to be cropped.
        :param x0, y0 (int): Top-left coordinates for cropping.
        :param x1, y1 (int): Bottom-right coordinates for cropping.
        :return: Cropped image as a numpy array. Note: This uses the same underlying data.
        """
        # Crop with numpy to return a 2D image with YX as last dimensions
        i = image.squeeze()
        if i.ndim != 2:
            raise Exception("Image classifier only supports 2D images: " + image.shape)
        return i[y0:y1, x0:x1]

    def _to_uint8(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to uint8 using the (1, 99) percentiles.
        """
        out = np.zeros(image.shape, dtype=np.uint8)
        for c in range(image.shape[-1]):
            data = image[..., c]
            pmin, pmax = np.percentile(data, (1, 99))
            # Rescale 0-1
            data = (data - pmin) / (pmax - pmin)
            # Clip to range
            data[data<0] = 0
            data[data>1] = 1
            # Convert to 0-255
            out[..., c] = (data * 255).astype(np.uint8)
        return out

    def _extract_roi(self, image, binary_mask):
        """
        Extracts the ROI (Region of Interest) from a multi-channel image using the mask.
        :param image (numpy array): Multi-channel input image (YXC).
        :param binary_mask (numpy array): Mask image (YX).
        :return: numpy array: ROI (CYX)
        """
        # Apply the mask to all channels
        roi = np.zeros_like(image)
        for channel in range(image.shape[-1]):
            roi[..., channel] = image[..., channel] * binary_mask
        return roi.transpose((2,0,1))

    def _add_padding(self, image, target_size, padding_value=0):
        """
        Adds padding to a NumPy array image to reach the target size.
        :param image: Input image as a NumPy array (H, W) or (C, H, W).
        :param target_size (tuple): Target size as (height, width).
        :param padding_value (int): Value to use for padding (default: 0).
        :return: np.ndarray: Padded image with the desired target size.
        """
        current_height, current_width = image.shape[-2:]
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
            padding_config = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        else:
            raise ValueError("Unsupported image dimensions. Image must be 2D or 3D.")

        # Apply padding
        padded_image = np.pad(image, padding_config, mode='constant', constant_values=padding_value)
        return padded_image

    def _erase_masks(self, cropped_label: np.ndarray, cx: int, cy: int) -> np.ndarray:
        """
        Erases all masks in the cropped_label (yx format) that do not overlap with the centroid.
        Data is modified in-place.
        """
        # Fast option assumes overlap of centroid with a label
        id = cropped_label[cy, cx]
        if id == 0:
            # This should not happen, log it so the user can investigate
            logger.warning(f"No label at {cx},{cy}")
            # Find closest label
            dmin = np.product(np.array(cropped_label.shape))
            dmin = dmin**2
            for p in regionprops(cropped_label.astype(int)):
                y, x = p.centroid
                d = (cx-x)**2 + (cy-y)**2
                if d < dmin:
                    dmin = d
                    id = p.label
            if id == 0:
                raise Exception(f"No label at {cx},{cy}")

        cropped_label[cropped_label != id] = 0
        return cropped_label
