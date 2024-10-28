#!/usr/bin/env python
import logging
from omero_screen.metadata import Defaults, MetaData, ProjectSetup
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.parse_mip import parse_mip
from omero_screen.general_functions import (
    filter_segmentation,
    omero_connect,
    scale_img,
    correct_channel_order,
)
from omero_screen.omero_functions import upload_masks
from ezomero import get_image

from skimage import measure, io
from skimage.measure import label, regionprops
import pandas as pd
import numpy as np

import torch
from cellpose import models

from PIL import Image as PILImage
from torchvision import models as torch_models
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


logger = logging.getLogger("omero-screen")

class Image:
    """
    generates the corrected images and segmentation masks.
    Stores corrected images as dict, and n_mask, c_mask and cyto_mask arrays.
    """

    def __init__(self, conn, well, image_obj, metadata, project_data, flatfield_dict):
        self._conn = conn
        self._well = well
        self.omero_image = image_obj
        self._meta_data = metadata
        self.dataset_id = project_data.dataset_id
        self._get_metadata()
        self.nuc_diameter = 10 # default value for nuclei diameter for 10x images
        self._flatfield_dict = flatfield_dict
        self.img_dict = self._get_img_dict()
        self.n_mask, self.c_mask, self.cyto_mask = self._segmentation()

    def _get_metadata(self):
        self.channels = self._meta_data.channels
        try:
            self.cell_line = self._meta_data.well_conditions(self._well.getId())[
                "cell_line"
            ]
        except KeyError:
            self.cell_line = self._meta_data.well_conditions(self._well.getId())[
                "Cell_Line"
            ]

        row_list = list("ABCDEFGHIJKL")
        self.well_pos = f"{row_list[self._well.row]}{self._well.column}"

    def _get_img_dict(self):
        """divide image_array with flatfield correction mask and return dictionary "channel_name": corrected image""" 
        img_dict = {}
        image_id = self.omero_image.getId()
        if self.omero_image.getSizeZ() > 1:
            array = parse_mip(image_id, self.dataset_id, self._conn)
        else:
            _, array = get_image(self._conn, image_id)
        
        for channel in list(self.channels.items()):
            corr_img = array[..., channel[1]] / self._flatfield_dict[channel[0]]
            img_dict[channel[0]] = np.squeeze(corr_img, axis=1)
        return img_dict

    def _get_models(self):
        """
        Matches well with cell line and gets model_path for cell line from plate_layout
        :param number: int 0 or 1, 0 for nuclei model, 1 for cell model
        :return: path to model (str)
        """
        cell_line = self.cell_line.replace(" ", "").upper() # remove spaces and make uppercase
        if '40X' in cell_line:
            logger.info("40x image detected, using 40x nuclei model")
            return "40x_Tub_H2B"
        elif '20X' in cell_line:
            logger.info("20x image detected, using 20x nuclei model")
            return "cyto"
        elif cell_line in Defaults["MODEL_DICT"]:
            return Defaults["MODEL_DICT"][cell_line]
        else:
            return Defaults["MODEL_DICT"]["U2OS"]
        

    def _n_segmentation(self):
        if '40X' in self.cell_line.upper():
            self.nuc_diameter = 100
        elif '20X' in self.cell_line.upper():
            self.nuc_diameter = 25
        else:
            self.nuc_diameter = 10
        segmentation_model = models.CellposeModel(
            gpu=True if Defaults["GPU"] else torch.cuda.is_available(),
            model_type=Defaults["MODEL_DICT"]["nuclei"],
        )
        # Get the image array
        img_array = self.img_dict["DAPI"]
    
        # Initialize an array to store the segmentation masks
        segmentation_masks = np.zeros_like(img_array)

        for t in range(img_array.shape[0]):
            # Select the image at the current timepoint
            img_t = img_array[t]
            
            # Prepare the image for segmentation
            scaled_img_t = scale_img(img_t)
            
            # Perform segmentation
            n_channels = [[0, 0]]
            logger.info(f"Segmenting nuclei with diamtere {self.nuc_diameter}")
            try:
                n_mask_array, n_flows, n_styles = segmentation_model.eval(
                    scaled_img_t, channels=n_channels, diameter=self.nuc_diameter, normalize=False
                )
            except IndexError:
                n_mask_array = np.zeros_like(scaled_img_t).astype(np.uint16)
            # Store the segmentation mask in the corresponding timepoint
            segmentation_masks[t] = filter_segmentation(n_mask_array)
        return segmentation_masks

    def _c_segmentation(self):
        """perform cellpose segmentation using cell mask"""
        segmentation_model = models.CellposeModel(
            gpu=True if Defaults["GPU"] else torch.cuda.is_available(),
            model_type=self._get_models(),
        )
        c_channels = [[2, 1]]
        
        # Get the image arrays for DAPI and Tubulin channels
        dapi_array = self.img_dict["DAPI"]
        tub_array = self.img_dict["Tub"]
        
        # Check if the time dimension matches
        assert dapi_array.shape[0] == tub_array.shape[0], "Time dimension mismatch between DAPI and Tubulin channels"
        
        # Initialize an array to store the segmentation masks
        segmentation_masks = np.zeros_like(dapi_array)
        
        # Process each timepoint
        for t in range(dapi_array.shape[0]):
            # Select the images at the current timepoint
            dapi_t = dapi_array[t]
            tub_t = tub_array[t]
            
            # Combine the 2 channel numpy array for cell segmentation with the nuclei channel
            comb_image_t = scale_img(np.dstack([dapi_t, tub_t]))
            
            # Perform segmentation
            try:
                c_masks_array, c_flows, c_styles = segmentation_model.eval(
                    comb_image_t, channels=c_channels, normalize=False
                )
            except IndexError:
                c_masks_array = np.zeros_like(comb_image_t).astype(np.uint16)
            
            # Store the segmentation mask in the corresponding timepoint
            segmentation_masks[t] = filter_segmentation(c_masks_array)
        return segmentation_masks

    def _download_masks(self, image_id):
        """Download masks from OMERO server and save as numpy arrays"""
        _, masks = get_image(self._conn, image_id)
        return correct_channel_order(masks) if masks.shape[-1] == 2 else masks

    def _get_cyto(self):
        """substract nuclei mask from cell mask to get cytoplasm mask"""
        if self.c_mask is None:
            return None
        overlap = (self.c_mask != 0) * (self.n_mask != 0)
        cyto_mask_binary = (self.c_mask != 0) * (overlap == 0)
        return self.c_mask * cyto_mask_binary

    def _segmentation(self):
        # check if masks already exist
        image_name = f"{self.omero_image.getId()}_segmentation"
        dataset_id = self.dataset_id
        dataset = self._conn.getObject("Dataset", dataset_id)
        image_id = None
        for image in dataset.listChildren():
            if image.getName() == image_name:
                image_id = image.getId()
                logger.info(f"Segmentation masks found for image {image_id}")
                if "Tub" in self.channels:
                    self.n_mask, self.c_mask = self._download_masks(image_id)
                    self.cyto_mask = self._get_cyto()
                else:
                    self.n_mask = self._download_masks(image_id)
                    self.c_mask = None
                    self.cyto_mask = None
                break  # stop the loop once the image is found
        if image_id is None:
            self.n_mask = self._n_segmentation()
            if "Tub" in self.channels:
                self.c_mask = self._c_segmentation()
                self.cyto_mask = self._get_cyto()
            else:
                self.c_mask = None
                self.cyto_mask = None
            
            upload_masks(
                self.dataset_id,
                self.omero_image,
                self.n_mask,
                self.c_mask,
                self._conn,
            )
        return self.n_mask, self.c_mask, self.cyto_mask


class ImageProperties:
    """
    Extracts feature measurements from segmented nuclei, cells and cytoplasm
    and generates combined data frames.
    """

    def __init__(self, well, image_obj, meta_data, featurelist=Defaults["FEATURELIST"]):
        self._meta_data = meta_data
        self.plate_name = meta_data.plate_obj.getName()
        self._well = well
        self._well_id = well.getId()
        self._image = image_obj
        self._cond_dict = image_obj._meta_data.well_conditions(self._well_id)
        self._overlay = self._overlay_mask()
        self.image_df = self._combine_channels(featurelist)
        self.quality_df = self._concat_quality_df()
        self._classify_image(image_obj.img_dict, self.image_df, image_obj.c_mask)

    def _overlay_mask(self) -> pd.DataFrame:
        """Links nuclear IDs with cell IDs"""
        if self._image.c_mask is None:
            return pd.DataFrame({"label": self._image.n_mask.flatten()})
        
        overlap = (self._image.c_mask != 0) * (self._image.n_mask != 0)
        list_n_masks = np.stack(
            [self._image.n_mask[overlap], self._image.c_mask[overlap]]
        )[-2].tolist()
        list_masks = np.stack(
            [self._image.n_mask[overlap], self._image.c_mask[overlap]]
        )[-1].tolist()
        overlay_all = {list_n_masks[i]: list_masks[i] for i in range(len(list_n_masks))}
        return pd.DataFrame(list(overlay_all.items()), columns=["label", "Cyto_ID"])

    @staticmethod
    def _edit_properties(channel, segment, featurelist):
        """generates a dictionary with"""
        feature_dict = {
            feature: f"{feature}_{channel}_{segment}" for feature in featurelist[2:]
        }
        feature_dict[
            "area"
        ] = f"area_{segment}"  # the area is the same for each channel
        return feature_dict

    def _get_properties(self, segmentation_mask, channel, segment, featurelist):
        """Measure selected features for each segmented cell in given channel"""
        timepoints = self._image.img_dict[channel].shape[0]
        label = np.squeeze(segmentation_mask).astype(np.int64)
        
        if timepoints > 1:
            data_list = []
            for t in range(timepoints):
                props = measure.regionprops_table(
                    label[t], np.squeeze(self._image.img_dict[channel][t]), properties=featurelist
                )
                data = pd.DataFrame(props)
                feature_dict = self._edit_properties(channel, segment, featurelist)
                data = data.rename(columns=feature_dict)
                data['timepoint'] = t  # Add timepoint for all channels
                data_list.append(data)
            combined_data = pd.concat(data_list, axis=0, ignore_index=True)
            return combined_data.sort_values(by=['timepoint', 'label']).reset_index(drop=True)
        else:
            props = measure.regionprops_table(
                label, np.squeeze(self._image.img_dict[channel]), properties=featurelist
            )
            data = pd.DataFrame(props)
            feature_dict = self._edit_properties(channel, segment, featurelist)
            data = data.rename(columns=feature_dict)
            data['timepoint'] = 0  # Add timepoint 0 for single timepoint data
            return data.sort_values(by=['label']).reset_index(drop=True)

    def _channel_data(self, channel, featurelist):
        nucleus_data = self._get_properties(
            self._image.n_mask, channel, "nucleus", featurelist
        )
        # merge channel data, outer merge combines all area columns into 1
        if self._image.c_mask is not None:
            nucleus_data = pd.merge(
                nucleus_data, self._overlay, how="outer", on=["label"]
            ).dropna(axis=0, how="any")
        if channel == "DAPI":
            nucleus_data["integrated_int_DAPI"] = (
                nucleus_data["intensity_mean_DAPI_nucleus"]
                * nucleus_data["area_nucleus"]
            )
        
        if self._image.c_mask is not None:
            cell_data = self._get_properties(
                self._image.c_mask, channel, "cell", featurelist
            )
            cyto_data = self._get_properties(
                self._image.cyto_mask, channel, "cyto", featurelist
            )
            merge_1 = pd.merge(cell_data, cyto_data, how="outer", on=["label", "timepoint"]).dropna(
                axis=0, how="any"
            )
            merge_1 = merge_1.rename(columns={"label": "Cyto_ID"})
            return pd.merge(nucleus_data, merge_1, how="outer", on=["Cyto_ID", "timepoint"]).dropna(
                axis=0, how="any"
            )
        else:
            return nucleus_data

    def _combine_channels(self, featurelist):
        channel_data = [
            self._channel_data(channel, featurelist)
            for channel in self._meta_data.channels
        ]
        props_data = pd.concat(channel_data, axis=1, join="inner")
        edited_props_data = props_data.loc[:, ~props_data.columns.duplicated()].copy()
        cond_list = [
            self.plate_name,
            self._meta_data.plate_obj.getId(),
            self._well.getWellPos(),
            self._well_id,
            self._image.omero_image.getId(),
        ]
        cond_list.extend(iter(self._cond_dict.values()))
        col_list = ["experiment", "plate_id", "well", "well_id", "image_id"]
        col_list.extend(iter(self._cond_dict.keys()))
        col_list_edited = [entry.lower() for entry in col_list]
        edited_props_data[col_list_edited] = cond_list

        return edited_props_data.sort_values(by=['timepoint']).reset_index(drop=True)

    def _set_quality_df(self, channel, corr_img):
        """generates df for image quality control saving the median intensity of the image"""
        return pd.DataFrame(
            {
                "experiment": [self.plate_name],
                "plate_id": [self._meta_data.plate_obj.getId()],
                "position": [self._image.well_pos],
                "image_id": [self._image.omero_image.getId()],
                "channel": [channel],
                "intensity_median": [np.median(corr_img)],
            }
        )

    def _concat_quality_df(self) -> pd.DataFrame:
        """Concatenate quality dfs for all channels in _corr_img_dict"""
        df_list = [
            self._set_quality_df(channel, image)
            for channel, image in self._image.img_dict.items()
        ]
        return pd.concat(df_list)


    def _crop_image(self, image, x0, y0, x1, y1):
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

        # Convert image to PIL format if it's a numpy array
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)
        
        # Crop the image using the corrected coordinates
        cropped_image = image.crop((x0, y0, x1, y1))
        
        # Convert back to numpy array for further processing
        return np.array(cropped_image)
    
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
    
    @omero_connect
    def _load_model_from_omero(self, project_name, dataset_name, model_filename, conn=None):

        # Check if the model already exists locally
        file_path = f"./{model_filename}"
        if os.path.exists(file_path):
            print(f"Model file '{model_filename}' already exists locally.")
            # Mevcut dosyayı yükleyin
            model = ROIBasedDenseNetModel(num_classes=7)
            model.load_state_dict(torch.load(file_path, weights_only=True))
            model.eval()
            return model

        # Find project
        project = conn.getObject("Project", attributes={"name": project_name})
        if project is None:
            raise ValueError(f"Project '{project_name}' not found in OMERO.")
        
        # Find dataset
        dataset = None
        for ds in project.listChildren():
            if ds.getName() == dataset_name:
                dataset = ds
                break
        if dataset is None:
            raise ValueError(f"Dataset '{dataset_name}' not found in project '{project_name}'.")

        for attachment in dataset.listAnnotations():
            if isinstance(attachment, omero.gateway.FileAnnotationWrapper):
                if attachment.getFileName() == model_filename:
                    # Download the model
                    file_path = f"./{model_filename}"
                    with open(file_path, "wb") as f:
                        for chunk in attachment.getFileInChunks():
                            f.write(chunk)
                    print(f"Downloaded model file to {file_path}")

                    # Load the downloaded model
                    model = ROIBasedDenseNetModel(num_classes=7)
                    model.load_state_dict(torch.load(file_path, weights_only=True))
                    model.eval()
                    return model

        raise FileNotFoundError(f"File '{model_filename}' not found in dataset '{dataset_name}' under project '{project_name}'.")
    
    def _apply_mask_to_image(self, rgb_image, mask, x0, y0, x1, y1):
        """
        Nullify pixels in color images that don't overlap with the corresponding masks.
        Images are expected to be in the shape of (H, W, 3) and masks in the shape of (H, W).
        """

        cropped_mask = self._crop_image(mask, x0, y0, x1, y1) #mask[:rgb_image.shape[1], :rgb_image.shape[0]]
        corrected_mask = self.erase_masks(cropped_mask)
        expanded_mask = np.repeat(corrected_mask[:, :, np.newaxis], rgb_image.shape[2], axis=2)
        masked_image = np.where(expanded_mask > 0, rgb_image, 0)

        return masked_image

    def _classify_image(self, image_data, image_df, mask):

        # Extract the DAPI and TuB channels from image_data
        dapi_image = image_data['DAPI']
        tub_image = image_data['Tub']
        edu_image = image_data['EdU']

        predicted_classes = []

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        model = self._load_model_from_omero("CNN_Models", "ROI_Based_DenseNet", "roi_based_densenet_model.pth")
        model = model.to(device)
        model.eval()

        for i in tqdm(range(len(image_df["centroid-0"]))):

            # Center the crop around the centroid coordinates with a 100x100 area
            half_crop_size = 50  # Half of 100 to create a centered crop

            # Calculate preliminary crop coordinates
            x0 = int(image_df["centroid-1"][i] - half_crop_size)
            y0 = int(image_df["centroid-0"][i] - half_crop_size)
            x1 = int(image_df["centroid-1"][i] + half_crop_size)
            y1 = int(image_df["centroid-0"][i] + half_crop_size)

            # Ensure crop coordinates are within image bounds
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(dapi_image.shape[1], x1)  # width limit
            y1 = min(dapi_image.shape[0], y1)  # height limit

            # Crop DAPI and TuB images
            cropped_dapi = self._crop_image(dapi_image, x0, y0, x1, y1)
            cropped_tub = self._crop_image(tub_image, x0, y0, x1, y1)
            cropped_edu = self._crop_image(edu_image, x0, y0, x1, y1)

            # Normalize the cropped images for better visualization
            dapi_normalized = (cropped_dapi - cropped_dapi.min()) / (cropped_dapi.max() - cropped_dapi.min())
            tub_normalized = (cropped_tub - cropped_tub.min()) / (cropped_tub.max() - cropped_tub.min())
            edu_normalized = (cropped_edu - cropped_edu.min()) / (cropped_edu.max() - cropped_edu.min())

            # Create an RGB image with the same size as the cropped images
            rgb_image = np.zeros((dapi_normalized.shape[0], dapi_normalized.shape[1], 3))
            
            # Assign DAPI to red channel and TuB to green channel
            rgb_image[:, :, 0] = dapi_normalized  # Red channel for DAPI
            rgb_image[:, :, 1] = tub_normalized   # Green channel for TuB
            rgb_image[:, :, 2] = edu_normalized

            cell_image = self._apply_mask_to_image(rgb_image, mask, x0, y0, x1, y1)

            channels = np.stack([cell_image[:, : , 0], cell_image[:, : , 1]], axis=0)

            image_tensor = torch.tensor(channels, dtype=torch.float32)

            image_tensor = image_tensor.unsqueeze(0)  # shape: (1, 2, height, width)

            image_tensor = image_tensor.to(device)

            # Load the model
            # model = ROIBasedDenseNetModel(num_classes=7)
            # model.load_state_dict(torch.load("roi_based_densenet_model.pth", weights_only=True))

            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)

            # print(f"Predicted class: {predicted.item()}")

            class_names = ['anaphase', 'interphase', 'metaphase', 'multipolar', 'prometaphase', 'prophase', 'telophase']

            predicted_classes.append(class_names[int(predicted.item())])
            
        self.image_df["Class"] = predicted_classes


class ROIBasedDenseNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ROIBasedDenseNetModel, self).__init__()

        # Pretrained DenseNet model
        self.roi_model = torch_models.densenet121(pretrained=True)
        self.roi_model.features.conv0 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Set the number of input channels to 2
        self.roi_model.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Downsample output to a fixed size
        num_features_roi = self.roi_model.classifier.in_features
        self.roi_model.classifier = nn.Identity()  # Remove the final layer

        # Fully connected layers
        self.fc1 = nn.Linear(num_features_roi, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, roi):
        # Feature extraction from the ROI
        roi_features = self.roi_model.features(roi)
        roi_features = torch.flatten(roi_features, 1)  # Flattening

        # Classification through fully connected layers
        x = torch.relu(self.fc1(roi_features))
        x = self.fc2(x)
        return x



# test
if __name__ == "__main__":

    @omero_connect
    def feature_extraction_test(conn=None):
        meta_data = MetaData(conn, plate_id=352)
        project_data = ProjectSetup(352, conn)
        well = conn.getObject("Well", 604)
        omero_image = well.getImage(2)
        flatfield_dict = flatfieldcorr(meta_data, project_data, conn)
        image = Image(conn, well, omero_image, meta_data, project_data, flatfield_dict)
        image_data = ImageProperties(well, image, meta_data)
        print(image.n_mask.shape)
        df_image = image_data.image_df
        print(df_image.head())
        print(df_image.columns)
        df_image.to_csv("~/Desktop/image_data.csv")

    feature_extraction_test()
