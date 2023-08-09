#!/usr/bin/env python
from omero_screen import Defaults
import omero
from omero_screen.metadata import Defaults, MetaData, ProjectSetup
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.general_functions import (
    save_fig,
    generate_image,
    filter_segmentation,
    omero_connect,
    scale_img,
    color_label,
)
from omero_screen.omero_functions import upload_masks
from skimage import measure, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from cellpose import models


class NucImage:
    """

    generates the corrected images and segmentation masks.
    Stores corrected images as dict, and n_mask arrays.
    """

    def __init__(self, conn, well, image_obj, meta_data, project_data, flatfield_dict):
        self._conn = conn
        self._well = well
        self.omero_image = image_obj
        self._meta_data = meta_data
        self.dataset_id = project_data.dataset_id
        self._get_metadata()
        self._flatfield_dict = flatfield_dict
        self.img_dict = self._get_img_dict()
        self.n_mask = self._n_segmentation()

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

        # self.condition = self._meta_data.well_conditions(self._well.getId())['condition']
        row_list = list("ABCDEFGHIJKL")
        self.well_pos = f"{row_list[self._well.row]}{self._well.column}"

    def _get_img_dict(self):
        """divide image_array with flatfield correction mask and return dictionary "channel_name": corrected image"""
        img_dict = {}
        for channel in list(
            self.channels.items()
        ):  # produces a tuple of channel key value pair (ie ('DAPI':0)
            corr_img = (
                generate_image(self.omero_image, channel[1])
                / self._flatfield_dict[channel[0]]
            )
            img_dict[channel[0]] = corr_img[
                30:1050, 30:1050
            ]  # cropping the image to avoid flat field corr problems at the border
        return img_dict

    def _download_masks(self, image_id):
        """Download masks from OMERO server and save as numpy arrays"""
        masks = self._conn.getObject("Image", image_id)
        return generate_image(masks, 0)

    def _n_segmentation(self):
        """perform cellpose segmentation using nuclear mask"""
        image_name = f"{self.omero_image.getId()}_segmentation"
        dataset_id = self.dataset_id
        dataset = self._conn.getObject("Dataset", dataset_id)
        image_id = None
        for image in dataset.listChildren():
            if image.getName() == image_name:
                image_id = image.getId()
                print(f"Found image with ID: {image_id}")
                self._n_mask_array = self._download_masks(image_id)
                break  # stop the loop once the image is found
        if image_id is None:
            if torch.cuda.is_available():
                segmentation_model = models.CellposeModel(
                    gpu=True, model_type=Defaults["MODEL_DICT"]["nuclei"]
                )
            else:
                segmentation_model = models.CellposeModel(
                    gpu=False, model_type=Defaults["MODEL_DICT"]["nuclei"]
                )
            n_channels = [[0, 0]]
            self._n_mask_array, n_flows, n_styles = segmentation_model.eval(
                self.img_dict["DAPI"], channels=n_channels
            )
            upload_masks(
                self.dataset_id, self.omero_image, [self._n_mask_array], self._conn
            )
            # return cleaned up mask using filter function
        return filter_segmentation(self._n_mask_array)


class NucImageProperties:
    """
    Extracts feature measurements from segmented nuclei
    and generates combined data frames.
    """

    def __init__(self, well, image_obj, meta_data, featurelist=None):
        if featurelist is None:
            featurelist = Defaults["FEATURELIST"]
        self._meta_data = meta_data
        self.plate_name = meta_data.plate_obj.getName()
        self._well = well
        self._well_id = well.getId()
        self._image = image_obj
        self._cond_dict = image_obj._meta_data.well_conditions(self._well_id)
        self.image_df = self._combine_channels(featurelist)
        self.quality_df = self._concat_quality_df()

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
        props = measure.regionprops_table(
            segmentation_mask, self._image.img_dict[channel], properties=featurelist
        )
        data = pd.DataFrame(props)
        feature_dict = self._edit_properties(channel, segment, featurelist)
        return data.rename(columns=feature_dict)

    def _channel_data(self, channel, featurelist):
        nucleus_data = self._get_properties(
            self._image.n_mask, channel, "nucleus", featurelist
        )
        if channel == "DAPI":
            nucleus_data["integrated_int_DAPI"] = (
                nucleus_data["intensity_mean_DAPI_nucleus"]
                * nucleus_data["area_nucleus"]
            )
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

        return edited_props_data

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
