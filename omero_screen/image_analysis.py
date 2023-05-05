#!/usr/bin/env python
from omero_screen.data_structure import Defaults, MetaData, ExpPaths
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.general_functions import save_fig, generate_image, filter_segmentation, omero_connect, scale_img, \
    color_label


from skimage import measure, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from cellpose import models





class Image:
    """
    generates the corrected images and segmentation masks.
    Stores corrected images as dict, and n_mask, c_mask and cyto_mask arrays.
    """

    def __init__(self, well, omero_image, meta_data, exp_paths, flatfield_dict):
        self._well = well
        self.omero_image = omero_image
        self._meta_data = meta_data
        self._paths = exp_paths
        self._get_metadata()
        self._flatfield_dict = flatfield_dict
        self.img_dict = self._get_img_dict()
        self.n_mask = self._n_segmentation()
        self.c_mask = self._c_segmentation()
        self.cyto_mask = self._get_cyto()

    def _get_metadata(self):
        self.channels = self._meta_data.channels
        try:
            self.cell_line = self._meta_data.well_conditions(self._well.getId())['cell_line']
        except KeyError:
            self.cell_line = self._meta_data.well_conditions(self._well.getId())['Cell_Line']


        # self.condition = self._meta_data.well_conditions(self._well.getId())['condition']
        row_list = list('ABCDEFGHIJKL')
        self.well_pos = f"{row_list[self._well.row]}{self._well.column}"

    def _get_img_dict(self):
        """divide image_array with flatfield correction mask and return dictionary "channel_name": corrected image"""
        img_dict = {}
        for channel in list(self.channels.items()):  # produces a tuple of channel key value pair (ie ('DAPI':0)
            corr_img = generate_image(self.omero_image, channel[1]) / self._flatfield_dict[channel[0]]
            bgcorr_img = corr_img - np.percentile(corr_img, 0.1) +1
            img_dict[channel[0]] = bgcorr_img  # using channel key here to link each image with its channel
        return img_dict

    def _get_models(self):
        """
        Matches well with cell line and gets model_path for cell line from plate_layout
        :param number: int 0 or 1, 0 for nuclei model, 1 for cell model
        :return: path to model (str)
        """
        return Defaults.MODEL_DICT[self.cell_line.replace(" ", "").upper()]

    def _n_segmentation(self):
        """perform cellpose segmentation using nuclear mask """
        if torch.cuda.is_available():
            segmentation_model = models.CellposeModel(gpu=True, model_type=Defaults.MODEL_DICT['nuclei'])
        else:
            segmentation_model = models.CellposeModel(gpu=False, model_type=Defaults.MODEL_DICT['nuclei'])


        n_channels = [[0, 0]]
        n_mask_array, n_flows, n_styles = segmentation_model.eval(self.img_dict['DAPI'], channels=n_channels)

        # return cleaned up mask using filter function
        return filter_segmentation(n_mask_array)

    def _c_segmentation(self):
        """perform cellpose segmentation using cell mask """
        if torch.cuda.is_available():
            segmentation_model = models.CellposeModel(gpu=True, model_type=self._get_models())
        else:
            segmentation_model = models.CellposeModel(gpu=False, model_type=self._get_models())
        c_channels = [[2, 1]]
        # combine the 2 channel numpy array for cell segmentation with the nuclei channel
        comb_image = np.dstack([self.img_dict['DAPI'], self.img_dict['Tub']])
        c_masks_array, c_flows, c_styles = segmentation_model.eval(comb_image, channels=c_channels)
        # return cleaned up mask using filter function
        return filter_segmentation(c_masks_array)

    def _get_cyto(self):
        """substract nuclei mask from cell mask to get cytoplasm mask """
        overlap = (self.c_mask != 0) * (self.n_mask != 0)
        cyto_mask_binary = (self.c_mask != 0) * (overlap == 0)
        return self.c_mask * cyto_mask_binary

    def segmentation_figure(self):
        """Generate matplotlib image for segmentation check and save to path (quality control)
        """
        dapi_img = scale_img(self.img_dict['DAPI'])
        tub_img = scale_img(self.img_dict['Tub'])
        dapi_color_labels = color_label(self.n_mask, dapi_img)
        tub_color_labels = color_label(self.cyto_mask, dapi_img)
        fig_list = [dapi_img, tub_img, dapi_color_labels, tub_color_labels]
        title_list = ["DAPI image", "Tubulin image", "DAPI segmentation", "Tubulin image"]
        fig, ax = plt.subplots(ncols=4, figsize=(16, 7))
        for i in range(4):
            ax[i].axis('off')
            ax[i].imshow(fig_list[i])
            ax[i].title.set_text(title_list[i])
        save_fig(self._paths.quality_ctr, f'{self.well_pos}_segmentation_check')
        plt.close(fig)

    def save_example_tiff(self):
        """Combines arrays from image_dict and saves images as tif files"""
        comb_image = np.dstack(list(self.img_dict.values()))
        io.imsave(str(self._paths.example_img / f'{self.well_pos}_segmentation_check.tif'), comb_image,
                  check_contrast=False)


class ImageProperties:
    """
    Extracts feature measurements from segmented nuclei, cells and cytoplasm
    and generates combined data frames.
    """

    def __init__(self, well, image_obj, meta_data, exp_paths, featurelist=None):
        if featurelist is None:
            featurelist = Defaults.FEATURELIST
        self._meta_data = meta_data
        self.plate_name = meta_data.plate
        self._well = well
        self._well_id = well.getId()
        self._image = image_obj
        self._cond_dict = image_obj._meta_data.well_conditions(self._well_id)
        self._overlay = self._overlay_mask()
        self.image_df = self._combine_channels(featurelist)
        self.quality_df = self._concat_quality_df()


    def _overlay_mask(self) -> pd.DataFrame:
        """Links nuclear IDs with cell IDs"""
        overlap = (self._image.c_mask != 0) * (self._image.n_mask != 0)
        list_n_masks = np.stack([self._image.n_mask[overlap], self._image.c_mask[overlap]])[-2].tolist()
        list_masks = np.stack([self._image.n_mask[overlap], self._image.c_mask[overlap]])[-1].tolist()
        overlay_all = {list_n_masks[i]: list_masks[i] for i in range(len(list_n_masks))}
        return pd.DataFrame(list(overlay_all.items()), columns=['label', 'Cyto_ID'])

    @staticmethod
    def _edit_properties(channel, segment, featurelist):
        """generates a dictionary with """
        feature_dict = {feature: f"{feature}_{channel}_{segment}" for feature in featurelist[2:]}
        feature_dict['area'] = f'area_{segment}'  # the area is the same for each channel
        return feature_dict

    def _get_properties(self, segmentation_mask, channel, segment, featurelist):
        """Measure selected features for each segmented cell in given channel"""
        props = measure.regionprops_table(segmentation_mask, self._image.img_dict[channel], properties=featurelist)
        data = pd.DataFrame(props)
        feature_dict = self._edit_properties(channel, segment, featurelist)
        return data.rename(columns=feature_dict)

    def _channel_data(self, channel, featurelist):
        nucleus_data = self._get_properties(self._image.n_mask, channel, 'nucleus', featurelist)
        # merge channel data, outer merge combines all area columns into 1
        nucleus_data = pd.merge(nucleus_data, self._overlay, how="outer", on=["label"]).dropna(axis=0, how='any')
        if channel == 'DAPI':
            nucleus_data['integrated_int_DAPI'] = nucleus_data['intensity_mean_DAPI_nucleus'] * nucleus_data[
                'area_nucleus']
        cell_data = self._get_properties(self._image.c_mask, channel, 'cell', featurelist)
        cyto_data = self._get_properties(self._image.cyto_mask, channel, 'cyto', featurelist)
        merge_1 = pd.merge(cell_data, cyto_data, how="outer", on=["label"]).dropna(axis=0, how='any')
        merge_1 = merge_1.rename(columns={'label': 'Cyto_ID'})
        return pd.merge(nucleus_data, merge_1, how="outer", on=["Cyto_ID"]).dropna(axis=0, how='any')

    def _combine_channels(self, featurelist):
        channel_data = [self._channel_data(channel, featurelist) for channel in self._meta_data.channels]
        props_data = pd.concat(channel_data, axis=1, join="inner")
        edited_props_data = props_data.loc[:, ~props_data.columns.duplicated()].copy()
        cond_list = [
            self.plate_name,
            self._meta_data.plate_obj.getId(),
            self._image.well_pos,
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
        return pd.DataFrame({"experiment": [self.plate_name],
                             "plate_id": [self._meta_data.plate_obj.getId()],
                             "position": [self._image.well_pos],
                             "image_id": [self._image.omero_image.getId()],
                             "channel": [channel],
                             "intensity_median": [np.median(corr_img)]})

    def _concat_quality_df(self) -> pd.DataFrame:
        """Concatenate quality dfs for all channels in _corr_img_dict"""
        df_list = [self._set_quality_df(channel, image) for channel, image in self._image.img_dict.items()]
        return pd.concat(df_list)


# test


if __name__ == "__main__":
    @omero_connect
    def feature_extraction_test(conn=None):
        meta_data = MetaData(1107, conn)
        exp_paths = ExpPaths(meta_data)
        well = conn.getObject("Well", 12757)
        omero_image = well.getImage(0)
        flatfield_dict = flatfieldcorr(well, meta_data, exp_paths)
        print(Image(well, omero_image, meta_data, exp_paths, flatfield_dict))
        image = Image(well, omero_image, meta_data, exp_paths, flatfield_dict)
        image_data = ImageProperties(well, image, meta_data, exp_paths)
        image.segmentation_figure()
        df_final = image_data.image_df
        df_final = pd.concat([df_final.loc[:, 'experiment':], df_final.loc[:, :'experiment']], axis=1).iloc[:, :-1]
        print(df_final)




    feature_extraction_test()
