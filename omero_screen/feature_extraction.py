from omero_screen.data_structure import ExperimentData
from omero_screen.flatfield_corr import FlatFieldCorr
from cellpose import models
from skimage import measure, exposure, color
import pandas as pd
import numpy as np
from omero_screen.general_functions import save_fig, generate_image

featurelist = ['label', 'area', 'intensity_max', 'intensity_mean']
from omero.gateway import BlitzGateway
import json

FEATURELIST = ['label', 'area', 'intensity_max', 'intensity_mean']


class Image:
    """generates the corrected images and segmentation masks.
    ...
    private methods:
            get_array(self, channel: int) reads in the numpy array for each channel. Channel is the value from the
                channel dictionary {'channel_name : channel_number } key value pair from experiment_data object.
            _corr_img_dict(self): generates a dictionary with the 'channel' : corrected image key value pairs that
            is stored as a class property corr_img_dict.
            n, c_segmentation and cyto_segmentation: returns the segmented images as class properties: n_mask, c_mask
            and cyto_mask
    properties:
            corr_img_dict: dictionary containing channel : flat field corrected image key value pair
            n_mask, c_mask, cyto_mask: segmentation masks for nuclei (n_) cell (c_)  and cytoplasm (cyto_)
    """

    # TODO at the moment the segmenting channels have to be called DAPI and Tub. This could be more flexible
    # TODO allow an option for nuclei analysis only for counting assays.
    # TODO allow option for Stardist for greater speed

    def __init__(self, well, image: 'Omero Image Object', experiment_data: ExperimentData,
                 flatfield_corr: FlatFieldCorr):
        """
        parameters:
            well_id: int, identifier for the well that contains the current image. This is info is provided via
            the Omero server.
            image: Omero image object provided from the well to be analysed.
            experiment_data: the ExperimentData class that provides metadata for the image
            flatfield_corr: provides the dictionary the contains the 'channel':correction value pairs
        """
        self._well = well
        self._image = image
        self._exp_data = experiment_data
        self._masks = flatfield_corr.mask_dict
        self._corr_img_dict = self._corr_img_dict()
        self._n_segmentation = self.n_segmentation()
        self._c_segmentation = self.c_segmentation()
        self._cyto_mask = self.get_cyto()

    def get_models(self, number: int) -> 'str':
        """
        Matches well with cell line and gets model_path for cell line from plate_layout
        :param number: int 0 or 1, 0 for nuclei model, 1 for cell model
        :return: path to model (str)
        """
        df = self._exp_data.plate_layout
        cell_line = df[df['Well_ID'] == self._well.getId()].Cell_Line.values[0]
        return self._exp_data.segmentation_models[cell_line][number]

    def _corr_img_dict(self) -> dict:
        """divide image_array with flatfield correction mask and return dictionary "channel_name": corrected image"""
        _corr_img_dict = {}
        for channel in list(
                self._exp_data.channels.items()):  # produces a tuple of channel key value pair (ie ('DAPI':0)
            # get array uses channel number, divide by mask identified from corr_mask_dict via channel name
            corr_img = generate_image(self._image, channel[1]) / self._masks[channel[0]]
            _corr_img_dict[channel[0]] = corr_img  # using channel key here to link each image with its channel
        return _corr_img_dict

    @property
    def corr_img_dict(self) -> dict:
        return self._corr_img_dict

    def n_segmentation(self) -> np.ndarray:
        """perform cellpose segmentation using nuclear mask """
        model = models.CellposeModel(gpu=False, model_type=self.get_models(0))
        n_channels = [[0, 0]]
        n_mask_array, n_flows, n_styles = model.eval(self._corr_img_dict['DAPI'], diameter=15, channels=n_channels)
        return n_mask_array

    @property
    def n_mask(self) -> np.ndarray:
        return self._n_segmentation

    def c_segmentation(self) -> np.ndarray:
        """perform cellpose segmentation using cell mask """
        model = models.CellposeModel(gpu=False, model_type=self.get_models(1))
        c_channels = [[0, 1]]
        # combine the 2 channel numpy array for cell segmentation with the nuclei channel
        comb_image = np.dstack([self._corr_img_dict['DAPI'], self._corr_img_dict['Tub']])
        c_masks_array, c_flows, c_styles = model.eval(comb_image, diameter=15, channels=c_channels)
        return c_masks_array

    @property
    def c_mask(self):
        return self._c_segmentation

    def get_cyto(self) -> np.ndarray:
        """substract nuclei mask from cell mask to get cytoplasm mask """
        overlap = (self._c_segmentation != 0) * (self._n_segmentation != 0)
        cyto_mask_binary = (self._c_segmentation != 0) * (overlap == 0)
        return self._c_segmentation * cyto_mask_binary

    @property
    def cyto_mask(self):
        return self._cyto_mask


class ImageProperties:
    """
    extracts feature measurements from segmented nuclei, cells and cytoplams
    and generates combined data frames.
    """

    def __init__(self, well, image: Image, experiment_data: ExperimentData, featurelist=None):
        """

        :param well_id: Omero well object
        :param image: image to be analysed
        :param experiment_data: meta data for annotation
        :param featurelist: default [area mean and max].
        """
        if featurelist is None:
            featurelist = FEATURELIST
        self.well_id = well.getId()
        self._image = image
        self._overlay = self._overlay_mask()
        self._exp_data = experiment_data
        self.tmp_layout = self._exp_data.plate_layout.loc[self._exp_data.plate_layout["Well_ID"] == well.getId()]
        self.image_properties = self._combine_channels()

    def _overlay_mask(self) -> pd.DataFrame:
        """
        Links nuclear IDs with cell IDs
        :return: pd.Dataframe to add measurements
        """
        overlap = (self._image.c_mask != 0) * (self._image.n_mask != 0)
        list_n_masks = np.stack([self._image.n_mask[overlap], self._image.c_mask[overlap]])[-2].tolist()
        list_masks = np.stack([self._image.n_mask[overlap], self._image.c_mask[overlap]])[-1].tolist()
        overlay_all = {list_n_masks[i]: list_masks[i] for i in range(len(list_n_masks))}
        return pd.DataFrame(list(overlay_all.items()), columns=['label', 'Cyto_ID'])

    def _get_properties(self, segmentation_mask, channel, segment):
        """
        Measure selected features for each segmented cell in given channel (default: area, mean and max int.)

        :param segmentation_mask: segmentation mask attribute generated by Image object
        :param channel: channel key from ExperimentData.channels dictionary
        :param segment: segmented structure to measure  ('nucleus, 'cell', 'cyto')
        :return:
        """
        props = measure.regionprops_table(segmentation_mask, self._image.corr_img_dict[channel], properties=featurelist)
        data = pd.DataFrame(props)
        feature_dict = self._edit_properties(channel, segment)
        return data.rename(columns=feature_dict)

    @staticmethod
    def _edit_properties(channel, segment):
        """generates a dictionary with """
        feature_dict = {feature: f"{feature}_{channel}_{segment}" for feature in featurelist[2:]}
        feature_dict['area'] = f'area_{segment}'  # the area is the same for each channel
        return feature_dict

    def _channel_data(self, channel):
        nucleus_data = self._get_properties(self._image.n_mask, channel, 'nucleus')
        # merge channel data, outer merge combines all area columns into 1
        nucleus_data = pd.merge(nucleus_data, self._overlay, how="outer", on=["label"]).dropna(axis=0, how='any')
        if channel == 'DAPI':
            nucleus_data['integrated_int_DAPI'] = nucleus_data['intensity_mean_DAPI_nucleus'] * nucleus_data[
                'area_nucleus']
        cell_data = self._get_properties(self._image.c_mask, channel, 'cell')
        cyto_data = self._get_properties(self._image.cyto_mask, channel, 'cyto')
        merge_1 = pd.merge(cell_data, cyto_data, how="outer", on=["label"]).dropna(axis=0, how='any')
        merge_1 = merge_1.rename(columns={'label': 'Cyto_ID'})
        return pd.merge(nucleus_data, merge_1, how="outer", on=["Cyto_ID"]).dropna(axis=0, how='any')

    def _combine_channels(self):
        channel_data = [self._channel_data(channel) for channel in self._exp_data.channels]
        props_data = pd.concat(channel_data, axis=1, join="inner")
        edited_props_data = props_data.loc[:, ~props_data.columns.duplicated()].copy()
        cond_list = [self._exp_data.plate_name, self._exp_data.plate_id, self.tmp_layout.iloc[0]["Well"],
                     self.well_id, self._image._image.getId(), self.tmp_layout.iloc[0]["Cell_Line"],
                     self.tmp_layout.iloc[0]["Condition"]]
        edited_props_data[["experiment", "plate_id", "well", "well_id", "image_id", "cell_line", "condition"]] \
            = cond_list
        return edited_props_data


if __name__ == "__main__":
    with open('../secrets/config.json') as file:
        data = json.load(file)
    username = data['username']
    password = data['password']
    excel_path = '/Users/hh65/Desktop/221102_cellcycle_exp5.xlsx'

    conn = BlitzGateway(username, password, host="ome2.hpc.susx.ac.uk")

    conn.connect()
    well = conn.getObject("Well", 10684)
    img = well.getImage(0)
    exp_data = ExperimentData(excel_path, conn=conn)
    flatfield_corr = FlatFieldCorr(well, exp_data)
    image = Image(well, img, exp_data, flatfield_corr)
    image_data = ImageProperties(well, image, exp_data)
    import pathlib
    import matplotlib.pyplot as plt


    def save_fig(fig_id, path=pathlib.Path('/Users/hh65/Desktop'), tight_layout=True, fig_extension="pdf",
                 resolution=300):
        dest = path / f"{fig_id}.{fig_extension}"
        print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(dest, format=fig_extension, dpi=resolution)


    def segmentations_check(image):
        def scale_img(img) -> np.array:
            """ enhance image contract by scaling and return it"""
            percentiles = np.percentile(img, (1, 99))
            return exposure.rescale_intensity(img, in_range=tuple(percentiles))

        dapi_img = scale_img(image.corr_img_dict['DAPI'])
        tub_img = scale_img(image.corr_img_dict['Tub'])
        dapi_color_labels = color.label2rgb(image.n_mask, dapi_img, alpha=0.4, bg_label=0, kind='overlay')
        tub_color_labels = color.label2rgb(image.cyto_mask, dapi_img, alpha=0.4, bg_label=0, kind='overlay')
        fig, ax = plt.subplots(ncols=4, figsize=(16, 7))
        for i in range(4):
            ax[i].axis('off')
            ax[i].title.set_text('timepoint')
        ax[0].imshow(dapi_img, cmap='gray')
        ax[0].title.set_text("DAPI channel")
        ax[1].imshow(dapi_color_labels)
        ax[1].title.set_text("DAPI segmentation")
        ax[2].imshow(tub_img, cmap='gray')
        ax[2].title.set_text("Tubulin image")
        ax[3].imshow(tub_color_labels)
        ax[3].title.set_text("Tubulin Segmentation")
        save_fig('segmentation_check')
        plt.show()


    segmentations_check(image)
    conn.close()

    image_data.image_properties.to_csv('/Users/hh65/Desktop/test_data.csv')
