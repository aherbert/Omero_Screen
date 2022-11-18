"""Module to organise the experiment data

Reads data from excel_path and stores them

"""
# TODO switch to json config file for Defaults
# TODO switch experiment meta data from excelfile to omero directly
import pandas as pd
import pathlib
from omero.gateway import BlitzGateway
import json


class Defaults:
    """Store the default variables to read the Excel input file"""
    DEFAULT_DEST_DIR = "Desktop"  # Decides where the final data folder will be made
    ID_INFO_TAB = "Plate_ID"
    ID_INFO_LOCATION = ("Plate_ID", 0)
    CHANNELS_TAB = "Channels"
    LAYOUT_TAB = "Plate_Layout"
    LAYOUT_COLS = ("Well", "Well_ID", "Cell_Line", "Condition")
    SEGMENTATION_TAB = "Segmentation"
    FLATFIELD_TEMPLATES = "flatfield_correction_images"
    FLATFIELD_REPRESENTATIVE = "flatfield_correction_representative_images"
    DATA = "single_cell_data"
    QUALITY_CONTROL = "quality_control"
    IMGS_CORR = "images_corrected"


class ExperimentData:
    """ Extract experiment info from Excel, Plate Name from Omero and generates hierarchy of directories. Store paths
    as class properties """

    def __init__(self, excel_path, conn):
        self._excel_path = excel_path
        self._extract_excel_data()
        self.plate = conn.getObject("Plate", self.plate_id)
        self._create_dir_paths()
        self._create_exp_dir()
        self._plate_name = self.plate.getName()

    def _extract_excel_data(self):
        """Read Tabs from Excel and stor as class properties
        Attributes: plate_id (int), channels (dict), plate_layout (df), segmentation models (paths)

        """

        self._priv_plate_id: int = pd.read_excel(self._excel_path, Defaults.ID_INFO_TAB)[Defaults.ID_INFO_LOCATION[0]][
            Defaults.ID_INFO_LOCATION[1]]
        self._priv_channels: dict = \
            pd.read_excel(self._excel_path, Defaults.CHANNELS_TAB).set_index(Defaults.CHANNELS_TAB).to_dict("dict")[
                "Index"]
        self._priv_plate_layout = pd.read_excel(self._excel_path, "Plate_Layout",
                                                converters={Defaults.LAYOUT_COLS[0]: str, Defaults.LAYOUT_COLS[0]: int,
                                                            Defaults.LAYOUT_COLS[0]: str,
                                                            Defaults.LAYOUT_COLS[0]: str}).dropna()
        self._priv_segmentation_models: dict = pd.read_excel(self._excel_path, Defaults.SEGMENTATION_TAB).drop(
            Defaults.SEGMENTATION_TAB, axis=1).to_dict("list")

    def _create_dir_paths(self):
        """ Generate path attributes for experiment"""
        self._priv_path = pathlib.Path.home() / Defaults.DEFAULT_DEST_DIR / f"{self.plate.getName()}"
        self._priv_flatfield = self._priv_path / Defaults.FLATFIELD_TEMPLATES
        self._priv_flatfield_imgs = self._priv_path / Defaults.FLATFIELD_REPRESENTATIVE
        self._priv_final_data = self._priv_path / Defaults.DATA
        self._priv_quality_ctr = self._priv_path / Defaults.QUALITY_CONTROL
        self._priv_corr_imgs = self._priv_path / Defaults.IMGS_CORR

    def _create_exp_dir(self):
        path_list = [self._priv_path, self._priv_flatfield, self._priv_flatfield_imgs, self._priv_final_data,
                     self._priv_quality_ctr, self._priv_corr_imgs]
        for path in path_list:
            path.mkdir(exist_ok=True)
        self._priv_plate_layout.to_csv(self._priv_path / "Plate_layout.csv")
        print(f'\nGathering data and assembling directories for experiment {self.plate.getName()}\n'
              '------------------------------------------------------------------------------------------------------')

    # turn private attributes from original Excel File to properties (opt.! error messages for setter functions)

    @property
    def plate_name(self):
        return self._plate_name

    @property
    def plate_id(self):
        return self._priv_plate_id

    # @plate_id.setter # adds warning message if attribute is changed
    # def plate_id(self, new_value):
    #     # optional for neater error message
    #     raise ValueError("the Plate ID cannot be modified")

    @property
    def channels(self):
        return self._priv_channels

    @property
    def plate_layout(self):
        return self._priv_plate_layout

    @property
    def segmentation_models(self):
        return self._priv_segmentation_models

    # turn private path attributes to properties

    @property
    def main_path(self):
        return self._priv_path

    @property
    def flatfield_path(self):
        return self._priv_flatfield

    @property
    def flatfield_imgs_path(self):
        return self._priv_flatfield_imgs

    @property
    def final_data_path(self):
        return self._priv_final_data

    @property
    def quality_ctr_path(self):
        return self._priv_quality_ctr

    @property
    def corr_imgs_path(self):
        return self._priv_corr_imgs


if __name__ == "__main__":
    excel_path = '/Users/hh65/Desktop/221102_cellcycle_exp5.xlsx'

    with open('../secrets/config.json') as file:
        data = json.load(file)
    username = data['username']
    password = data['password']

    conn = BlitzGateway(username, password, host="ome2.hpc.susx.ac.uk")
    conn.connect()

    exp_data = ExperimentData(excel_path, conn)

    conn.close()

    print(exp_data.plate_layout)
