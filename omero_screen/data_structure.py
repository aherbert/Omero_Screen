"""Module to organise the experiment data

Reads data from excel_path and stores them

"""
# TODO switch to json config file for Defaults
# TODO switch experiment meta data from excelfile to omero directly
import pandas as pd
import pathlib
from omero_screen import EXCEL_PATH, SEPARATOR



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


class MetaData:
    """ Extract experiment info from Excel, Plate Name from Omero.
     Class Attributes, Various Plate MetaData (Plate_ID,
     Class Methods Extract Well specific information
     """

    def __init__(self, excel_path):
        self._excel_path = excel_path
        self._extract_excel_data()
        self.df_final = pd.DataFrame()
        self.df_quality = pd.DataFrame()

    def _extract_excel_data(self):
        """Read Tabs from Excel and stor as class properties
        Attributes: plate_id (int), channels (dict), plate_layout (df), segmentation models (paths)

        """
        self.plate_id: int = pd.read_excel(self._excel_path, Defaults.ID_INFO_TAB)[Defaults.ID_INFO_LOCATION[0]][
            Defaults.ID_INFO_LOCATION[1]]
        self.channels: dict = \
            pd.read_excel(self._excel_path, Defaults.CHANNELS_TAB).set_index(Defaults.CHANNELS_TAB).to_dict("dict")[
                "Index"]
        self.plate_layout = pd.read_excel(self._excel_path, "Plate_Layout",
                                          converters={Defaults.LAYOUT_COLS[0]: str, Defaults.LAYOUT_COLS[0]: int,
                                                      Defaults.LAYOUT_COLS[0]: str,
                                                      Defaults.LAYOUT_COLS[0]: str}).dropna()
        self.segmentation_models: dict = pd.read_excel(self._excel_path, Defaults.SEGMENTATION_TAB).drop(
            Defaults.SEGMENTATION_TAB, axis=1).to_dict("list")
        self._priv_segmentation_models: dict = pd.read_excel(self._excel_path, Defaults.SEGMENTATION_TAB).drop(
            Defaults.SEGMENTATION_TAB, axis=1).to_dict("list")
        self.plate_length = len(self.plate_layout)

    def well_pos(self, current_well_id):
        df = self.plate_layout
        return df[df['Well_ID'] == current_well_id]['Well'].iloc[0]

    def well_cell_line(self, current_well_id):
        df = self.plate_layout
        return df[df['Well_ID'] == current_well_id]['Cell_Line'].iloc[0]

    def well_condition(self, current_well_id):
        df = self.plate_layout
        return df[df['Well_ID'] == current_well_id]['Condition'].iloc[0]


class ExpPaths:
    def __init__(self, conn, meta_data: MetaData):
        self.meta_data = meta_data
        self.plate = conn.getObject("Plate", self.meta_data.plate_id)
        self.plate_name = self.plate.getName()
        self._create_dir_paths()
        self._create_exp_dir()

    def _create_dir_paths(self):
        """ Generate path attributes for experiment"""
        self.path = pathlib.Path.home() / Defaults.DEFAULT_DEST_DIR / f"{self.plate.getName()}"
        self.flatfield_templates = self.path / Defaults.FLATFIELD_TEMPLATES
        self.flatfield_rep_figs = self.path / Defaults.FLATFIELD_REPRESENTATIVE
        self.final_data = self.path / Defaults.DATA
        self.quality_ctr = self.path / Defaults.QUALITY_CONTROL
        self.example_img = self.path / Defaults.IMGS_CORR

    def _create_exp_dir(self):
        path_list = [self.path, self.flatfield_templates, self.flatfield_rep_figs, self.final_data,
                     self.quality_ctr, self.example_img]
        for path in path_list:
            path.mkdir(exist_ok=True)
        self.meta_data.plate_layout.to_csv(self.path / "Plate_layout.csv")
        print(f'Gathering data and assembling directories for experiment {self.plate.getName()}\n{SEPARATOR}')


if __name__ == "__main__":
    meta_data = MetaData(EXCEL_PATH)

    print(meta_data.well_pos())
