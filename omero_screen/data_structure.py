#!/usr/bin/env python
"""Module to organise the experiment data

Reads data from excel_path and stores them

"""

import pandas as pd
import pathlib
from omero_screen import Defaults, SEPARATOR
from omero_screen.general_functions import omero_connect





class MetaData:
    """ Extract experiment info from Excel, Plate Name from Omero.
     Class Attributes, Various Plate MetaData (Plate_ID,
     Class Methods Extract Well specific information
     """

    def __init__(self, plate_id, conn):
        self.conn = conn
        self.plate_obj = self.conn.getObject("Plate", plate_id)
        self.plate = self.plate_obj.getName()
        self._extract_meta_data()
        self.df_final = pd.DataFrame()
        self.df_quality = pd.DataFrame()

    def _extract_meta_data(self):
        """Read Tabs from Excel and stor as class properties
        Attributes: plate_id (int), channels (dict), plate_layout (df), segmentation models (paths)

        """

        ann = self.plate_obj.getAnnotation(Defaults['NS'])
        channels = dict(ann.getValue())
        if 'Hoechst' in channels:
            channels['DAPI'] = channels.pop('Hoechst')
        # changing channel number to integer type
        for key in channels:
            channels[key] = int(channels[key])
        self.channels = channels
        self.plate_length = len(list(self.plate_obj.listChildren()))

    def well_conditions(self, current_well):
        well = self.conn.getObject("Well", current_well)
        ann = well.getAnnotation(Defaults['NS'])
        return dict(ann.getValue())



class ExpPaths:
    def __init__(self, meta_data: MetaData):
        self.meta_data = meta_data
        self._create_dir_paths()
        self._create_exp_dir()

    def _create_dir_paths(self):
        """ Generate path attributes for experiment"""
        self.path = pathlib.Path.home() / Defaults['DEFAULT_DEST_DIR'] / f"{self.meta_data.plate}"
        self.flatfield_templates = self.path / Defaults['FLATFIELD_TEMPLATES']
        self.final_data = self.path / Defaults['DATA']
        self.temp_well_data = self.path / Defaults['TEMP_WELL_DATA']

    def _create_exp_dir(self):
        path_list = [self.path, self.flatfield_templates, self.final_data,
                     self.temp_well_data]
        for path in path_list:
            path.mkdir(exist_ok=True)

        print(f'Gathering data and assembling directories for experiment {self.meta_data.plate}\n{SEPARATOR}')

@omero_connect
def test_module(conn=None):
    meta_data = MetaData(1107, conn)
    paths = ExpPaths(meta_data)
    print(meta_data.well_conditions(12760)['cell_line'])
    print(meta_data.channels)



if __name__ == "__main__":
    print( pathlib.Path.home() / Defaults['DEFAULT_DEST_DIR'])
    # meta_data = test_module()

