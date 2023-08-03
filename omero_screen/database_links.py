#!/usr/bin/env python
"""Module to link to the omero db, extract metadata and link a project/dataset to the plate."""
import tempfile
from omero_screen.general_functions import omero_connect
import omero
import pandas as pd
from omero.gateway import DatasetWrapper, FileAnnotationWrapper, MapAnnotationWrapper
from omero.rtypes import rstring
from omero_screen import Defaults
from omero_screen.omero_functions import add_map_annotations, delete_map_annotations



class MetaData:
    """Class to add the metadata to the plate."""

    def __init__(self, conn, plate_id=None):
        self.conn = conn
        self.plate_id = plate_id
        self.plate_obj = self.conn.getObject("Plate", self.plate_id)
        self.plate_length = len(list(self.plate_obj.listChildren()))
        self.channels, self.well_inputs = self._get_metadata()
        self._set_well_inputs()


    def _get_metadata(self):
        """
        Get the metadata from the Excel file attached to the plate.
        :return: a dictionary with the channel data and a pandas DataFrame
        with the well data if the Excel file is found. Otherwise return none
        """
        file_anns = self.plate_obj.listAnnotations()

        for ann in file_anns:
            if isinstance(ann, FileAnnotationWrapper) and ann.getFile().getName().endswith(
                    'metadata.xlsx'):
                return self._get_channel_data_from_excel(ann)

        return self._get_channel_data_from_map()

    def _get_channel_data_from_map(self):
        annotations = self.plate_obj.listAnnotations()
        map_annotations = [ann for ann in annotations if isinstance(ann, omero.gateway.MapAnnotationWrapper)]

        for map_ann in map_annotations:
            map_data = dict(map_ann.getValue())
            if 'DAPI' in map_data or 'Hoechst' in map_data:
                print("Found map annotation with 'DAPI' or 'Hoechst'")
                key_value_data = map_ann.getValue()
                return self._get_channel_data(key_value_data), None

        raise ValueError("No map annotation available and Excel file not found.")


    def _get_channel_data_from_excel(self, ann):
        self._clear_map_annotation()
        print("Found Excel file:", ann.getFile().getName())
        original_file = ann.getFile()
        with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp:
            self._download_file_to_tmp(original_file, tmp)
            data = pd.read_excel(tmp.name, sheet_name=None)
        key_value_data = data['Sheet1'].astype(str).values.tolist()
        add_map_annotations(self.plate_obj, key_value_data, conn=self.conn)
        channel_data = {row[0]: row[1] for row in key_value_data}
        well_data = data['Sheet2']
        return self._get_channel_data(channel_data), well_data

    def _clear_map_annotation(self):
        if map_ann := self.plate_obj.getAnnotation(Defaults['NS']):
            map_ann.setValue([])  # Set a new empty list
            map_ann.save()

    def _download_file_to_tmp(self, original_file, tmp):
        with open(tmp.name, 'wb') as f:
            for chunk in original_file.asFileObj():
                f.write(chunk)

    def _get_channel_data(self, key_value_data):
        """"""
        channels = dict(key_value_data)
        if 'Hoechst' in channels:
            channels['DAPI'] = channels.pop('Hoechst')
        # changing channel number to integer type
        for key in channels:
            channels[key] = int(channels[key])
        return channels


    def _set_well_inputs(self):
        """Function to deal with the well metadata"""
        # if there are no well input data check if there are metadata already present
        if self.well_inputs is None:
            if not self._found_cell_line():
                raise ValueError("Well metadata are not present")
        else:
            df = self.well_inputs
            df_dict = {row['Well']: [[col, row[col]] for col in df.columns if col != 'Well'] for _, row in df.iterrows()}
            for well in self.plate_obj.listChildren():
                # overwrite map annotation if present
                delete_map_annotations(well, conn=self.conn)
                wellname = self.convert_well_names(well)
                for key in df_dict:
                    if wellname == key:
                        add_map_annotations(well, df_dict[key], conn=self.conn)



    def convert_well_names(self, well):
        row_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # assuming no more than 26 rows
        row_number = well.row
        column_number = well.column + 1
        return f'{row_letters[row_number]}{column_number}'

    def _found_cell_line(self):
        """
        Checks if the plate with id 'plate_id' contains a 'cell_line' annotation for all wells
        """
        plate = self.conn.getObject("Plate", self.plate_id)
        if plate is None:
            print(f"Cannot find plate: {self.plate_id}")
            return False

        well_list = []
        for well in plate.listChildren():
            annotations = [ann for ann in well.listAnnotations() if isinstance(ann, MapAnnotationWrapper)]
            found_cell_line = any(
                'cell_line' in dict(ann.getValue()) for ann in annotations
            )

            if not found_cell_line:
                well_list.append(well.id)

        if well_list:
            print(f"Found {len(well_list)} wells without a 'cell_line' annotation")
        else:
            print("All wells have a 'cell_line' annotation")

        return not well_list

    def well_conditions(self, current_well):
        """Method to get the well conditions from the well metadata"""
        well = self.conn.getObject("Well", current_well)
        ann = well.getAnnotation(Defaults['NS'])
        return dict(ann.getValue())

class ProjectSetup:
    """Class to set up the Omero-Screen project and organise the metadata"""

    def __init__(self, plate_id, conn):
        self.conn = conn
        self.plate_id = plate_id
        self.project_id = self._create_project()
        self.dataset_id = self._create_dataset()
        self._link_project_dataset()

    def _create_project(self):

        """Check for a Screen project to store the linked data set, if not present, create it."""

        # Fetch all projects
        project_name = 'Screens'
        projects = list(self.conn.getObjects("Project"))
        project_names = [p.getName() for p in projects]

        # Check if project exists
        if project_name in project_names:
            # Get the project
            project = projects[project_names.index(project_name)]
            project_id = project.getId()
            print("Project exists")
        else:
            print("Project does not exist. Creating now...")

            # Create a new project
            project = omero.model.ProjectI()
            project.setName(rstring(project_name))
            project.setDescription(rstring('This is a description'))

            # Save the project
            update_service = self.conn.getUpdateService()
            saved_project = update_service.saveAndReturnObject(project)
            project_id = saved_project.getId().getValue()

        print('Project ID:', project_id)

        return project_id

    def _create_dataset(self):
        """Create a new dataset."""

        # Fetch all datasets
        datasets = self.conn.getObjects("Dataset")
        dataset_name = str(self.plate_id)
        # Check if dataset exists
        for dataset in datasets:
            if dataset.getName() == dataset_name:
                print("Dataset exists, Id:", dataset.getId())
                return dataset.getId()
        # If code reaches here, dataset doesn't exist. Create a new dataset
        print("Dataset does not exist. Creating now...")
        new_dataset = DatasetWrapper(self.conn, omero.model.DatasetI())
        new_dataset.setName(dataset_name)
        new_dataset.save()
        print("New dataset, Id:", new_dataset.getId())
        # self.link_project_dataset()
        return new_dataset.getId()

    def _link_project_dataset(self):
        """Link a project and a dataset."""
        # Fetch the project
        project = self.conn.getObject("Project", self.project_id)
        if not project:
            print("Project not found")
            return

        # Iterate over linked datasets to check if our dataset is already linked
        for dataset in project.listChildren():
            if dataset.getId() == self.dataset_id:
                print("Link already exists")
                return

        # If we reach here, it means the dataset is not linked to the project. So, create a new link.
        link = omero.model.ProjectDatasetLinkI()
        link.setChild(omero.model.DatasetI(self.dataset_id, False))
        link.setParent(omero.model.ProjectI(self.project_id, False))
        self.conn.getUpdateService().saveObject(link)
        print("Link created")

#
# class MetaData:
#     """ Extract Plate Name from Omero.
#      Class Attributes, Various Plate MetaData (Plate_ID,
#      Class Methods: Extract Well specific information
#      """
#
#     def __init__(self, plate_id, conn):
#         self.plate_id = plate_id
#         self.conn = conn
#         self.plate_obj = self.conn.getObject("Plate", self.plate_id)
#         self.plate = self.plate_obj.getName()
#         self._extract_meta_data()
#         self.data_set = self._create_dataset(plate_id)
#         self.df_final = pd.DataFrame()
#         self.df_quality = pd.DataFrame()
#
#     def _extract_meta_data(self):
#         """ Extract metadata from Omero Plate object
#         Attributes:
#             plate_id (int),
#             channels (dict),
#             plate_layout (df),
#             segmentation models (paths)
#         """
#
#         ann = self.plate_obj.getAnnotation(Defaults['NS'])
#         channels = dict(ann.getValue())
#         if 'Hoechst' in channels:
#             channels['DAPI'] = channels.pop('Hoechst')
#         # changing channel number to integer type
#         for key in channels:
#             channels[key] = int(channels[key])
#         self.channels = channels
#         self.plate_length = len(list(self.plate_obj.listChildren()))
#
#     def well_conditions(self, current_well):
#         well = self.conn.getObject("Well", current_well)
#         ann = well.getAnnotation(Defaults['NS'])
#         return dict(ann.getValue())
#
#
#
# class ExpPaths:
#     def __init__(self, meta_data: MetaData):
#         self.meta_data = meta_data
#         self._create_dir_paths()
#         self._create_exp_dir()
#
#     def _create_dir_paths(self):
#         """ Generate path attributes for experiment"""
#         self.path = pathlib.Path(Defaults['DEFAULT_DEST_DIR']) / f"{self.meta_data.plate}"
#         self.flatfield_templates = self.path / Defaults['FLATFIELD_TEMPLATES']
#         self.final_data = self.path / Defaults['DATA']
#         self.cellcycle_summary_data = self.path / Defaults['DATA_CELLCYCLE_SUMMARY']
#
#         if Defaults['DEBUG']:
#             self.quality_ctr = self.path / Defaults['QUALITY_CONTROL']
#             self.example_img = self.path / Defaults['IMGS_CORR']
#
#
#     def _create_exp_dir(self):
#         path_list = [self.path, self.flatfield_templates, self.final_data]
#         for path in path_list:
#             path.mkdir(exist_ok=True)
#         if Defaults['DEBUG']:
#             extended_path_list = [self.quality_ctr, self.example_img]
#             for path in extended_path_list:
#                 path.mkdir(exist_ok=True)
#         message = f"Gathering data and assembling directories for experiment {self.meta_data.plate}"
#         self.separator = '='*len(message)
#         print(f"{self.separator}\n{message}")


if __name__ == "__main__":

    @omero_connect
    def systems_test(conn=None):
        instance = MetaData(conn, plate_id=1237)
        print(instance.channels)
        plate = conn.getObject("Plate", 1237)
        for well in plate.listChildren():
            print(instance.well_conditions(well.getId()))


    systems_test()
