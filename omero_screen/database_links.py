#!/usr/bin/env python
"""Module to link to the omero db, extract metadata and link a project/dataset to the plate."""
import tempfile

import omero
import pandas as pd
from omero.gateway import DatasetWrapper
from omero.rtypes import rstring

from omero_screen import Defaults
from omero_screen.general_functions import add_map_annotation


class MetaData:
    """Class to add the metadata to the plate."""

    def __init__(self, conn, plate_id=None):
        self.conn = conn
        self.plate_id = plate_id
        self.plate_obj = self.conn.getObject("Plate", self.plate_id)
        self.plate_length = len(list(self.plate_obj.listChildren()))
        self.channels, self.well_inputs = self._get_metadata()

    def _get_metadata(self):
        """
        Get the metadata from the Excel file attached to the plate.
        :return: a dictionary with the channel data and a pandas DataFrame
        with the well data if the Excel file is found. Otherwise return none
        """
        file_anns = self.plate_obj.listAnnotations()

        for ann in file_anns:
            if isinstance(ann, omero.gateway.FileAnnotationWrapper) and ann.getFile().getName().endswith(
                    'metadata.xlsx'):
                return self._get_channel_data_from_excel(ann)

        return self._get_channel_data_from_map()

    def _get_channel_data_from_map(self):
        if ann := self.plate_obj.getAnnotation(Defaults['NS']):
            map_data = dict(ann.getValue())
            if 'DAPI' in map_data or 'Hoechst' in map_data:
                print("Found map annotation with 'DAPI' or 'Hoechst'")
                return self._get_channel_data(ann), None
        raise ValueError("No map annotation available and Excel file not found.")

    def _get_channel_data_from_excel(self, ann):
        self._clear_map_annotation()
        print("Found Excel file:", ann.getFile().getName())
        original_file = ann.getFile()
        with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp:
            self._download_file_to_tmp(original_file, tmp)
            data = pd.read_excel(tmp.name, sheet_name=None)
        key_value_data = data['Sheet1'].astype(str).values.tolist()
        add_map_annotation(self.plate_obj, key_value_data, conn=self.conn)
        channel_data = {row[0]: row[1] for row in key_value_data}
        return channel_data, data['Sheet2']

    def _clear_map_annotation(self):
        if map_ann := self.plate_obj.getAnnotation(Defaults['NS']):
            map_ann.setValue([])  # Set a new empty list
            map_ann.save()

    def _download_file_to_tmp(self, original_file, tmp):
        with open(tmp.name, 'wb') as f:
            for chunk in original_file.asFileObj():
                f.write(chunk)

    def _get_channel_data(self, ann):
        """"""
        channels = dict(ann.getValue())
        if 'Hoechst' in channels:
            channels['DAPI'] = channels.pop('Hoechst')
        # changing channel number to integer type
        for key in channels:
            channels[key] = int(channels[key])
        return channels


class ProjectSetup:
    """Class to set up the Omero-Screen project and organise the metadata"""

    def __init__(self, conn, plate_id):
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
