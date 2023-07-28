#!/usr/bin/env python
"""Module to link to the omero db, extract metadata and link a project/dataset to the plate."""

import pandas as pd
import pathlib
from omero_screen import Defaults
import omero
from omero.gateway import DatasetWrapper
from omero.rtypes import rstring








def create_project(conn=None, project_name='Screens'):
    """Check for a project, if not present, create it."""

    # Fetch all projects
    projects = list(conn.getObjects("Project"))
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
        update_service = conn.getUpdateService()
        saved_project = update_service.saveAndReturnObject(project)
        project_id = saved_project.getId().getValue()

    print('Project ID:', project_id)

    return project_id


def create_dataset(dataset_name, conn=None):
    """Create a new dataset."""

    # Fetch all datasets
    datasets = conn.getObjects("Dataset")

    # Check if dataset exists
    for dataset in datasets:
        if dataset.getName() == dataset_name:
            print("Dataset exists, Id:", dataset.getId())
            return dataset.getId()
    # If code reaches here, dataset doesn't exist. Create a new dataset
    print("Dataset does not exist. Creating now...")
    new_dataset = DatasetWrapper(conn, omero.model.DatasetI())
    new_dataset.setName(dataset_name)
    new_dataset.save()
    print("New dataset, Id:", new_dataset.getId())
    return new_dataset.getId()


def link_project_dataset(conn=None, project_id=None, dataset_id=None):
    """Link a project and a dataset."""
    # Fetch the project
    project = conn.getObject("Project", project_id)
    if not project:
        print("Project not found")
        return

    # Iterate over linked datasets to check if our dataset is already linked
    for dataset in project.listChildren():
        if dataset.getId() == dataset_id:
            print("Link already exists")
            return

    # If we reach here, it means the dataset is not linked to the project. So, create a new link.
    link = omero.model.ProjectDatasetLinkI()
    link.setChild(omero.model.DatasetI(dataset_id, False))
    link.setParent(omero.model.ProjectI(project_id, False))
    conn.getUpdateService().saveObject(link)
    print("Link created")


def create_project_dataset(dataset_name, conn=None):
    project_id = create_project(conn)
    dataset_obj = create_dataset(dataset_name, conn)
    link_project_dataset(conn, project_id, dataset_obj)
    return dataset_obj.getId()


class MetaData:
    """ Extract Plate Name from Omero.
     Class Attributes, Various Plate MetaData (Plate_ID,
     Class Methods: Extract Well specific information
     """

    def __init__(self, plate_id, conn):
        self.conn = conn
        self.plate_obj = self.conn.getObject("Plate", plate_id)
        self.plate = self.plate_obj.getName()
        self._extract_meta_data()
        self.data_set = self._create_dataset(plate_id)
        self.df_final = pd.DataFrame()
        self.df_quality = pd.DataFrame()

    def _extract_meta_data(self):
        """ Extract metadata from Omero Plate object
        Attributes:
            plate_id (int),
            channels (dict),
            plate_layout (df),
            segmentation models (paths)
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

    def _create_dataset(self, plate_id):
        project_id = create_project(conn=self.conn)
        dataset_id = create_dataset(f"Flatfieldcorr_{plate_id}", conn=self.conn)
        link_project_dataset(conn=self.conn, project_id=project_id, dataset_id=dataset_id)
        return dataset_id


class ExpPaths:
    def __init__(self, meta_data: MetaData):
        self.meta_data = meta_data
        self._create_dir_paths()
        self._create_exp_dir()

    def _create_dir_paths(self):
        """ Generate path attributes for experiment"""
        self.path = pathlib.Path(Defaults['DEFAULT_DEST_DIR']) / f"{self.meta_data.plate}"
        self.flatfield_templates = self.path / Defaults['FLATFIELD_TEMPLATES']
        self.final_data = self.path / Defaults['DATA']
        self.cellcycle_summary_data = self.path / Defaults['DATA_CELLCYCLE_SUMMARY']

        if Defaults['DEBUG']:
            self.quality_ctr = self.path / Defaults['QUALITY_CONTROL']
            self.example_img = self.path / Defaults['IMGS_CORR']


    def _create_exp_dir(self):
        path_list = [self.path, self.flatfield_templates, self.final_data]
        for path in path_list:
            path.mkdir(exist_ok=True)
        if Defaults['DEBUG']:
            extended_path_list = [self.quality_ctr, self.example_img]
            for path in extended_path_list:
                path.mkdir(exist_ok=True)
        message = f"Gathering data and assembling directories for experiment {self.meta_data.plate}"
        self.separator = '='*len(message)
        print(f"{self.separator}\n{message}")


