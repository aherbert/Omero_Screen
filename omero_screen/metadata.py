#!/usr/bin/env python
"""Module to link to the omero db, extract metadata and link a project/dataset to the plate."""
import tempfile
from omero_screen.general_functions import omero_connect
import omero
import pandas as pd
from omero.gateway import DatasetWrapper, FileAnnotationWrapper, MapAnnotationWrapper
from omero.rtypes import rstring
from omero_screen import Defaults
from omero_screen.omero_functions import (
    add_map_annotations,
    delete_map_annotations,
    create_object,
)


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
        message = f"Loading metadata for {self.plate_obj.getName()}"
        self.separator = "=" * len(message)
        print(f"{self.separator}\n{message}")
        for ann in file_anns:
            if isinstance(
                ann, FileAnnotationWrapper
            ) and ann.getFile().getName().endswith(".xlsx"):
                return self._get_channel_data_from_excel(ann)

        return self._get_channel_data_from_map()

    def _get_channel_data_from_map(self):
        annotations = self.plate_obj.listAnnotations()
        map_annotations = [
            ann
            for ann in annotations
            if isinstance(ann, omero.gateway.MapAnnotationWrapper)
        ]

        for map_ann in map_annotations:
            map_data = dict(map_ann.getValue())
            if "DAPI" in map_data or "Hoechst" in map_data:
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
        key_value_data = data["Sheet1"].astype(str).values.tolist()
        add_map_annotations(self.plate_obj, key_value_data, conn=self.conn)
        channel_data = {row[0]: row[1] for row in key_value_data}
        well_data = data["Sheet2"]
        return self._get_channel_data(channel_data), well_data

    def _clear_map_annotation(self):
        if map_ann := self.plate_obj.getAnnotation(Defaults["NS"]):
            map_ann.setValue([])  # Set a new empty list
            map_ann.save()

    def _download_file_to_tmp(self, original_file, tmp):
        with open(tmp.name, "wb") as f:
            for chunk in original_file.asFileObj():
                f.write(chunk)

    def _get_channel_data(self, key_value_data):
        """"""
        channels = dict(key_value_data)
        cleaned_channels = {key.strip(): value for key, value in channels.items()}
        if "Hoechst" in cleaned_channels:
            cleaned_channels["DAPI"] = cleaned_channels.pop("Hoechst")
        # changing channel number to integer type
        for key in cleaned_channels:
            cleaned_channels[key] = int(cleaned_channels[key])
        return cleaned_channels

    def _set_well_inputs(self):
        """Function to deal with the well metadata"""
        # if there are no well input data check if there are metadata already present
        if self.well_inputs is None:
            if not self._found_cell_line():
                raise ValueError("Well metadata are not present")
        else:
            df = self.well_inputs
            df_dict = {
                row["Well"]: [[col, row[col]] for col in df.columns if col != "Well"]
                for _, row in df.iterrows()
            }
            for well in self.plate_obj.listChildren():
                # overwrite map annotation if present
                delete_map_annotations(well, conn=self.conn)
                wellname = self.convert_well_names(well)
                for key in df_dict:
                    if wellname == key:
                        add_map_annotations(well, df_dict[key], conn=self.conn)

    def convert_well_names(self, well):
        row_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # assuming no more than 26 rows
        row_number = well.row
        column_number = well.column + 1
        return f"{row_letters[row_number]}{column_number}"

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
            annotations = [
                ann
                for ann in well.listAnnotations()
                if isinstance(ann, MapAnnotationWrapper)
            ]
            found_cell_line = any(
                "cell_line" in dict(ann.getValue()) for ann in annotations
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
        ann = well.getAnnotation(Defaults["NS"])
        return dict(ann.getValue())

class ProjectSetup:
    """Class to set up the Omero-Screen project and organise the metadata"""

    def __init__(self, plate_id, conn):
        self.conn = conn
        self.user_id = self.conn.getUser().getId()
        self.plate_id = plate_id
        self.dataset_id = self._create_dataset()
        self._link_project_dataset()

    def _create_dataset(self):  # sourcery skip: raise-specific-error
        """Create a new dataset."""
        dataset_name = str(self.plate_id)
        try:
            project = self.conn.getObject("Project", Defaults["PROJECT_ID"])
        except Exception as e:
            raise Exception(
                f"Project for Screen with ID {Defaults['PROJECT_ID']} not found"
            ) from e
        # Fetch all datasets
        datasets = list(
            self.conn.getObjects(
                "Dataset",
                opts={"project": project.getId()},
                attributes={"name": dataset_name},
            )
        )
        if len(datasets) > 1:
            raise Exception(
                f"Data integrity issue: Multiple datasets found with the same name '{dataset_name}' for user ID {self.user_id}"
            )

        elif len(datasets) == 1:
            data_set_id = datasets[0].getId()  # The only match
            print(f"Dataset exists with ID: {data_set_id}")
            return datasets[0].getId()

        else:
            new_dataset = create_object(self.conn, "Dataset", self.plate_id)
            new_dataset_id = new_dataset.getId()
            print(f"Dataset created with ID: {new_dataset_id}")
            return new_dataset_id

    def _link_project_dataset(self):
        """Link a project and a dataset."""
        # If we reach here, it means the dataset is not linked to the project. So, create a new link.
        link = omero.model.ProjectDatasetLinkI()
        link.setChild(omero.model.DatasetI(self.dataset_id, False))
        link.setParent(omero.model.ProjectI(Defaults["PROJECT_ID"], False))
        self.conn.getUpdateService().saveObject(link)
        print("Link created")
