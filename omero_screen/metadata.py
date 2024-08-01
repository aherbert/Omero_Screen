#!/usr/bin/env python
"""Module to link to the omero db, extract metadata and link a project/dataset to the plate."""
import tempfile
import logging
from typing import Dict, List, Optional, Tuple

import omero
import pandas as pd
from omero.gateway import BlitzGateway, FileAnnotationWrapper, MapAnnotationWrapper

from omero_screen import Defaults
from omero_screen.omero_functions import (
    add_map_annotations,
    delete_map_annotations,
    create_object,
)

logger = logging.getLogger("omero-screen")

class MetaData:
    """Class to add and manage metadata for a plate."""

    def __init__(self, conn: BlitzGateway, plate_id: int):
        self.conn = conn
        self.plate_id = plate_id
        self.plate_obj = self.conn.getObject("Plate", self.plate_id)
        logger.debug(f"Plate object: {self.plate_obj}")
        self.plate_length = len(list(self.plate_obj.listChildren()))
        self.channels, self.well_inputs = self._get_metadata()
        self._set_well_inputs()

    def _get_metadata(self) -> Tuple[Dict[str, int], Optional[pd.DataFrame]]:
        """
        Get the metadata from the Excel file attached to the plate.

        Returns:
            A tuple containing:
            - A dictionary with the channel data
            - A pandas DataFrame with the well data if the Excel file is found, otherwise None
        """
        file_anns = self.plate_obj.listAnnotations()
        message = f"Loading metadata for {self.plate_obj.getName()}"
        self.separator = "=" * len(message)
        print(f"{self.separator}\n{message}")
        
        for ann in file_anns:
            if isinstance(ann, FileAnnotationWrapper) and ann.getFile().getName().endswith(".xlsx"):
                return self._get_channel_data_from_excel(ann)

        return self._get_channel_data_from_map()

    def _get_channel_data_from_map(self) -> Tuple[Dict[str, int], None]:
        """Extract channel data from map annotations."""
        annotations = self.plate_obj.listAnnotations()
        map_annotations = [
            ann for ann in annotations
            if isinstance(ann, omero.gateway.MapAnnotationWrapper)
        ]

        for map_ann in map_annotations:
            map_data = dict(map_ann.getValue())
            if "DAPI" in map_data or "Hoechst" in map_data:
                print("Found map annotation with 'DAPI' or 'Hoechst'")
                key_value_data = map_ann.getValue()
                return self._get_channel_data(key_value_data), None

        raise ValueError("No map annotation available and Excel file not found.")

    def _get_channel_data_from_excel(self, ann: FileAnnotationWrapper) -> Tuple[Dict[str, int], pd.DataFrame]:
        """Extract channel data from attached Excel file."""
        self._clear_map_annotation()
        print(f"Found Excel file: {ann.getFile().getName()}")
        original_file = ann.getFile()
        
        with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp:
            self._download_file_to_tmp(original_file, tmp)
            data = pd.read_excel(tmp.name, sheet_name=None)
        
        key_value_data = data["Sheet1"].astype(str).values.tolist()
        add_map_annotations(self.plate_obj, key_value_data, conn=self.conn)
        channel_data = {row[0]: row[1] for row in key_value_data}
        well_data = data["Sheet2"]
        return self._get_channel_data(channel_data), well_data

    def _clear_map_annotation(self) -> None:
        """Clear existing map annotations."""
        if map_ann := self.plate_obj.getAnnotation(Defaults["NS"]):
            map_ann.setValue([])
            map_ann.save()

    def _download_file_to_tmp(self, original_file, tmp) -> None:
        """Download file to temporary location."""
        with open(tmp.name, "wb") as f:
            for chunk in original_file.asFileObj():
                f.write(chunk)

    def _get_channel_data(self, key_value_data: List[List[str]]) -> Dict[str, int]:
        """Process and clean channel data."""
        channels = dict(key_value_data)
        cleaned_channels = {key.strip(): value for key, value in channels.items()}
        if "Hoechst" in cleaned_channels:
            cleaned_channels["DAPI"] = cleaned_channels.pop("Hoechst")
        
        # Convert channel numbers to integer type
        cleaned_channels = {key: int(value) for key, value in cleaned_channels.items()}
        logger.info(f"Channels: {cleaned_channels}")
        return cleaned_channels

    def _set_well_inputs(self) -> None:
        """Process and set well metadata."""
        if self.well_inputs is None:
            if not self._found_cell_line():
                raise ValueError("Well metadata are not present")
        else:
            df = self.well_inputs.astype(str)
            df_dict = {
                row["Well"]: [[col, row[col]] for col in df.columns if col != "Well"]
                for _, row in df.iterrows()
            }
            for well in self.plate_obj.listChildren():
                delete_map_annotations(well, conn=self.conn)
                wellname = self.convert_well_names(well)
                if wellname in df_dict:
                    add_map_annotations(well, df_dict[wellname], conn=self.conn)

    @staticmethod
    def convert_well_names(well) -> str:
        """Convert well coordinates to well names."""
        row_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return f"{row_letters[well.row]}{well.column + 1}"

    def _found_cell_line(self) -> bool:
        """Check if all wells in the plate contain a 'cell_line' annotation."""
        plate = self.conn.getObject("Plate", self.plate_id)
        if plate is None:
            print(f"Cannot find plate: {self.plate_id}")
            return False

        wells_without_cell_line = [
            well.id for well in plate.listChildren()
            if not any(
                "cell_line" in dict(ann.getValue()) or "Cell_Line" in dict(ann.getValue())
                for ann in well.listAnnotations()
                if isinstance(ann, MapAnnotationWrapper)
            )
        ]

        if wells_without_cell_line:
            print(f"Found {len(wells_without_cell_line)} wells without a 'cell_line' annotation")
        else:
            print("All wells have a 'cell_line' annotation")

        return not wells_without_cell_line

    def well_conditions(self, current_well: int) -> Dict[str, str]:
        """Get the well conditions from the well metadata."""
        well = self.conn.getObject("Well", current_well)
        ann = well.getAnnotation(Defaults["NS"])
        return dict(ann.getValue())


class ProjectSetup:
    """Class to set up the Omero-Screen project and organize the metadata."""

    def __init__(self, plate_id: int, conn: BlitzGateway):
        self.conn = conn
        self.user_id = self.conn.getUser().getId()
        self.plate_id = plate_id
        self.dataset_id = self._create_dataset()

    def _create_dataset(self) -> int:
        """Create a new dataset or return the ID of an existing one."""
        dataset_name = str(self.plate_id)
        try:
            project = self.conn.getObject("Project", Defaults["PROJECT_ID"])
            assert project.getName() == 'Screens', "Project name does not match 'Screens'"
        except Exception as e:
            raise ValueError(f"Project for Screen with ID {Defaults['PROJECT_ID']} not found") from e

        datasets = list(
            self.conn.getObjects(
                "Dataset",
                opts={"project": project.getId()},
                attributes={"name": dataset_name},
            )
        )

        if len(datasets) > 1:
            raise ValueError(
                f"Data integrity issue: Multiple datasets found with the same name '{dataset_name}' for user ID {self.user_id}"
            )
        elif len(datasets) == 1:
            dataset_id = datasets[0].getId()
            print(f"Dataset exists with ID: {dataset_id}")
            return dataset_id
        else:
            new_dataset = create_object(self.conn, "Dataset", self.plate_id)
            new_dataset_id = new_dataset.getId()
            link = omero.model.ProjectDatasetLinkI()
            link.setChild(omero.model.DatasetI(new_dataset_id, False))
            link.setParent(omero.model.ProjectI(Defaults["PROJECT_ID"], False))
            self.conn.getUpdateService().saveObject(link)
            print(f"Dataset {new_dataset.getName()} created and linked to Screens project")
            return new_dataset_id
