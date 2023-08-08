import pytest
from omero.gateway import BlitzGateway
from omero.rtypes import rstring
from omero.gateway import DatasetWrapper
from omero_screen.metadata import MetaData, ProjectSetup
from tests.conftest import delete_object


def test_connection(omero_conn):
    plate = omero_conn.getObject("Plate", 2)
    print(plate.id)
    assert plate.getName() == "test_plate01"


def test_add_metadata(omero_conn):
    metadata = MetaData(omero_conn, 2)
    assert len(metadata.channels) == 4
    assert len(metadata.well_inputs) == 2


def test_project_setup(omero_conn):
    # We'll use a random plate_id for this test
    plate_id = 2

    # Create an instance of ProjectSetup
    setup = ProjectSetup(omero_conn, plate_id)
    project = omero_conn.getObject("Project", setup.project_id)
    dataset = omero_conn.getObject("Dataset", setup.dataset_id)
    assert project.getName() == "Screens"
    assert dataset.getName() == str(plate_id)

    # Check if the dataset is linked to the project
    linked_datasets = [d.getId() for d in project.listChildren()]
    assert setup.dataset_id in linked_datasets, "Dataset is not linked to the project"
    delete_object(omero_conn, "Dataset", setup.dataset_id)
    delete_object(omero_conn, "Project", setup.project_id)
