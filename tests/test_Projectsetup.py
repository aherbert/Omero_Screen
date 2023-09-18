import pytest
from omero.gateway import BlitzGateway
from omero.rtypes import rstring
from omero.gateway import DatasetWrapper
from omero_screen.metadata import MetaData, ProjectSetup
from omero_screen.omero_functions import create_object
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


def test_project_setup_noproject(omero_conn):
    user_id = omero_conn.getUser().getId()
    projects = list(
        omero_conn.getObjects(
            "Project",
            opts={"owner": user_id},
            attributes={"name": "Screens"},
        )
    )
    for project in projects:
        delete_object(omero_conn, "Project", project.getId())

    plate_id = 2
    setup = ProjectSetup(plate_id, omero_conn)
    project = omero_conn.getObject("Project", setup.project_id)
    dataset = omero_conn.getObject("Dataset", setup.dataset_id)
    assert project.getName() == "Screens"
    assert dataset.getName() == str(plate_id)


def test_project_setup_twoprojects(omero_conn):
    user_id = omero_conn.getUser().getId()
    projects = list(
        omero_conn.getObjects(
            "Project",
            opts={"owner": user_id},
            attributes={"name": "Screens"},
        )
    )
    for project in projects:
        delete_object(omero_conn, "Project", project.getId())
    Screens = create_object(omero_conn, "Project", "Screens")
    Screens_1 = create_object(omero_conn, "Project", "Screens")
    plate_id = 2
    with pytest.raises(Exception) as excinfo:
        setup = ProjectSetup(plate_id, omero_conn)

        # Validate the exception message
    assert (
        f"Data integrity issue: Multiple projects found with the same name 'Screens' for user ID {user_id}"
        in str(excinfo.value)
    )
    projects = list(
        omero_conn.getObjects(
            "Project",
            opts={"owner": user_id},
            attributes={"name": "Screens"},
        )
    )
    for project in projects:
        delete_object(omero_conn, "Project", project.getId())


def test_project_setup_oneproject(omero_conn):
    user_id = omero_conn.getUser().getId()
    projects = list(
        omero_conn.getObjects(
            "Project",
            opts={"owner": user_id},
            attributes={"name": "Screens"},
        )
    )
    for project in projects:
        delete_object(omero_conn, "Project", project.getId())
    Screens = create_object(omero_conn, "Project", "Screens")
    plate_id = 2
    setup = ProjectSetup(plate_id, omero_conn)
    project = omero_conn.getObject("Project", setup.project_id)
    dataset = omero_conn.getObject("Dataset", setup.dataset_id)
    assert project.getName() == "Screens"
    assert dataset.getName() == str(plate_id)
