import pytest
import omero
from omero.gateway import BlitzGateway
from omero_screen.metadata import (
    MetaData,
    create_project,
    create_dataset,
    link_project_dataset,
)

project_name = "TestProject"
dataset_name = "Dataforplate_1"


@pytest.fixture(scope="module")
def conn():
    HOST = "localhost"
    PORT = 4064
    USERNAME = "root"
    PASSWORD = "omero"

    conn = BlitzGateway(USERNAME, PASSWORD, host=HOST, port=PORT)
    conn.connect()

    yield conn

    print("Finalizing, cleaning up...")
    # Delete the specific project
    projects = list(conn.getObjects("Project"))
    project_names = [p.getName() for p in projects]
    if project_name in project_names:
        project_id = projects[project_names.index(project_name)].getId()
        delete = omero.cmd.Delete2(targetObjects={"Project": [project_id]})
        conn.c.sf.submit(delete)
    datasets = list(conn.getObjects("Dataset"))
    dataset_names = [d.getName() for d in datasets]
    if dataset_name in dataset_names:
        dataset_id = datasets[dataset_names.index(dataset_name)].getId()
        delete = omero.cmd.Delete2(targetObjects={"Dataset": [dataset_id]})
        conn.c.sf.submit(delete)


def test_create_project_when_none_exists(conn):
    # Assert project does not exist initially
    projects = list(conn.getObjects("Project"))
    project_names = [p.getName() for p in projects]
    assert project_name not in project_names

    # Check and create project
    create_project(conn, project_name)

    # Fetch all projects again
    projects = list(conn.getObjects("Project"))
    project_names = [p.getName() for p in projects]

    # Assert project now exists
    assert project_name in project_names


def test_create_project_when_exists(conn):
    # Assert project already exists
    projects = list(conn.getObjects("Project"))
    project_names = [p.getName() for p in projects]
    assert project_name in project_names

    # Check and create project, it should not create a new project
    project_id = create_project(conn, project_name)

    # The ID should be the same as the old one
    projects = list(conn.getObjects("Project"))
    project_ids = [p.getId() for p in projects if p.getName() == project_name]
    assert len(project_ids) == 1
    assert project_ids[0] == project_id


def test_create_dataset(conn):
    dataset_name = "TestDataset"

    # Assert dataset does not exist initially
    datasets = list(conn.getObjects("Dataset"))
    dataset_names = [d.getName() for d in datasets]
    # assert dataset_name not in dataset_names

    # Create dataset
    dataset_id = create_dataset(dataset_name, conn)

    # Assert dataset ID is not None
    assert dataset_id is not None

    # Fetch all datasets again
    datasets = list(conn.getObjects("Dataset"))
    dataset_names = [d.getName() for d in datasets]

    # Assert dataset now exists
    assert dataset_name in dataset_names


def test_link_project_dataset(conn):
    project_name = "TestProject"
    dataset_name = "TestDataset"

    # Get project and dataset
    project = conn.getObject("Project", attributes={"name": project_name})
    dataset = conn.getObject("Dataset", attributes={"name": dataset_name})

    # Assert they are not linked initially
    assert dataset not in project.listChildren()

    # Link project and dataset
    link_project_dataset(conn, project.getId(), dataset.getId())

    # Assert they are linked now
    project = conn.getObject("Project", attributes={"name": project_name})
    assert dataset in project.listChildren()


def test_metadata(conn):
    # Create the MetaData object
    metadata = MetaData(1, conn)

    # Assert the extracted data
    assert metadata.channels == {"DAPI": 0, "Tub": 1, "Alexa555": 2, "EdU": 3}
    assert metadata.plate_length > 0  # Expecting some positive number
    assert metadata.well_conditions(43) == {
        "cell_line": "RPE-1",
        "condition": "NT 1000",
    }

    # Assert the dataset has been created
    assert isinstance(metadata.data_set, int)
