import pytest
from unittest.mock import Mock, patch
from omero_screen.metadata import MetaData, ProjectSetup

@pytest.fixture
def mock_conn():
    conn = Mock()
    conn.getObject.return_value = Mock()
    conn.getUser.return_value.getId.return_value = 1
    return conn

@pytest.fixture
def mock_plate():
    plate = Mock()
    plate.getName.return_value = "Test Plate"
    plate.listAnnotations.return_value = []
    plate.listChildren.return_value = [Mock(), Mock()]
    return plate

def test_metadata_init(mock_conn, mock_plate):
    mock_conn.getObject.return_value = mock_plate
    metadata = MetaData(mock_conn, 1)
    assert metadata.plate_id == 1
    assert metadata.plate_length == 2

@patch('omero_screen.metadata.add_map_annotations')
@patch('omero_screen.metadata.delete_map_annotations')
def test_set_well_inputs(mock_delete, mock_add, mock_conn, mock_plate):
    mock_conn.getObject.return_value = mock_plate
    metadata = MetaData(mock_conn, 1)
    metadata.well_inputs = Mock()
    metadata.well_inputs.astype.return_value.iterrows.return_value = [
        (0, {'Well': 'A1', 'Condition': 'Control'}),
        (1, {'Well': 'A2', 'Condition': 'Treatment'})
    ]
    metadata._set_well_inputs()
    assert mock_delete.call_count == 2
    assert mock_add.call_count == 2

def test_convert_well_names():
    assert MetaData.convert_well_names(Mock(row=0, column=0)) == 'A1'
    assert MetaData.convert_well_names(Mock(row=1, column=1)) == 'B2'

def test_project_setup_init(mock_conn):
    project_setup = ProjectSetup(1, mock_conn)
    assert project_setup.plate_id == 1
    assert project_setup.user_id == 1

@patch('omero_screen.metadata.create_object')
def test_create_dataset(mock_create, mock_conn):
    mock_conn.getObject.return_value = Mock(getName=lambda: 'Screens')
    mock_conn.getObjects.return_value = []
    mock_create.return_value = Mock(getId=lambda: 10)
    project_setup = ProjectSetup(1, mock_conn)
    assert project_setup.dataset_id == 10
    mock_create.assert_called_once()

# Add more tests as needed