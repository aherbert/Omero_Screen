
import pytest
from omero_screen import Defaults
from omero_screen.database_links import MetaData
from omero_screen.omero_functions import add_map_annotations, delete_map_annotations
from tests.conftest import attach_excel_file


def test_no_excel_no_map(omero_conn):
    with pytest.raises(ValueError):
        instance = MetaData(omero_conn, plate_id=2)


def test_map_no_excel(omero_conn):
    plate_obj = omero_conn.getObject("Plate", 2)  # or some other plate ID
    add_map_annotations(plate_obj, [["DAPI", "1"], ["Tub", "2"]], conn=omero_conn)
    instance = MetaData(omero_conn, plate_id=2)
    assert instance.channels == {'DAPI': 1, 'Tub': 2}


def test_excel_no_map(omero_conn):
    plate_obj = omero_conn.getObject("Plate", 2)  # or some other plate ID
    attach_excel_file(plate_obj, './data/metadata.xlsx', omero_conn)  # or some real Excel data
    instance = MetaData(omero_conn, plate_id=2)
    assert instance.channels == {'DAPI': 0, 'Tub': 1, 'p21': 2, 'EdU': 3}, "Channels from excelfile not as expected"


def test_excel_and_map(omero_conn):
    plate_obj = omero_conn.getObject("Plate", 2)  # or some other plate ID
    add_map_annotations(plate_obj, [["Drug Name", "Monastrol"], ["Concentration", "5 mg/ml"]], conn=omero_conn)
    attach_excel_file(plate_obj, './data/metadata.xlsx', omero_conn)
    instance = MetaData(omero_conn, plate_id=2)
    assert instance.channels == {'DAPI': 0, 'Tub': 1, 'p21': 2, 'EdU': 3}, "Channels from excelfile not as expected"

def test_found_cell_line_false(omero_conn):
    """Test if the absence of well metadata is spotted"""
    # add plate map annotation to avoid raising value error
    plate_obj = omero_conn.getObject("Plate", 2)  # or some other plate ID
    add_map_annotations(plate_obj, [["DAPI", "1"], ["Tub", "2"]], conn=omero_conn)
    instance = MetaData(omero_conn, plate_id=2)
    found_celline = instance._found_cell_line()
    assert found_celline == False, "Did not spot absence of well metadata"


def test_found_cell_line_true(omero_conn):
    """Test if the absence of well metadata is spotted"""

    plate_obj = omero_conn.getObject("Plate", 2)  # or some other plate ID
    add_map_annotations(plate_obj, [["DAPI", "1"], ["Tub", "2"]], conn=omero_conn)
    for well in plate_obj.listChildren():
        add_map_annotations(well, [["cell_line", "HeLa"]], omero_conn)
    instance = MetaData(omero_conn, plate_id=2)
    found_celline = instance._found_cell_line()
    for well in plate_obj.listChildren():
        delete_map_annotations(well, omero_conn)
    assert found_celline == True, "Did not spot all metadata"

def test_set_well_inputs_error(omero_conn):
    """Test if the absence of well metadata is spotted"""
    with pytest.raises(ValueError):
        instance = MetaData(omero_conn, plate_id=2)
        instance._set_well_inputs()

def test_set_well_inputs(omero_conn):
    """Test if the absence of well metadata is spotted"""
    plate_obj = omero_conn.getObject("Plate", 2)  # or some other plate ID
    attach_excel_file(plate_obj, './data/metadata.xlsx', omero_conn)
    instance = MetaData(omero_conn, plate_id=2)
    for well in plate_obj.listChildren():
        ann = well.getAnnotation(Defaults['NS'])
        assert ann.getValue()[0] == ('cell_line', 'RPE-1'),  "Did not set well inputs"
    delete_map_annotations(plate_obj, omero_conn)
    for well in plate_obj.listChildren():
        delete_map_annotations(well, omero_conn)

def test_set_well_inputs_overwrite(omero_conn):
    """Test if the absence of well metadata is spotted"""
    plate_obj = omero_conn.getObject("Plate", 2)  # or some other plate ID
    for well in plate_obj.listChildren():
        add_map_annotations(well, [["cell_line", "HeLa"]], omero_conn)
    attach_excel_file(plate_obj, './data/metadata.xlsx', omero_conn)
    instance = MetaData(omero_conn, plate_id=2)
    for well in plate_obj.listChildren():
        ann = well.getAnnotation(Defaults['NS'])
        assert all('HeLa' not in sublist for sublist in ann.getValue())
        assert ann.getValue()[0] == ('cell_line', 'RPE-1'),  "Did not set well inputs"
    delete_map_annotations(plate_obj, omero_conn)
    for well in plate_obj.listChildren():
        delete_map_annotations(well, omero_conn)

def test_metadata_new(omero_conn):
    """Test if the absence of well metadata is spotted"""
    plate_obj = omero_conn.getObject("Plate", 2)  # or some other plate ID  # or some other plate ID
    attach_excel_file(plate_obj, './data/metadata.xlsx', omero_conn)
    instance = MetaData(omero_conn, plate_id=2)
    assert instance.channels == {'DAPI': 0, 'Tub': 1, 'p21': 2, 'EdU': 3}
    assert instance.well_conditions(49) == {'cell_line': 'RPE-1', 'condition': 'SCR'}
    assert instance.well_conditions(50) == {'cell_line': 'RPE-1', 'condition': 'CDK4'}
    delete_map_annotations(plate_obj, omero_conn)
    for well in plate_obj.listChildren():
        delete_map_annotations(well, omero_conn)

def test_metadata_old(omero_conn):
    """Test if the absence of well metadata is spotted"""
    plate_obj = omero_conn.getObject("Plate", 2)  # or some other plate ID  # or some other plate ID
    add_map_annotations(plate_obj, [["DAPI", "1"], ["Tub", "2"]], conn=omero_conn)
    for well in plate_obj.listChildren():
        add_map_annotations(well, [["cell_line", "HeLa"]], omero_conn)
    instance = MetaData(omero_conn, plate_id=2)
    assert instance.channels == {'DAPI': 1, 'Tub': 2}
    assert instance.well_conditions(49) == {'cell_line': 'HeLa'}
    assert instance.well_conditions(50) == {'cell_line': 'HeLa'}
    delete_map_annotations(plate_obj, omero_conn)
    for well in plate_obj.listChildren():
        delete_map_annotations(well, omero_conn)