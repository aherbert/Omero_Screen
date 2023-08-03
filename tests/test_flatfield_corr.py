
from omero_screen.database_links import MetaData, ProjectSetup
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.omero_functions import delete_map_annotations
from tests.conftest import attach_excel_file, delete_excel_attachments, delete_object
plate_id= 2

def test_flatfield_corr_create(omero_conn):
    """
    Tests the flatfield_corr function.
    """
    # load metadata excel file
    plate_obj = omero_conn.getObject("Plate", plate_id)
    attach_excel_file(plate_obj, './data/metadata.xlsx', omero_conn)
    metadata = MetaData(omero_conn, plate_id=plate_id)
    project_setup = ProjectSetup(plate_id, omero_conn)
    flatfield_dict = flatfieldcorr(metadata, project_setup, omero_conn)
    assert list(flatfield_dict.keys()) == ['DAPI', 'Tub', 'p21', 'EdU'], "Flatefield dictionary does not contain the correct keys"
    assert flatfield_dict['DAPI'].shape == (1080, 1080), "Flatefield dictionary does not contain the correct keys"
    delete_excel_attachments(plate_id, omero_conn)
    delete_map_annotations(plate_obj, omero_conn)
    for well in plate_obj.listChildren():
        delete_map_annotations(well, omero_conn)

def test_flatfield_corr_load(omero_conn):
    """
    Tests the flatfield_corr function.
    """
    # load metadata excel file
    plate_obj = omero_conn.getObject("Plate", plate_id)
    attach_excel_file(plate_obj, './data/metadata.xlsx', omero_conn)
    metadata = MetaData(omero_conn, plate_id=plate_id)
    project_setup = ProjectSetup(plate_id, omero_conn)
    flatfield_dict = flatfieldcorr(metadata, project_setup, omero_conn)
    assert list(flatfield_dict.keys()) == ['DAPI', 'Tub', 'p21', 'EdU'], "Flatefield dictionary does not contain the correct keys"
    assert flatfield_dict['DAPI'].shape == (1080, 1080), "Flatefield dictionary does not contain the correct keys"
    delete_excel_attachments(plate_id, omero_conn)
    delete_map_annotations(plate_obj, omero_conn)
    for well in plate_obj.listChildren():
        delete_map_annotations(well, omero_conn)
    delete_object(omero_conn, "Dataset", project_setup.dataset_id)
    delete_object(omero_conn, "Project", project_setup.project_id)

#TODO Check that example images are attached the the data set
