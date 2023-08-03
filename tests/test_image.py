
from omero_screen.database_links import MetaData, ProjectSetup
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.image_analysis import Image
from omero_screen.omero_functions import delete_map_annotations
from tests.conftest import attach_excel_file, delete_excel_attachments, delete_object
plate_id= 2



def test_image_create(omero_conn):
    """Test the Image Class when no data are present on the server"""
    # load metadata excel file
    plate_obj = omero_conn.getObject("Plate", plate_id)
    well = omero_conn.getObject("Well", 49)
    omero_image = well.getImage(0)
    attach_excel_file(plate_obj, './data/metadata.xlsx', omero_conn)
    metadata = MetaData(omero_conn, plate_id=plate_id)
    project_setup = ProjectSetup(plate_id, omero_conn)
    flatfield_dict = flatfieldcorr(metadata, project_setup, omero_conn)
    image = Image(omero_conn, well, omero_image, metadata, project_setup, flatfield_dict)
    assert image.n_mask.max() == 222, "Nuclear Segmentation is incorrect"
    assert image.c_mask.max() == 220, "Nuclear Segmentation is incorrect"
    assert image.img_dict['DAPI'].shape == (1080,1080), "problem with image_dict"


def test_image_load(omero_conn):
    """Test the Image Class when data are present on the server"""
    # load metadata excel file
    plate_obj = omero_conn.getObject("Plate", plate_id)
    well = omero_conn.getObject("Well", 49)
    omero_image = well.getImage(0)
    attach_excel_file(plate_obj, './data/metadata.xlsx', omero_conn)
    metadata = MetaData(omero_conn, plate_id=plate_id)
    project_setup = ProjectSetup(plate_id, omero_conn)
    flatfield_dict = flatfieldcorr(metadata, project_setup, omero_conn)
    image = Image(omero_conn, well, omero_image, metadata, project_setup, flatfield_dict)
    assert image.n_mask.max() == 222, "Nuclear Segmentation is incorrect"
    assert image.c_mask.max() == 220, "Nuclear Segmentation is incorrect"
    assert image.img_dict['DAPI'].shape == (1080, 1080), "Mask is not binary"
    delete_excel_attachments(plate_id, omero_conn)
    delete_map_annotations(plate_obj, omero_conn)
    for well in plate_obj.listChildren():
        delete_map_annotations(well, omero_conn)
    delete_object(omero_conn, "Dataset", project_setup.dataset_id)
    delete_object(omero_conn, "Project", project_setup.project_id)




