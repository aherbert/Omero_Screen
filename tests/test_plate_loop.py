from omero_screen.loops import plate_loop
from omero_screen.omero_functions import delete_map_annotations
from tests.conftest import attach_excel_file, delete_excel_attachments, delete_object
from conftest import PLATE_ID as plate_id


def test_plate_loop(omero_conn):
    """Test the Image Class when no data are present on the server"""
    # load metadata excel file
    plate_obj = omero_conn.getObject("Plate", plate_id)
    attach_excel_file(plate_obj, "./data/metadata.xlsx", omero_conn)
    df_plate, df_quality = plate_loop(plate_id, omero_conn)
    print(df_plate.head())
