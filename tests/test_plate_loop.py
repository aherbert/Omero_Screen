from omero_screen.loops import plate_loop
import omero
from omero_screen.omero_functions import delete_map_annotations
from tests.conftest import attach_excel_file, delete_excel_attachments, delete_object
from conftest import PLATE_ID as plate_id


def test_plate_loop_df(omero_conn):
    """Test the Image Class when no data are present on the server"""
    # load metadata excel file
    plate_obj = omero_conn.getObject("Plate", plate_id)
    attach_excel_file(plate_obj, "./data/metadata.xlsx", omero_conn)
    df_plate, df_quality = plate_loop(plate_id, omero_conn)
    assert len(df_plate) == 976
    assert df_plate["area_nucleus"].sum() == 121502


def test_plate_loop_wellimg(omero_conn):
    """Test the Image Class when no data are present on the server"""
    # load metadata excel file
    plate_obj = omero_conn.getObject("Plate", plate_id)
    well_list = list(plate_obj.listChildren())
    for well in well_list:
        file_names = [ann.getFile().getName() for ann in well.listAnnotations()]

        # Assert that there is a PNG file among the file annotations
        assert any(
            name.endswith(".png") for name in file_names
        ), f"Missing PNG file for well {well.getId()}"
