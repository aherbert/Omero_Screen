from omero_screen.omero_functions import (
    load_exceldata,
    delete_excel_attachments,
    add_map_annotation,
    delete_map_annotations,
)
from omero.gateway import FileAnnotationWrapper, MapAnnotationWrapper


def test_load_data_valid_plate_and_file(omero_conn):
    """
    Tests the load_exceldata function with a valid plate id and existing file.
    """
    plate_id = 2  # use actual plate id from your test server
    file_path = "./data/metadata.xlsx"  # ensure this file exists

    load_exceldata(plate_id, file_path, conn=omero_conn)
    # get the plate object to check if file annotation was linked
    plate = omero_conn.getObject("Plate", plate_id)
    # get the file annotations of the plate
    file_annotations = [
        ann for ann in plate.listAnnotations() if isinstance(ann, FileAnnotationWrapper)
    ]
    # assert that a file annotation was added
    assert file_annotations, "File annotation was not added to the plate."


def test_delete_excel_attachments(omero_conn):
    """
    Tests the delete_excel_attachments function.
    """
    plate_id = 2

    delete_excel_attachments(plate_id, conn=omero_conn)

    # Get the Plate object
    plate = omero_conn.getObject("Plate", plate_id)

    # Get all file annotations
    file_annotations = [
        ann for ann in plate.listAnnotations() if isinstance(ann, FileAnnotationWrapper)
    ]

    # Filter Excel file annotations
    excel_annotations = [
        ann
        for ann in file_annotations
        if ann.getFile().getMimetype()
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ]

    # Assert that no Excel file annotations exist
    assert not excel_annotations


def test_add_map_annotation(omero_conn):
    """
    Tests the add_map_annotation function.
    """
    plate_id = 2  # use actual plate id from your test server
    key_value = [("test_annotation", "1")]  # the annotation to add

    # Get the Plate object
    plate = omero_conn.getObject("Plate", plate_id)

    # Call the function to add the map annotation
    add_map_annotation(plate, key_value, conn=omero_conn)

    # Get all map annotations of the plate
    map_annotations = [
        ann for ann in plate.listAnnotations() if isinstance(ann, MapAnnotationWrapper)
    ]

    # Assert that a map annotation was added
    assert map_annotations, "Map annotation was not added to the plate."

    # Assert that the added map annotation has the correct key-value pair
    added_annotation = map_annotations[
        -1
    ]  # assuming the last annotation is the one added
    assert (
        added_annotation.getValue() == key_value
    ), "The added map annotation does not have the correct key-value pair."


def test_delete_map_annotations(omero_conn):
    """
    Tests the delete_map_annotations function.
    """
    plate_id = 2  # use actual plate id from your test server

    # Get the Plate object
    plate = omero_conn.getObject("Plate", plate_id)

    # Call the function to delete the map annotations
    delete_map_annotations(plate, omero_conn)
    # Refresh the omero_object
    plate = omero_conn.getObject("plate", plate_id)

    # Get all map annotations of the plate after deletion
    map_annotations_after = [
        ann for ann in plate.listAnnotations() if isinstance(ann, MapAnnotationWrapper)
    ]

    # Assert that map annotations were deleted
    assert not map_annotations_after, "Map annotations were not deleted."
