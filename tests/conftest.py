# conftest.py
import time
from pathlib import Path
import pytest
import omero
from omero.gateway import BlitzGateway

# Other imports needed for your tests...

def delete_map_annotations(omero_object, conn=None):
    """
    :param omero_object: omero object to delete map annotations from
    :param conn: omero connection
    :return: None, and unlink all map annotations from the omero object
    """

    # Get all map annotations of the object
    map_annotations = [ann for ann in omero_object.listAnnotations()
                       if isinstance(ann, omero.gateway.MapAnnotationWrapper)]
    # Delete map annotations

    delete = omero.cmd.Delete2(targetObjects={'MapAnnotation': [ann.getId() for ann in map_annotations]})
    handle = conn.c.sf.submit(delete)
    time.sleep(0.5)
    handle.close()


def attach_excel_file(plate_obj, file_path, omero_conn):
    file = Path(file_path)
    file_size = file.stat().st_size
    file_name = file.name
    with file.open('rb') as file_data:
        original_file = omero_conn.createOriginalFileFromFileObj(
            file_data, path=None, name=file_name, fileSize=file_size,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    file_ann = omero.gateway.FileAnnotationWrapper(omero_conn)
    file_ann.setNs(omero.rtypes.rstring("omero.constants.namespaces.NSAUTOCOMPLETE"))
    file_ann.setFile(original_file)
    file_ann.save()
    plate_obj.linkAnnotation(file_ann)


def load_data(plate_id, filepath, conn=None):
    """
    Load the data from the Excel file attached to the plate.
    :param conn: omero connection
    :param plate_id: ID of the plate
    :return: a dictionary with the channel data and a pandas DataFrame
    with the well data if the Excel file is found. Otherwise return none
    """
    plate = conn.getObject("Plate", plate_id)
    if not plate:
        raise ValueError("Plate not found")

    file = Path(filepath)
    file_size = file.stat().st_size
    file_name = file.name

    # Upload the file
    try:
        # Open and upload the file
        with file.open('rb') as file_data:
            original_file = conn.createOriginalFileFromFileObj(
                file_data, path=None, name=file_name, fileSize=file_size,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return

    # Create a file annotation
    file_ann = omero.gateway.FileAnnotationWrapper(conn)
    file_ann.setNs(omero.rtypes.rstring("omero.constants.namespaces.NSAUTOCOMPLETE"))
    file_ann.setFile(original_file)

    # Save the file annotation
    file_ann.save()
    print(f"Data attached to plate {plate.getId()}")
    # Link the file annotation to the Plate
    plate.linkAnnotation(file_ann)


def delete_excel_attachments(plate_id, conn):
    """
    Delete all Excel file annotations from a plate
    :param plate_id: ID of the plate
    :param conn: omero connection
    :return: none, delete the file annotations
    """
    # Get the Plate object
    plate = conn.getObject("Plate", plate_id)
    file_annotations = [ann for ann in plate.listAnnotations()
                        if isinstance(ann, omero.gateway.FileAnnotationWrapper)]
    excel_annotations = [ann for ann in file_annotations
                         if
                         ann.getFile().getMimetype() == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
    # Delete Excel file annotations

    delete = omero.cmd.Delete2(targetObjects={'FileAnnotation': [ann.getId() for ann in excel_annotations]})
    handle = conn.c.sf.submit(delete)
    time.sleep(0.5)
    handle.close()






def delete_object(conn, object_type, object_id):
    """
    Delete an object of the given type and ID.
    """
    delete = omero.cmd.Delete2(targetObjects={object_type: [object_id]})
    conn.c.sf.submit(delete)


@pytest.fixture(scope='function')
def omero_conn():
    # Create a connection to the omero server
    conn = BlitzGateway(
        'root',
        'omero',
        host='localhost',
        port=4064
    )

    conn.connect()

    print("Setting up connection to OMERO server")
    plate_obj = conn.getObject("Plate", 2)
    delete_map_annotations(plate_obj, conn)
    for well in plate_obj.listChildren():
        delete_map_annotations(well, conn)
    delete_excel_attachments(2, conn)
    yield conn
    delete_excel_attachments(2, conn)
    delete_map_annotations(plate_obj, conn)
    for well in plate_obj.listChildren():
        delete_map_annotations(well, conn)
    # After the tests are done, disconnect
    conn.close()
    print("Closed connection to OMERO server")

