from pathlib import Path
import omero
from omero.gateway import BlitzGateway

#file_path = Path('./data/metadata.xlsx')

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

def delete_excel_attachments(plate_id, conn=None):
    """
    Delete all Excel file annotations from a plate
    :param plate_id: ID of the plate
    :param conn: omero connection
    :return: none, delete the file annotations
    """
    # Get the Plate object
    plate = conn.getObject("Plate", plate_id)
    if not plate:
        raise ValueError("Plate not found")

    # Get all file annotations
    file_annotations = [ann for ann in plate.listAnnotations()
                        if isinstance(ann, omero.gateway.FileAnnotationWrapper)]

    # Filter Excel file annotations
    excel_annotations = [ann for ann in file_annotations
                         if ann.getFile().getMimetype() == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]

    if not excel_annotations:
        print("No Excel file annotations found on the Plate")
        return

    # Delete Excel file annotations
    delete = omero.cmd.Delete2(targetObjects={'FileAnnotation': [ann.getId() for ann in excel_annotations]})
    handle = conn.c.sf.submit(delete)
    cb = omero.callbacks.CmdCallbackI(conn.c, handle)

    # Wait for the deletion to complete
    while not cb.block(500):  # set a reasonable timeout
        pass

    if cb.getResponse() is not None:
        print("Successfully deleted excel file")
    else:
        print("Unknown error occurred during deletion")

def add_map_annotations(omero_object, key_value, conn=None):
    """
    :param omero_object: omero object to link the map annotation to
    :param key_value: list of tuples with key and value
    :param conn: omero connection
    :return: None and link the map annotation to the omero object
    """
    map_ann = omero.gateway.MapAnnotationWrapper(conn)
    map_ann.setNs(omero.constants.metadata.NSCLIENTMAPANNOTATION)
    map_ann.setValue(key_value)
    map_ann.save()
    omero_object.linkAnnotation(map_ann)

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
    cb = omero.callbacks.CmdCallbackI(conn.c, handle)

    # Wait for the deletion to complete
    while not cb.block(500):  # set a reasonable timeout
        pass



def remove_map_annotations(omero_object):
    """
    Remove all map annotations from an OMERO object.
    """
    map_anns = [ann for ann in omero_object.listAnnotations()
                        if isinstance(ann, omero.gateway.FileAnnotationWrapper)]

    for ann in map_anns:
        omero_object.removeAnnotation(ann)
    # Save the changes to the object.
    omero_object.save()